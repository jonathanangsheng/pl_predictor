"""Stage-2 study: does pre-match fan sentiment add information beyond stage 1 and the market?"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

from .fixtures import SIDE_TWEETS
from .matches import KEY, ODDS, kickoff_keys
from .model import FEATURES, demargin, per_match_log_loss, proba, stage1, stage1_proba, trainable

log = logging.getLogger(__name__)

CUTOFF = pd.Timedelta(minutes=60)  # predictions are made at T = kickoff - 60 min
WINDOW = pd.Timedelta(hours=24)  # tweets counted: kickoff - 24h <= created_at < T
FIRST_RUN_SLACK = pd.Timedelta(minutes=5)  # allowed lead of the first collection before T
SENTIMENT = ["home_net", "away_net", "net_diff"]
MIN_FIRST_FOLD = 25  # stage 2 needs at least this many earlier fixtures to train on


def sentiment_features(
    tweets: pd.DataFrame,
    db_fixtures: pd.DataFrame,
    targets: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per fixture: net sentiment per side, tweet counts, and whether collection timing was valid.

    `tweets`/`db_fixtures` come from Store.tweets_frame()/fixtures_frame() (scored tweets only).
    `targets` has KEY + kickoff (UTC) from the match table or FPL.
    """
    out = pd.DataFrame(
        {
            "home_net": np.nan,
            "away_net": np.nan,
            "home_n": 0,
            "away_n": 0,
            "timing_ok": False,
            "note": "no sentiment",
        },
        index=targets.index,
    )
    if not db_fixtures.empty:
        season = kickoff_keys(db_fixtures["kickoff"])[0]
        db = db_fixtures.assign(
            season=season, home=db_fixtures["home_code"], away=db_fixtures["away_code"]
        )
        matched = (
            targets[KEY + ["kickoff"]]
            .reset_index(names="target")
            .merge(
                db[KEY + ["fixture_id", "kickoff"]].rename(columns={"kickoff": "db_kickoff"}),
                on=KEY,
            )
        )
        matched = matched[matched["kickoff"].notna()]
        moved = matched["kickoff"] != matched["db_kickoff"]
        for r in matched[moved].itertuples():
            log.warning(
                "%s v %s: sentiment DB kickoff %s != %s; ignoring its tweets",
                r.home,
                r.away,
                r.db_kickoff,
                r.kickoff,
            )
        out.loc[matched.loc[moved, "target"], "note"] = "kickoff mismatch"
        matched = matched[~moved]
        own = tweets.merge(matched[["fixture_id", "target", "kickoff"]], on="fixture_id")
        cutoff = own["kickoff"] - CUTOFF
        if now is not None:  # live: only tweets that exist by now
            cutoff = cutoff.clip(upper=now)
        window = own[(own["created_at"] >= own["kickoff"] - WINDOW) & (own["created_at"] < cutoff)]
        for side, sides in SIDE_TWEETS.items():
            scores = window[window["side"].isin(sides)].groupby("target")["score"]
            out.loc[scores.mean().index, f"{side}_net"] = scores.mean()
            out.loc[scores.size().index, f"{side}_n"] = scores.size()
        # Later runs only fill from the newest tweet back, so the first collection must follow T.
        first = own.groupby("target").agg(
            collected=("collected_at", "min"), kickoff=("kickoff", "first")
        )
        out.loc[first.index, "timing_ok"] = (
            first["collected"] >= first["kickoff"] - CUTOFF - FIRST_RUN_SLACK
        )
        found = matched["target"]
        has_tweets = (out.loc[found, "home_n"] + out.loc[found, "away_n"]) > 0
        out.loc[found, "note"] = np.where(has_tweets, "", "no sentiment")
    out["net_diff"] = out["home_net"] - out["away_net"]
    return out


def eligible(rows: pd.DataFrame, sent: pd.DataFrame, since: date | None) -> pd.Series:
    """Fixtures usable for the study: played, on/after `since`, tweets on a side, valid timing."""
    if since is None:
        return pd.Series(False, index=rows.index)
    return (
        rows["result"].notna()
        & (rows["date"].dt.date >= since)
        & ((sent["home_n"] + sent["away_n"]) > 0)
        & sent["timing_ok"].astype(bool)
    )


def stage2_model():
    return make_pipeline(SimpleImputer(add_indicator=True), LogisticRegression(max_iter=2000))


def stage2_inputs(base_p: np.ndarray, sent: pd.DataFrame) -> pd.DataFrame:
    """Base log-odds of home and draw against away, plus the sentiment columns."""
    clipped = np.clip(base_p, 1e-6, 1)
    return pd.DataFrame(
        {
            "lo_home": np.log(clipped[:, 0] / clipped[:, 2]),
            "lo_draw": np.log(clipped[:, 1] / clipped[:, 2]),
            **{c: sent[c].astype(float).to_numpy() for c in SENTIMENT},
        },
        index=sent.index,
    )


def oof_stage1(
    rows: pd.DataFrame, feats: pd.DataFrame, idx: pd.Index, folds: list[np.ndarray]
) -> np.ndarray:
    """Stage-1 probabilities per eligible fixture from a model fitted only on earlier matches."""
    out = np.full((len(idx), 3), np.nan)
    # The training-only block before the first fold, then each fold.
    blocks = [np.arange(folds[0][0])] + folds if len(folds) else []
    for block in blocks:
        if not len(block):
            continue
        start = rows.loc[idx[block[0]], "date"]
        train = trainable(rows) & (rows["date"] < start)
        model = stage1().fit(feats.loc[train, FEATURES], rows.loc[train, "result"])
        out[block] = stage1_proba(model, rows, feats, idx[block])
    return out


def run_arm(
    name: str,
    y: pd.Series,
    base_p: np.ndarray,
    sent: pd.DataFrame,
    folds: list[np.ndarray],
    seed: int = 0,
) -> dict:
    """Held-out log-loss change from adding sentiment to `base_p`, with a bootstrap 95% CI."""
    X = stage2_inputs(base_p, sent)
    diffs = []
    for fold in folds:
        train = np.arange(fold[0])
        model = stage2_model().fit(X.iloc[train], y.iloc[train])
        y_fold = y.iloc[fold]
        diffs.append(
            per_match_log_loss(y_fold, proba(model, X.iloc[fold]))
            - per_match_log_loss(y_fold, base_p[fold])
        )
    deltas = np.concatenate(diffs) if diffs else np.array([])
    mean = low = high = np.nan
    if len(deltas):
        rng = np.random.default_rng(seed)
        boot = rng.choice(deltas, size=(1000, len(deltas)), replace=True).mean(axis=1)
        mean, (low, high) = float(deltas.mean()), np.percentile(boot, [2.5, 97.5])
    return {
        "arm": name,
        "n": len(deltas),
        "mean_change": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "passed": bool(high < 0),
    }


def fixtures_needed(per_match_sd: float, effects=(0.002, 0.005, 0.01)) -> dict[float, int]:
    """Fixtures for 80% power (two-sided 5%) to detect each mean log-loss improvement."""
    z = 1.959964 + 0.841621  # z for two-sided 5% significance + z for 80% power
    return {e: int(np.ceil((z * per_match_sd / e) ** 2)) for e in effects}


def run_study(
    rows: pd.DataFrame,
    feats: pd.DataFrame,
    sent: pd.DataFrame,
    since: date | None,
    per_match_sd: float,
) -> tuple[pd.DataFrame, dict, pd.Index, np.ndarray | None]:
    """Both arms on the same eligible fixtures.

    Returns (arm results, power table, eligible index, out-of-fold stage-1 probabilities).
    """
    idx = rows.index[eligible(rows, sent, since)]
    idx = idx[np.argsort(rows.loc[idx, "kickoff"].to_numpy())]  # folds must be in time order
    power = fixtures_needed(per_match_sd)
    if len(idx) < MIN_FIRST_FOLD * 2:
        log.info("%d eligible fixtures; the study needs more data", len(idx))
        return pd.DataFrame(), power, idx, None
    folds = [test for _, test in TimeSeriesSplit(5).split(idx)]
    if folds[0][0] < MIN_FIRST_FOLD:
        return pd.DataFrame(), power, idx, None
    y = rows.loc[idx, "result"].reset_index(drop=True)
    sentiment = sent.loc[idx].reset_index(drop=True)
    base = oof_stage1(rows, feats, idx, folds)
    market = demargin(rows.loc[idx, ODDS["avg"]])
    arms = pd.DataFrame(
        [
            run_arm("stage 1 + sentiment vs stage 1", y, base, sentiment, folds),
            run_arm("market + sentiment vs market", y, market, sentiment, folds),
        ]
    )
    return arms, power, idx, base


def fit_live_stage2(rows: pd.DataFrame, sent: pd.DataFrame, idx: pd.Index, base: np.ndarray):
    """Live stage-2 model: fitted on all eligible fixtures with out-of-fold stage-1 inputs."""
    keep = ~np.isnan(base).any(axis=1)
    X = stage2_inputs(base[keep], sent.loc[idx[keep]].reset_index(drop=True))
    return stage2_model().fit(X, rows.loc[idx[keep], "result"].to_numpy())
