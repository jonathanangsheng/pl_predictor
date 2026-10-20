"""Next-gameweek predictions, the prediction history and live scoring."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import dixon_coles
from .matches import KEY, ODDS, fpl_matches, kickoff_keys
from .model import demargin, explain, metrics, proba, stage1_proba
from .study import CUTOFF, stage2_inputs

log = logging.getLogger(__name__)

HISTORY = Path("predictions/history.csv")
MODEL_P = ["p_home", "p_draw", "p_away"]
ELO_P = ["elo_home", "elo_draw", "elo_away"]
MARKET_P = ["mkt_home", "mkt_draw", "mkt_away"]  # de-margined pre-match odds when predicted
HISTORY_COLS = (
    KEY
    + ["kickoff", "predicted_at"]
    + MODEL_P
    + ELO_P
    + MARKET_P
    + ["stage", "note", "why", "home_tweets", "away_tweets", "model_version"]
)


class NoUpcomingFixtures(LookupError):
    pass


def latest_per_fixture(history: pd.DataFrame) -> pd.DataFrame:
    """Each fixture's most recent prediction row."""
    # tail(1), not last(): last() takes the last non-null value per column, mixing rows.
    return history.sort_values("predicted_at").groupby(KEY).tail(1)


def summary_table(summary: dict) -> pd.DataFrame:
    """score()'s metrics as one row per predictor."""
    return pd.DataFrame(summary).T[["n", "log_loss", "brier", "accuracy"]].round(4)


def upcoming_fixtures(
    teams: pd.DataFrame, fixtures: pd.DataFrame, gameweek: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Unplayed fixtures of the next gameweek (or `gameweek`), plus "home v away" for skipped
    fixtures with no kickoff time. `teams`/`fixtures` come from FPLHook.teams()/fixtures().
    """
    matches = fpl_matches(teams, fixtures)
    unplayed = matches[~matches["started"]]
    no_kickoff = unplayed["kickoff"].isna()
    unscheduled = [f"{h} v {a}" for h, a in unplayed.loc[no_kickoff, ["home", "away"]].values]
    unplayed = unplayed[~no_kickoff & unplayed["gameweek"].notna()]
    if unplayed.empty:
        raise NoUpcomingFixtures("No upcoming fixtures (season over?)")
    if gameweek is None:
        gameweek = int(unplayed["gameweek"].min())
    fx = unplayed[unplayed["gameweek"] == gameweek].reset_index(drop=True)
    if fx.empty:
        raise NoUpcomingFixtures(f"No unplayed fixtures in gameweek {gameweek}")
    season, date = kickoff_keys(fx["kickoff"])
    rows = pd.DataFrame(
        {
            "season": season,
            "date": date,
            "kickoff": fx["kickoff"],
            "home": fx["home"],
            "away": fx["away"],
            "gameweek": gameweek,
        }
    )
    return rows.sort_values(["kickoff", "home"]).reset_index(drop=True), unscheduled


def _why(pushes: pd.Series, goals: np.ndarray, home: str, away: str, top: int = 3) -> str:
    """One line on what drives a stage-1 prediction: the biggest logistic pushes, Dixon-Coles xG."""
    biggest = pushes[pushes.abs().nlargest(top).index]
    listed = ", ".join(f"{label} {value:+.2f}" for label, value in biggest.items())
    xg = f"expected goals {goals[0]:.1f}-{goals[1]:.1f}"
    return f"+ favours {home}, - favours {away}: {listed} · {xg}"


def predict_upcoming(
    bundle: dict,
    rows: pd.DataFrame,
    feats: pd.DataFrame,
    upcoming: pd.Index,
    now: datetime,
    sent: pd.DataFrame | None = None,
    use_stage2: bool = False,
    odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Probabilities for the upcoming rows, in history.csv layout.

    `odds` is matches.upcoming_odds(); fixtures it doesn't list get NaN market probabilities.

    Stage 2 is used for a fixture only when enabled, the cutoff T has passed and a side has
    sentiment.
    """
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    X = feats.loc[upcoming, bundle["features"]]
    out = rows.loc[upcoming, KEY + ["kickoff"]].reset_index(drop=True)
    model_p = stage1_proba(bundle["stage1"], rows, feats, upcoming, bundle["features"])
    elo_p = proba(bundle["elo"], X[["elo_diff"]])
    out["stage"], out["note"] = 1, ""
    out["home_tweets"], out["away_tweets"] = np.nan, np.nan
    if sent is not None:
        sentiment = sent.loc[upcoming].reset_index(drop=True)
        out["home_tweets"] = sentiment["home_n"].values
        out["away_tweets"] = sentiment["away_n"].values
        out["note"] = sentiment["note"].values
        if use_stage2 and bundle.get("stage2") is not None:
            has_tweets = (sentiment["home_n"] + sentiment["away_n"]) > 0
            ready = ((now >= out["kickoff"] - CUTOFF) & has_tweets).to_numpy()
            if ready.any():
                model_p = model_p.copy()
                inputs = stage2_inputs(model_p[ready], sentiment.loc[ready])
                model_p[ready] = proba(bundle["stage2"], inputs)
                out.loc[ready, "stage"] = 2
            out.loc[~ready & (out["note"] == ""), "note"] = "stage 1: before cutoff"
    out[MODEL_P] = model_p
    out[ELO_P] = elo_p
    pushes, goals = explain(bundle["stage1"], X), dixon_coles.predict_goals(rows, upcoming)
    out["why"] = [
        _why(pushes.iloc[i], goals[i], home, away)
        for i, (home, away) in enumerate(zip(out["home"], out["away"], strict=True))
    ]
    stage2 = out["stage"] == 2
    out.loc[stage2, "why"] = "Sentiment-adjusted (stage 2). Stage 1: " + out.loc[stage2, "why"]
    out[MARKET_P] = np.nan
    if odds is not None:
        listed = out[KEY].merge(odds, on=KEY, how="left")
        out[MARKET_P] = demargin(listed[ODDS["avg"]])
    out["predicted_at"] = now
    out["model_version"] = bundle["model_version"]
    return out[HISTORY_COLS]


def append_history(pred: pd.DataFrame, path: Path = HISTORY, retries: int = 3) -> None:
    """Append rows; if another program (e.g. Excel) has the file open, retry, then report it.

    Appending never rereads the file, so concurrent writers (watch + a manual predict) don't
    lose rows. A file from before a column was added is rewritten once in HISTORY_COLS order,
    through a temp file so a crash mid-write can't truncate it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred.reindex(columns=HISTORY_COLS)
    for attempt in range(retries + 1):
        try:
            header = None
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    header = fh.readline().strip()
            if header in (None, ",".join(HISTORY_COLS)):
                pred.to_csv(path, mode="a", header=header is None, index=False)
            else:
                # shortcut: a writer racing this one-off migration can lose its rows.
                rows = pd.concat([pd.read_csv(path), pred]).reindex(columns=HISTORY_COLS)
                tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
                rows.to_csv(tmp, index=False)
                os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries:
                raise PermissionError(
                    f"Cannot write {path}: close it in any other program and retry"
                ) from None
            time.sleep(1)


def read_history(path: Path = HISTORY) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    hist = pd.read_csv(path).reindex(columns=HISTORY_COLS)  # files from before a column was added
    for col in ("kickoff", "predicted_at"):
        hist[col] = pd.to_datetime(hist[col], utc=True, format="ISO8601")
    return hist


def score(history: pd.DataFrame, matches: pd.DataFrame) -> tuple[dict, pd.DataFrame, int]:
    """Score the latest pre-kickoff prediction per fixture.

    Returns (metrics by predictor, scored fixtures, count still unplayed). Model, Elo and market
    are compared on the fixtures that have pre-match odds; "model (all scored)" uses every fixture.
    """
    hist = history[history["predicted_at"] < history["kickoff"]]
    latest = latest_per_fixture(hist).reset_index(drop=True)
    joined = latest.merge(
        matches[KEY + ["result"] + ODDS["avg"] + ODDS["close"]],
        on=KEY,
        how="left",
    )
    pending = int(joined["result"].isna().sum())
    done = joined[joined["result"].notna()].reset_index(drop=True)
    model_p = done[MODEL_P].to_numpy(float)
    elo_p = done[ELO_P].to_numpy(float)
    market_p = demargin(done[ODDS["avg"]])
    closing_p = demargin(done[ODDS["close"]])
    three_way = ~np.isnan(market_p).any(axis=1)
    summary = {
        "model (all scored)": metrics(done["result"], model_p),
        "model": metrics(done["result"][three_way], model_p[three_way]),
        "elo": metrics(done["result"][three_way], elo_p[three_way]),
        "market (pre-match odds)": metrics(done["result"][three_way], market_p[three_way]),
        "closing odds (ceiling)": metrics(done["result"], closing_p),
    }
    return summary, done, pending
