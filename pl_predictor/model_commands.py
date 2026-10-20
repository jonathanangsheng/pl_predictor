"""The `pl-predictor model ...` commands. `watch` calls the same functions."""

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import model
from .config import Settings
from .hooks import FootballDataCoUkHook, FootballDataOrgHook, FPLHook
from .matches import KEY, SOURCE_ERRORS, club_names, load_matches, upcoming_odds
from .predict import (
    MARKET_P,
    MODEL_P,
    append_history,
    predict_upcoming,
    read_history,
    score,
    summary_table,
    upcoming_fixtures,
)
from .storage import Store
from .study import fit_live_stage2, run_study, sentiment_features

log = logging.getLogger(__name__)


def load_results(settings: Settings, now: datetime, season_started: bool) -> pd.DataFrame:
    """Match table, topped up from football-data.org when a token is set."""
    fdorg = None
    if settings.football_data_org_token:
        fdorg = FootballDataOrgHook(token=settings.football_data_org_token)
    else:
        log.info("No FOOTBALL_DATA_ORG_TOKEN: results come from football-data.co.uk only")
    return load_matches(now=now, season_started=season_started, fdorg=fdorg)


def _dataset(
    settings: Settings,
    fixtures: pd.DataFrame,
    now: datetime,
    upcoming=None,
):
    """Match table (+ upcoming rows) and features; `fixtures` is FPLHook.fixtures() for this run."""
    started = bool(fixtures["started"].fillna(False).astype(bool).any())
    matches = load_results(settings, now, started)
    rows, feats = model.dataset(matches, upcoming)
    return matches, rows, feats


@functools.lru_cache(maxsize=1)
def _market_odds_cached(_hour: int) -> pd.DataFrame:
    return upcoming_odds(FootballDataCoUkHook().upcoming(), club_names())


def _market_odds() -> pd.DataFrame | None:
    """Pre-match odds for the listed upcoming fixtures; optional, so a failure only warns.

    Reused for an hour: `watch` predicts once per kickoff slot, and the file changes midweek.
    """
    try:
        return _market_odds_cached(int(time.time() // 3600))
    except SOURCE_ERRORS as exc:
        log.warning("Market odds skipped (%s)", exc)
        return None


def _sentiment(settings: Settings, targets: pd.DataFrame, now=None) -> pd.DataFrame | None:
    if not settings.db_path.exists():
        log.warning("No sentiment database at %s: stage 1 only", settings.db_path)
        return None
    store = Store(settings.db_path)
    try:
        return sentiment_features(store.tweets_frame(), store.fixtures_frame(), targets, now)
    finally:
        store.close()


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(df.to_string())


def evaluate(settings: Settings) -> None:
    _, rows, feats = _dataset(settings, FPLHook().fixtures(), datetime.now(timezone.utc))
    report, y, probs, _ = model.walk_forward(rows, feats)
    print("Walk-forward evaluation. Lower log loss and Brier are better.\n")
    _print(report.pivot(index="predictor", columns="season", values="log_loss").round(4))
    print("\nMean over seasons (accuracy is secondary):")
    _print(model.summarise(report).round(4))
    print("\nModel calibration, pooled held-out predictions:")
    _print(model.calibration_table(y, probs).round(3))


def train(settings: Settings) -> None:
    _, rows, feats = _dataset(settings, FPLHook().fixtures(), datetime.now(timezone.utc))
    bundle = model.train(rows, feats)
    model.save(bundle)
    print(
        f"Saved {model.MODEL_PATH}: version {bundle['model_version']}, "
        f"{bundle['trained_on']} matches"
    )


def predict(
    settings: Settings,
    gameweek: int | None = None,
    only: set[tuple] | None = None,
    now: datetime | None = None,
) -> None:
    """Predict the next (or given) gameweek; `only` limits it to fixture keys (used by watch)."""
    now = now or datetime.now(timezone.utc)
    bundle = model.load()
    fpl = FPLHook()
    fixtures = fpl.fixtures()
    upcoming, unscheduled = upcoming_fixtures(fpl.teams(), fixtures, gameweek)
    if unscheduled:
        log.warning("Unscheduled (no kickoff time): %s", ", ".join(unscheduled))
    if only is not None:
        upcoming = upcoming[pd.MultiIndex.from_frame(upcoming[KEY]).isin(list(only))]
    matches, rows, feats = _dataset(settings, fixtures, now, upcoming.reset_index(drop=True))
    upcoming_idx = rows.index[len(matches) :]
    sent = _sentiment(settings, rows.loc[upcoming_idx, KEY + ["kickoff"]], pd.Timestamp(now))
    pred = predict_upcoming(
        bundle, rows, feats, upcoming_idx, now, sent, settings.stage2, _market_odds()
    )
    cols = ["kickoff", *MODEL_P, *MARKET_P, "stage", "note", "home_tweets", "away_tweets"]
    shown = pred[KEY + cols]
    _print(shown.assign(kickoff=shown["kickoff"].dt.strftime("%a %d %b %H:%M UTC")).round(3))
    append_history(pred)


def score_history(settings: Settings) -> None:
    hist = read_history()
    if hist.empty:
        print("No logged predictions yet: run `pl-predictor model predict` first")
        return
    matches, *_ = _dataset(settings, FPLHook().fixtures(), datetime.now(timezone.utc))
    summary, done, pending = score(hist, matches)
    print(f"Scored fixtures: {len(done)}  pending (not played yet): {pending}\n")
    _print(summary_table(summary))


def study(settings: Settings) -> None:
    now = datetime.now(timezone.utc)
    _, rows, feats = _dataset(settings, FPLHook().fixtures(), now)
    # Spread of per-match log-loss differences (model vs Elo) on the last season sizes the study.
    _, y, model_p, elo_p = model.walk_forward(rows, feats, eval_seasons=1)
    per_match = model.per_match_log_loss(y, model_p) - model.per_match_log_loss(y, elo_p)
    sent = _sentiment(settings, rows[KEY + ["kickoff"]])
    if sent is None:
        sent = pd.DataFrame({"home_n": 0, "away_n": 0, "timing_ok": False}, index=rows.index)
    arms, power, eligible_idx, base = run_study(
        rows, feats, sent, settings.sentiment_since, float(np.std(per_match))
    )
    since = settings.sentiment_since or "PLS_SENTIMENT_SINCE unset"
    print(f"Eligible fixtures: {len(eligible_idx)} (since {since})")
    print(
        "Fixtures needed for 80% power: "
        + ", ".join(f"{n} to detect {e}" for e, n in power.items())
    )
    if arms.empty:
        print("Not enough eligible fixtures to run the test yet.")
        return
    _print(arms.round(4))
    if bool(arms.loc[arms["arm"].str.startswith("stage 1"), "passed"].iloc[0]):
        bundle = model.load()
        bundle["stage2"] = fit_live_stage2(rows, sent, eligible_idx, base)
        model.save(bundle)
        print("Stage 2 passed and was saved; enable it with PLS_STAGE2=1.")
