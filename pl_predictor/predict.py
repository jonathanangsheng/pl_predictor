"""Next-gameweek predictions, the prediction history and live scoring."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .matches import KEY, ODDS, kickoff_keys
from .model import demargin, metrics, proba
from .study import CUTOFF, stage2_inputs

log = logging.getLogger(__name__)

HISTORY = Path("predictions/history.csv")
MODEL_P = ["p_home", "p_draw", "p_away"]
ELO_P = ["elo_home", "elo_draw", "elo_away"]
HISTORY_COLS = (
    KEY
    + ["kickoff", "predicted_at"]
    + MODEL_P
    + ELO_P
    + ["stage", "note", "home_tweets", "away_tweets", "model_version"]
)


class NoUpcomingFixtures(LookupError):
    pass


def upcoming_fixtures(
    teams: pd.DataFrame, fixtures: pd.DataFrame, gameweek: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Unplayed fixtures of the next gameweek (or `gameweek`), plus "home v away" for skipped
    fixtures with no kickoff time. `teams`/`fixtures` come from FPLHook.teams()/fixtures().
    """
    codes = dict(zip(teams["id"], teams["short_name"], strict=True))
    unplayed = fixtures[~fixtures["started"].fillna(False).astype(bool)]
    no_kickoff = unplayed["kickoff_time"].isna()
    unscheduled = [
        f"{codes[h]} v {codes[a]}" for h, a in unplayed.loc[no_kickoff, ["team_h", "team_a"]].values
    ]
    unplayed = unplayed[~no_kickoff & unplayed["event"].notna()]
    if unplayed.empty:
        raise NoUpcomingFixtures("No upcoming fixtures (season over?)")
    if gameweek is None:
        gameweek = int(unplayed["event"].min())
    fx = unplayed[unplayed["event"] == gameweek]
    if fx.empty:
        raise NoUpcomingFixtures(f"No unplayed fixtures in gameweek {gameweek}")
    kickoff = pd.to_datetime(fx["kickoff_time"], utc=True).reset_index(drop=True)
    season, date = kickoff_keys(kickoff)
    rows = pd.DataFrame(
        {
            "season": season,
            "date": date,
            "kickoff": kickoff,
            "home": fx["team_h"].map(codes).values,
            "away": fx["team_a"].map(codes).values,
            "gameweek": gameweek,
        }
    )
    return rows.sort_values(["kickoff", "home"]).reset_index(drop=True), unscheduled


def predict_upcoming(
    bundle: dict,
    rows: pd.DataFrame,
    feats: pd.DataFrame,
    upcoming: pd.Index,
    now: datetime,
    sent: pd.DataFrame | None = None,
    use_stage2: bool = False,
) -> pd.DataFrame:
    """Probabilities for the upcoming rows, in history.csv layout.

    Stage 2 is used for a fixture only when enabled, the cutoff T has passed and a side has
    sentiment.
    """
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    X = feats.loc[upcoming, bundle["features"]]
    out = rows.loc[upcoming, KEY + ["kickoff"]].reset_index(drop=True)
    model_p, elo_p = proba(bundle["stage1"], X), proba(bundle["elo"], X[["elo_diff"]])
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
    out["predicted_at"] = now
    out["model_version"] = bundle["model_version"]
    return out[HISTORY_COLS]


def append_history(pred: pd.DataFrame, path: Path = HISTORY, retries: int = 3) -> None:
    """Append rows; if another program (e.g. Excel) has the file open, retry, then report it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            pred.to_csv(path, mode="a", header=not path.exists(), index=False)
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
    hist = pd.read_csv(path)
    for col in ("kickoff", "predicted_at"):
        hist[col] = pd.to_datetime(hist[col], utc=True, format="ISO8601")
    return hist


def score(history: pd.DataFrame, matches: pd.DataFrame) -> tuple[dict, pd.DataFrame, int]:
    """Score the latest pre-kickoff prediction per fixture.

    Returns (metrics by predictor, scored fixtures, count still unplayed). Model, Elo and market
    are compared on the fixtures that have pre-match odds; "model (all scored)" uses every fixture.
    """
    hist = history[history["predicted_at"] < history["kickoff"]]
    # tail(1), not last(): last() takes the last non-null value per column, mixing rows.
    latest = hist.sort_values("predicted_at").groupby(KEY).tail(1).reset_index(drop=True)
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
