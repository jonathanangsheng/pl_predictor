"""Team strength: ClubElo ratings (cached), or local Elo from results if ClubElo is unreachable."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .hooks import ClubEloHook
from .matches import CACHE_DIR

log = logging.getLogger(__name__)

HOME_ADVANTAGE = 60
K = 20
START_RATING = 1500.0
LEAGUE_SIZE = 20  # once a full league is rated, newcomers are promoted clubs
CACHE_MAX_AGE_S = 24 * 3600  # ClubElo updates daily; a younger cached history is reused


class EloUnavailable(RuntimeError):
    """ClubElo could not be reached and no cached history exists."""


def local_elo(rows: pd.DataFrame) -> pd.DataFrame:
    """Pre-match ratings for every row (played or upcoming), oldest first.

    World-football-Elo style: K=20 scaled by goal margin, 60-point home advantage. A club seen for
    the first time starts at the mean of the three lowest current ratings (it is almost always
    promoted).
    """
    ratings: dict[str, float] = {}
    home_elo, away_elo = [], []
    ordered = rows.sort_values("date", kind="stable")
    cols = ["home", "away", "result", "home_goals", "away_goals"]
    for home, away, result, home_goals, away_goals in ordered[cols].itertuples(index=False):
        for club in (home, away):
            if club not in ratings:
                lowest = sorted(ratings.values())[:3]
                full = len(ratings) >= LEAGUE_SIZE
                ratings[club] = sum(lowest) / len(lowest) if full else START_RATING
        h, a = ratings[home], ratings[away]
        home_elo.append(h)
        away_elo.append(a)
        if pd.isna(result):
            continue
        expected = 1 / (1 + 10 ** ((a - h - HOME_ADVANTAGE) / 400))
        actual = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        margin = abs(home_goals - away_goals)
        scale = 1 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
        delta = K * scale * (actual - expected)
        ratings[home] = h + delta
        ratings[away] = a - delta
    elo = pd.DataFrame({"home_elo": home_elo, "away_elo": away_elo}, index=ordered.index)
    return elo.loc[rows.index]


def club_history(hook: ClubEloHook, name: str, cache: Path) -> pd.DataFrame:
    """One club's ClubElo history: a cache under a day old, else the API (cache if it fails)."""
    path = cache / "clubelo" / f"{name.replace(' ', '')}.csv"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_S:
        return pd.read_csv(path, parse_dates=["From", "To"])
    try:
        history = hook.club_history(name)
    except requests.RequestException as exc:
        if not path.exists():
            raise EloUnavailable(f"ClubElo unreachable for {name!r} and no cache: {exc}") from exc
        age_h = (time.time() - path.stat().st_mtime) / 3600
        log.warning("ClubElo unreachable for %s (%s); using cache %.0fh old", name, exc, age_h)
        return pd.read_csv(path, parse_dates=["From", "To"])
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False)
    return history


def clubelo_ratings(
    rows: pd.DataFrame, names: pd.DataFrame, hook: ClubEloHook, cache: Path
) -> pd.DataFrame:
    """Each side's ClubElo rating as of the day before the match."""
    to_elo = dict(zip(names["code"], names["clubelo"], strict=True))
    out = pd.DataFrame(index=rows.index, columns=["home_elo", "away_elo"], dtype=float)
    for code in pd.unique(rows[["home", "away"]].values.ravel()):
        if pd.isna(to_elo.get(code)):
            raise EloUnavailable(f"No ClubElo name for {code}; add it to config/club_names.csv")
        history = club_history(hook, to_elo[code], cache).sort_values("From")[["From", "Elo"]]
        for side in ("home", "away"):
            day = rows.loc[rows[side] == code, "date"] - pd.Timedelta(days=1)
            if day.empty:
                continue
            merged = pd.merge_asof(
                day.sort_values().rename("day").reset_index(),
                history,
                left_on="day",
                right_on="From",
                direction="backward",
            )
            out.loc[merged["index"].values, f"{side}_elo"] = merged["Elo"].values
    return out


def elo_ratings(
    rows: pd.DataFrame, names: pd.DataFrame, source: str = "auto"
) -> tuple[pd.DataFrame, str]:
    """Pre-match Elo for each row and the source used ("clubelo" or "local").

    "auto" tries ClubElo and falls back to local Elo for every row if any club is unavailable, so
    one model never mixes the two scales.
    """
    if source in ("auto", "clubelo"):
        try:
            return clubelo_ratings(rows, names, ClubEloHook(timeout=20), CACHE_DIR), "clubelo"
        except EloUnavailable as exc:
            if source == "clubelo":
                raise
            log.warning("Using local Elo: %s", exc)
    return local_elo(rows), "local"
