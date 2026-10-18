"""Match table: one row per Premier League match since 2005/06, built from the data hooks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .hooks import FootballDataCoUkHook, FootballDataOrgHook

log = logging.getLogger(__name__)

FIRST_SEASON = 2005
LONDON = "Europe/London"
CLUB_NAMES = Path("config/club_names.csv")
CACHE_DIR = Path("data/cache")
KEY = ["season", "home", "away"]  # one fixture per pairing per season, even if rescheduled
# Odds columns: avg_* = average pre-match odds, close_* = closing odds; h/d/a = home/draw/away.
ODDS = {kind: [f"{kind}_{k}" for k in "hda"] for kind in ("avg", "close")}

# football-data.co.uk stat suffix -> match-table name; the file has an H- and A- column for each.
STATS = {"S": "shots", "ST": "shots_on_target", "C": "corners", "F": "fouls"}
REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] + [
    f"{side}{col}" for col in STATS for side in "HA"
]
# Odds column sets in order of preference; names changed in 2019/20.
PRE_MATCH_ODDS = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")]
CLOSING_ODDS = [("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA")]


class SchemaError(ValueError):
    """A source file is missing columns the match table needs."""


class UnmappedTeamError(KeyError):
    """A source uses a club name with no row in config/club_names.csv."""


def season_of(ts: pd.Timestamp | datetime) -> int:
    """Start year of the season containing `ts` (Europe/London date): Aug-Jul."""
    local = pd.Timestamp(ts)
    local = local.tz_convert(LONDON) if local.tzinfo else local
    return local.year if local.month >= 8 else local.year - 1


def kickoff_keys(kickoff: pd.Series) -> tuple[pd.Series, pd.Series]:
    """UTC kickoffs -> (season, naive Europe/London date): how every source is keyed."""
    local = kickoff.dt.tz_convert(LONDON)
    season = local.dt.year - (local.dt.month < 8).astype(int)
    return season, local.dt.tz_localize(None).dt.normalize()


def club_names() -> pd.DataFrame:
    return pd.read_csv(CLUB_NAMES, dtype=str)


def to_codes(names: pd.Series, source: str, table: pd.DataFrame) -> pd.Series:
    """Map a source's club names to canonical codes; unknown names raise, never drop."""
    mapping = dict(zip(table[source], table["code"], strict=True))
    missing = sorted(set(names.dropna()) - mapping.keys())
    if missing:
        raise UnmappedTeamError(f"{source}: no code for {missing}; add them to {CLUB_NAMES}")
    return names.map(mapping)


def _pick_odds(raw: pd.DataFrame, options: list[tuple[str, str, str]]) -> pd.DataFrame:
    """The first (home, draw, away) odds columns present in `raw`, else all NaN."""
    for cols in options:
        if set(cols) <= set(raw.columns):
            return (
                raw[list(cols)]
                .apply(pd.to_numeric, errors="coerce")
                .set_axis(["h", "d", "a"], axis=1)
            )
    return pd.DataFrame(float("nan"), index=raw.index, columns=["h", "d", "a"])


def normalise(raw: pd.DataFrame, season: int, names: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Turn one football-data.co.uk season file into match-table rows."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise SchemaError(f"{label or season}: missing columns {missing}")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    out = pd.DataFrame(
        {
            "season": season,
            "date": pd.to_datetime(raw["Date"]).dt.normalize(),
            "home": to_codes(raw["HomeTeam"], "fdcouk", names).values,
            "away": to_codes(raw["AwayTeam"], "fdcouk", names).values,
            "home_goals": pd.to_numeric(raw["FTHG"]).values,
            "away_goals": pd.to_numeric(raw["FTAG"]).values,
            "result": raw["FTR"].values,
        },
        index=raw.index,
    )
    if "Time" in raw.columns:  # UK local kickoff time, from 2019/20
        local = pd.to_datetime(
            raw["Date"].dt.strftime("%Y-%m-%d") + " " + raw["Time"].astype(str), errors="coerce"
        )
        out["kickoff"] = local.dt.tz_localize(
            LONDON, ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
    else:
        out["kickoff"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    for col, name in STATS.items():
        out[f"home_{name}"] = pd.to_numeric(raw[f"H{col}"], errors="coerce")
        out[f"away_{name}"] = pd.to_numeric(raw[f"A{col}"], errors="coerce")
    for prefix, options in (("avg", PRE_MATCH_ODDS), ("close", CLOSING_ODDS)):
        out[ODDS[prefix]] = _pick_odds(raw, options).to_numpy()
    return out.reset_index(drop=True)


def _season_file(hook: FootballDataCoUkHook, season: int, current: int) -> pd.DataFrame:
    """Raw season file; completed seasons are cached because they never change."""
    path = CACHE_DIR / "fdcouk" / f"E0_{season}.csv"
    if season < current and path.exists():
        return pd.read_csv(path, parse_dates=["Date"])
    raw = hook.results(season)
    if season < current:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(path, index=False)
    return raw


def load_matches(
    now: datetime | None = None,
    season_started: bool = True,
    fdorg: FootballDataOrgHook | None = None,
) -> pd.DataFrame:
    """All played matches, oldest first, keyed by KEY."""
    now = now or datetime.now(timezone.utc)
    hook = FootballDataCoUkHook()
    names = club_names()
    current = season_of(now)
    frames = []
    for season in range(FIRST_SEASON, current + 1):
        try:
            raw = _season_file(hook, season, current)
        except requests.HTTPError as exc:
            missing_file = exc.response is not None and exc.response.status_code == 404
            if season == current and missing_file and not season_started:
                log.warning("No %s/%s file yet; season has not started", season, season + 1)
                continue
            raise
        frames.append(normalise(raw, season, names, f"E0 {season}"))
    table = pd.concat(frames, ignore_index=True)
    if fdorg is not None:
        table = fill_from_fdorg(table, fdorg, current, names)
    log.info("Match table: %d matches, newest %s", len(table), table["date"].max().date())
    table = table.sort_values(["date", "kickoff", "home"], na_position="first")
    return table.reset_index(drop=True)


def fill_from_fdorg(
    table: pd.DataFrame, hook: FootballDataOrgHook, season: int, names: pd.DataFrame
) -> pd.DataFrame:
    """Add finished current-season matches football-data.co.uk lacks so far (goals only).

    The fill is optional: any API, mapping or shape error warns and leaves the table unchanged.
    """
    try:
        raw = hook.matches(season=season)
        done = raw[raw["status"] == "FINISHED"] if "status" in raw else raw.iloc[0:0]
        if done.empty:
            return table
        kickoff = pd.to_datetime(done["utcDate"], utc=True)
        rows = pd.DataFrame(
            {
                "season": season,
                "date": kickoff_keys(kickoff)[1],
                "kickoff": kickoff,
                "home": to_codes(done["homeTeam.tla"], "fdorg", names).values,
                "away": to_codes(done["awayTeam.tla"], "fdorg", names).values,
                "home_goals": done["score.fullTime.home"].values,
                "away_goals": done["score.fullTime.away"].values,
            }
        )
    # UnmappedTeamError is a KeyError.
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning(
            "football-data.org fill skipped (%s); newest match %s", exc, table["date"].max().date()
        )
        return table
    rows["result"] = np.sign(rows["home_goals"] - rows["away_goals"]).map({1: "H", 0: "D", -1: "A"})
    new = rows[~pd.MultiIndex.from_frame(rows[KEY]).isin(pd.MultiIndex.from_frame(table[KEY]))]
    if len(new):
        log.info("football-data.org added %d matches not yet on football-data.co.uk", len(new))
    return pd.concat([table, new], ignore_index=True)
