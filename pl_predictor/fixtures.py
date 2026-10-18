"""Fixtures: loading, collection windows, match phases and X search queries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from pathlib import Path

from .teams import Team

# Kickoff to final whistle including half-time and stoppage time, roughly.
MATCH_DURATION = timedelta(minutes=115)
# Which tweet `side` values count for each club: a tweet naming both clubs counts for each.
SIDE_TWEETS = {"home": ("home", "both"), "away": ("away", "both")}


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    matchweek: int
    kickoff: datetime
    home: Team
    away: Team

    @property
    def label(self) -> str:
        return f"{self.home.name} v {self.away.name}"

    @property
    def match_hashtag(self) -> str:
        return f"#{self.home.code}{self.away.code}"

    @property
    def full_time(self) -> datetime:
        return self.kickoff + MATCH_DURATION

    def window(self, pre_minutes: int, post_minutes: int) -> tuple[datetime, datetime]:
        """Time range in which tweets about this fixture are collected."""
        return (
            self.kickoff - timedelta(minutes=pre_minutes),
            self.full_time + timedelta(minutes=post_minutes),
        )

    def phase(self, ts: datetime) -> str:
        if ts < self.kickoff:
            return "pre"
        if ts <= self.full_time:
            return "live"
        return "post"

    def side(self, text: str) -> str:
        """Which club a tweet is about: home, away, both or neither."""
        home = self.home.is_mentioned(text)
        away = self.away.is_mentioned(text)
        if home and away:
            return "both"
        if home:
            return "home"
        if away:
            return "away"
        return "neither"

    def build_query(self, max_length: int = 512) -> str:
        """Build an X recent-search query that fits within the tier's length limit.

        Terms are interleaved home/away so both clubs stay represented when the
        query has to be trimmed; lower-priority terms are dropped first.
        """
        suffix = " -is:retweet lang:en"
        candidates = [self.match_hashtag]
        for h, a in zip_longest(self.home.query_terms(), self.away.query_terms()):
            candidates += [t for t in (h, a) if t]

        terms: list[str] = []
        for term in candidates:
            query = f"({' OR '.join([*terms, term])}){suffix}"
            if len(query) > max_length:
                break
            terms.append(term)
        if not terms:
            raise ValueError(f"max_length={max_length} is too short for any search term")
        return f"({' OR '.join(terms)}){suffix}"


def _parse_kickoff(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def load_fixtures(path: str | Path, teams: dict[str, Team]) -> list[Fixture]:
    """Load fixtures.csv (fixture_id, matchweek, kickoff_utc, home, away)."""
    fixtures = []
    # utf-8-sig: Excel saves CSVs with a byte-order mark.
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            home, away = row["home"].strip().upper(), row["away"].strip().upper()
            missing = {home, away} - teams.keys()
            if missing:
                raise KeyError(f"{row['fixture_id']}: unknown team code(s) {sorted(missing)}")
            fixtures.append(
                Fixture(
                    fixture_id=row["fixture_id"].strip(),
                    matchweek=int(row["matchweek"]),
                    kickoff=_parse_kickoff(row["kickoff_utc"].strip()),
                    home=teams[home],
                    away=teams[away],
                )
            )
    return sorted(fixtures, key=lambda f: f.kickoff)
