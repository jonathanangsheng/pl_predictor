"""SQLite persistence for fixtures, tweets and sentiment scores."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .collector import RawTweet
from .fixtures import Fixture
from .sentiment import SentimentResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id  TEXT PRIMARY KEY,
    matchweek   INTEGER NOT NULL,
    kickoff     TEXT NOT NULL,
    full_time   TEXT NOT NULL,
    home_code   TEXT NOT NULL,
    home_name   TEXT NOT NULL,
    away_code   TEXT NOT NULL,
    away_name   TEXT NOT NULL,
    completed   INTEGER NOT NULL DEFAULT 0  -- 1 once the closed window has been searched
);

CREATE TABLE IF NOT EXISTS tweets (
    fixture_id    TEXT NOT NULL REFERENCES fixtures(fixture_id),
    tweet_id      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    text          TEXT NOT NULL,
    author_id     TEXT,
    lang          TEXT,
    like_count    INTEGER NOT NULL DEFAULT 0,
    retweet_count INTEGER NOT NULL DEFAULT 0,
    reply_count   INTEGER NOT NULL DEFAULT 0,
    side          TEXT NOT NULL,
    phase         TEXT NOT NULL,
    collected_at  TEXT NOT NULL,
    PRIMARY KEY (fixture_id, tweet_id)
);
CREATE INDEX IF NOT EXISTS idx_tweets_fixture_time ON tweets(fixture_id, created_at);

CREATE TABLE IF NOT EXISTS sentiment (
    tweet_id   TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    negative   REAL NOT NULL,
    neutral    REAL NOT NULL,
    positive   REAL NOT NULL,
    score      REAL NOT NULL,
    model      TEXT NOT NULL,
    scored_at  TEXT NOT NULL
);
"""


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def upsert_fixtures(self, fixtures: Iterable[Fixture]) -> None:
        rows = [
            (
                f.fixture_id,
                f.matchweek,
                _iso(f.kickoff),
                _iso(f.full_time),
                f.home.code,
                f.home.name,
                f.away.code,
                f.away.name,
            )
            for f in fixtures
        ]
        with self.conn:
            # Upsert rather than REPLACE so the `completed` flag survives a reload.
            self.conn.executemany(
                "INSERT INTO fixtures (fixture_id, matchweek, kickoff, full_time, home_code, "
                "home_name, away_code, away_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(fixture_id) DO UPDATE SET matchweek = excluded.matchweek, "
                "kickoff = excluded.kickoff, full_time = excluded.full_time, "
                "home_code = excluded.home_code, home_name = excluded.home_name, "
                "away_code = excluded.away_code, away_name = excluded.away_name",
                rows,
            )

    def mark_complete(self, fixture_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE fixtures SET completed = 1 WHERE fixture_id = ?", (fixture_id,)
            )

    def is_complete(self, fixture_id: str) -> bool:
        row = self.conn.execute(
            "SELECT completed FROM fixtures WHERE fixture_id = ?", (fixture_id,)
        ).fetchone()
        return bool(row and row[0])

    def insert_tweets(self, fixture: Fixture, tweets: Iterable[RawTweet]) -> int:
        """Store tweets for a fixture, skipping ones already stored. Returns rows added."""
        now = _iso(datetime.now(timezone.utc))
        rows = [
            (
                fixture.fixture_id,
                t.tweet_id,
                _iso(t.created_at),
                t.text,
                t.author_id,
                t.lang,
                t.like_count,
                t.retweet_count,
                t.reply_count,
                fixture.side(t.text),
                fixture.phase(t.created_at),
                now,
            )
            for t in tweets
        ]
        with self.conn:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            return self.conn.total_changes - before

    def latest_tweet_id(self, fixture_id: str) -> str | None:
        # Tweet IDs are snowflakes, so numeric order is chronological order.
        row = self.conn.execute(
            "SELECT tweet_id FROM tweets WHERE fixture_id = ? "
            "ORDER BY CAST(tweet_id AS INTEGER) DESC LIMIT 1",
            (fixture_id,),
        ).fetchone()
        return row[0] if row else None

    def unscored(self, limit: int = 1000) -> list[tuple[str, str]]:
        """(tweet_id, text) pairs that have no sentiment score yet."""
        return self.conn.execute(
            "SELECT DISTINCT t.tweet_id, t.text FROM tweets t "
            "LEFT JOIN sentiment s ON s.tweet_id = t.tweet_id "
            "WHERE s.tweet_id IS NULL LIMIT ?",
            (limit,),
        ).fetchall()

    def save_sentiment(
        self, tweet_ids: Sequence[str], results: Sequence[SentimentResult], model: str
    ) -> None:
        now = _iso(datetime.now(timezone.utc))
        rows = [
            (tid, r.label, r.negative, r.neutral, r.positive, r.score, model, now)
            for tid, r in zip(tweet_ids, results, strict=True)
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO sentiment VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )

    def fixtures_frame(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM fixtures ORDER BY kickoff", self.conn)
        for col in ("kickoff", "full_time"):
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
        return df

    def scored_with(self, model: str) -> bool:
        """Whether any tweet was scored by `model` (e.g. demo.DEMO_MODEL)."""
        sql = "SELECT 1 FROM sentiment WHERE model = ? LIMIT 1"
        return self.conn.execute(sql, (model,)).fetchone() is not None

    def scored_fixture_ids(self) -> set[str]:
        sql = (
            "SELECT DISTINCT t.fixture_id FROM tweets t JOIN sentiment s ON s.tweet_id = t.tweet_id"
        )
        return {r[0] for r in self.conn.execute(sql)}

    def tweets_frame(self) -> pd.DataFrame:
        """Scored tweets joined with their sentiment."""
        sql = (
            "SELECT t.*, s.label, s.negative, s.neutral, s.positive, s.score, s.model "
            "FROM tweets t JOIN sentiment s ON s.tweet_id = t.tweet_id ORDER BY t.created_at"
        )
        df = pd.read_sql_query(sql, self.conn)
        for col in ("created_at", "collected_at"):
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
        return df
