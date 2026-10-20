"""`pl-predictor watch`: at kickoff - 55 min per fixture, collect and score tweets, then predict."""

from __future__ import annotations

import functools
import logging
import os
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import requests

from .config import LOG_FORMAT, Settings, require_token
from .hooks import FPLHook
from .predict import KEY, NoUpcomingFixtures, upcoming_fixtures

log = logging.getLogger(__name__)

SLOT = timedelta(minutes=55)  # collect and predict from kickoff - 55 min (just after the cutoff)
REFRESH = timedelta(minutes=15)  # how often FPL fixtures are re-read
LOCK = Path("data/watch.lock")
LOG_FILE = Path("logs/watch.log")


class AlreadyRunning(RuntimeError):
    pass


def acquire_lock(path: Path = LOCK) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise AlreadyRunning(
            f"{path} exists: another watch is running (delete it if not)"
        ) from None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)


def due(fixtures: pd.DataFrame, now: datetime, handled: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fixtures whose slot is open now, fixtures whose kickoff passed without collection)."""
    unhandled = ~pd.MultiIndex.from_frame(fixtures[KEY]).isin(list(handled))
    kickoff = fixtures["kickoff"]
    ready = unhandled & (kickoff - SLOT <= now) & (now < kickoff)
    missed = unhandled & (kickoff <= now)
    return fixtures[ready], fixtures[missed]


def vanished(old: pd.DataFrame | None, new: pd.DataFrame, handled) -> list[tuple[str, str]]:
    """(home, away) of unhandled fixtures gone from the FPL list (started or postponed)."""
    if old is None:
        return []
    old_keys = pd.MultiIndex.from_frame(old[KEY])
    gone = ~old_keys.isin(pd.MultiIndex.from_frame(new[KEY])) & ~old_keys.isin(list(handled))
    return list(zip(old.loc[gone, "home"], old.loc[gone, "away"], strict=True))


# Built once per watch process: the scorer loads a ~500 MB transformer model.
@functools.cache
def _teams(path: Path):
    from .teams import load_teams

    return load_teams(path)


@functools.cache
def _collector(token: str):
    from .collector import TweetCollector

    return TweetCollector.from_bearer_token(token)


@functools.cache
def _scorer(model_name: str):
    from .sentiment import SentimentScorer

    return SentimentScorer(model_name)


def collect_fixture(settings: Settings, row: pd.Series, now: datetime) -> int:
    """Upsert one FPL fixture into the sentiment DB, collect its tweets and score them."""
    from .fixtures import fpl_fixture
    from .pipeline import collect_fixture as collect
    from .pipeline import score_pending
    from .storage import Store

    kickoff = row["kickoff"].to_pydatetime()
    fixture = fpl_fixture(
        int(row["gameweek"]), kickoff, row["home"], row["away"], _teams(settings.teams_path)
    )
    store = Store(settings.db_path)
    store.conn.execute("PRAGMA busy_timeout = 30000")  # another reader/writer: wait, don't fail
    try:
        store.upsert_fixtures([fixture])
        added = collect(fixture, _collector(settings.bearer_token), store, settings, now)
        score_pending(store, _scorer(settings.model_name))
        return added
    finally:
        store.close()


def run(
    settings: Settings,
    once: bool = False,
    dry_run: bool = False,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    collect: Callable = collect_fixture,
    predict: Callable | None = None,
    fpl: FPLHook | None = None,
    lock: Path = LOCK,
) -> None:
    """Loop until stopped (one pass with `once`); `dry_run` lists the next slots only."""
    if predict is None:
        from .model_commands import predict as predict_cmd

        predict = lambda keys, now: predict_cmd(settings, only=keys, now=now)  # noqa: E731
    fpl = fpl or FPLHook()
    if not dry_run and collect is collect_fixture:
        # Fail now, not per fixture after its X reads are spent (test fakes need neither).
        from .sentiment import require_transformers

        require_token(settings)
        require_transformers()
    if not dry_run:
        acquire_lock(lock)
    handled: set = set()  # keys collected and predicted, missed or failed: never retried
    pending: dict = {}  # collected but not yet predicted: key -> kickoff
    fixtures, refreshed = None, None
    try:
        while True:
            now = clock()
            if fixtures is None or now - refreshed >= REFRESH:
                try:
                    fresh, _ = upcoming_fixtures(fpl.teams(), fpl.fixtures())
                except NoUpcomingFixtures:
                    log.info("No upcoming fixtures; watch stops")
                    return
                except requests.RequestException as exc:
                    log.warning("FPL refresh failed (%s); retrying next loop", exc)
                else:
                    for home, away in vanished(fixtures, fresh, handled | pending.keys()):
                        log.warning(
                            "MISSED %s v %s: started or postponed before collection "
                            "(not eligible for the study)",
                            home,
                            away,
                        )
                    if fixtures is None and not fresh.empty:
                        first = fresh.iloc[0]  # sorted by kickoff
                        log.info(
                            "Watching %d fixtures; next: %s v %s, collect at %s UTC",
                            len(fresh),
                            first["home"],
                            first["away"],
                            f"{first['kickoff'] - SLOT:%a %d %b %H:%M}",
                        )
                    fixtures, refreshed = fresh, now
            if fixtures is None:  # no successful FPL refresh yet
                if once or dry_run:
                    return
                time.sleep(60)
                continue
            if dry_run:
                for _, r in fixtures.iterrows():
                    print(
                        f"{r['home']} v {r['away']}: kickoff {r['kickoff']:%a %d %b %H:%M} UTC, "
                        f"collect at {r['kickoff'] - SLOT:%H:%M} UTC"
                    )
                return
            ready, missed = due(fixtures, now, handled | pending.keys())
            for _, fixture in missed.iterrows():
                handled.add(tuple(fixture[KEY]))
                log.warning(
                    "MISSED %s v %s: slot passed without collection (not eligible for the study)",
                    fixture["home"],
                    fixture["away"],
                )
            for key in [k for k, kickoff in pending.items() if now >= kickoff]:
                log.warning("No prediction for %s v %s before kickoff", key[1], key[2])
                handled.add(key)
                del pending[key]
            for _, fixture in ready.iterrows():
                key = tuple(fixture[KEY])
                try:
                    log.info("COLLECTING %s v %s", fixture["home"], fixture["away"])
                    added = collect(settings, fixture, now)
                    log.info(
                        "SCORED %s v %s: %d new tweets", fixture["home"], fixture["away"], added
                    )
                    pending[key] = fixture["kickoff"]
                except sqlite3.OperationalError as exc:
                    log.warning(
                        "Database busy for %s v %s (%s); retrying next loop",
                        fixture["home"],
                        fixture["away"],
                        exc,
                    )
                except Exception:  # one fixture must not stop the others
                    log.exception("FAILED %s v %s", fixture["home"], fixture["away"])
                    handled.add(key)
            if pending:  # one prediction run for the slot; on failure retried next loop
                keys = set(pending)
                try:
                    predict(keys, now)
                    log.info("PREDICTED %s", ", ".join(f"{h} v {a}" for _, h, a in sorted(keys)))
                    handled |= keys
                    pending.clear()
                except Exception:
                    log.exception("FAILED prediction for %s; retrying next loop", sorted(keys))
            if once:
                return
            time.sleep(60)
    finally:
        if not dry_run:
            lock.unlink(missing_ok=True)


def file_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
