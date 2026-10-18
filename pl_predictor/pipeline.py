"""Glue: pick active fixtures, collect their tweets, score anything unscored."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone

import requests
import tweepy

from .collector import END_TIME_BUFFER, SEARCH_LOOKBACK, TweetCollector
from .config import Settings
from .fixtures import Fixture, load_fixtures
from .sentiment import SentimentScorer
from .storage import Store
from .teams import load_teams

log = logging.getLogger(__name__)

FIRST_RUN_SLICES = 8


def load_configured_fixtures(settings: Settings) -> list[Fixture]:
    return load_fixtures(settings.fixtures_path, load_teams(settings.teams_path))


def collectable_fixtures(
    fixtures: Iterable[Fixture], settings: Settings, now: datetime | None = None
) -> list[Fixture]:
    """Fixtures whose collection window overlaps what recent search can still reach."""
    now = now or datetime.now(timezone.utc)
    earliest = now - SEARCH_LOOKBACK
    active = []
    for f in fixtures:
        start, end = f.window(settings.pre_match_minutes, settings.post_match_minutes)
        if start < now and end > earliest:
            active.append(f)
    return active


def collect_fixture(
    fixture: Fixture,
    collector: TweetCollector,
    store: Store,
    settings: Settings,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(timezone.utc)
    start, end = fixture.window(settings.pre_match_minutes, settings.post_match_minutes)
    if store.is_complete(fixture.fixture_id):
        log.debug("%s: already collected through full window", fixture.fixture_id)
        return 0
    query = fixture.build_query(settings.max_query_length)
    since_id = store.latest_tweet_id(fixture.fixture_id)
    budget = settings.max_tweets_per_fixture
    if since_id:
        ranges, per_range = [(start, end)], budget
    else:
        # Recent search returns newest first, so one capped query would only cover the
        # end of the window. Spread the first run's budget evenly across it instead.
        # The API returns at least 10 tweets per request, so small budgets get fewer slices.
        n = min(FIRST_RUN_SLICES, max(1, budget // 10))
        step = (end - start) / n
        ranges = [(start + i * step, start + (i + 1) * step) for i in range(n)]
        per_range = budget // n

    log.info("Collecting %s: %s", fixture.fixture_id, query)
    tweets, failed = [], False
    for lo, hi in ranges:
        try:
            for tweet in collector.search(query, lo, hi, per_range, since_id=since_id):
                tweets.append(tweet)
        except (tweepy.TweepyException, requests.RequestException):
            # Keep what was already read (it counts against the quota) and try the next slice.
            log.exception("%s: search failed after %d tweets", fixture.fixture_id, len(tweets))
            failed = True
    added = store.insert_tweets(fixture, tweets)
    # Recent search returns newest first, so once the window has closed and been searched,
    # later runs can't find anything new. Stop spending requests on it.
    if not failed and now - END_TIME_BUFFER >= end:
        store.mark_complete(fixture.fixture_id)
    log.info("%s: %d new tweets", fixture.fixture_id, added)
    return added


def score_pending(store: Store, scorer: SentimentScorer, chunk_size: int = 512) -> int:
    total = 0
    while batch := store.unscored(limit=chunk_size):
        ids, texts = zip(*batch, strict=True)
        store.save_sentiment(ids, scorer.score(texts), model=scorer.model_name)
        total += len(batch)
        log.info("Scored %d tweets", total)
    return total


def run_once(
    settings: Settings,
    store: Store,
    collector: TweetCollector,
    scorer: SentimentScorer,
    fixture_ids: Iterable[str] | None = None,
) -> tuple[int, int]:
    """One collect + score pass. Returns (tweets collected, tweets scored)."""
    fixtures = load_configured_fixtures(settings)
    store.upsert_fixtures(fixtures)
    if fixture_ids:
        wanted = set(fixture_ids)
        targets = [f for f in fixtures if f.fixture_id in wanted]
        unknown = wanted - {f.fixture_id for f in targets}
        if unknown:
            raise KeyError(f"Unknown fixture id(s): {sorted(unknown)}")
    else:
        targets = collectable_fixtures(fixtures, settings)
        if not targets:
            log.warning("No fixtures inside the 7-day recent-search window")

    collected = sum(collect_fixture(f, collector, store, settings) for f in targets)
    return collected, score_pending(store, scorer)
