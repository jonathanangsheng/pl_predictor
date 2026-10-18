"""Seed the database with synthetic, pre-scored tweets so the dashboard runs offline.

No API calls and no model download: useful for UI work and for trying the project
before you have X API access.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from datetime import timedelta

from .collector import RawTweet
from .fixtures import Fixture
from .sentiment import SentimentResult
from .storage import Store

_TEMPLATES = {
    "positive": [
        "What a performance from {team} today #{tag}",
        "{team} absolutely flying, love this side",
        "Get in!! {team} deserved that #{tag}",
        "Proud of the lads, {team} on the up",
    ],
    "neutral": [
        "{team} lineup is out #{tag}",
        "Kickoff soon, {team} in the usual shape",
        "Half time stats for {team} #{tag}",
        "Anyone got a stream for {team}?",
    ],
    "negative": [
        "{team} are a shambles, sort it out #{tag}",
        "Same old {team}, no idea going forward",
        "That defending from {team} is embarrassing",
        "Manager out. {team} deserve better #{tag}",
    ],
}


def _mood_curve(rng: random.Random, n_events: int = 3) -> list[tuple[float, float]]:
    """Random 'goal' events: (minute after kickoff, sentiment swing for home side)."""
    return [(rng.uniform(5, 110), rng.choice([-0.55, 0.55])) for _ in range(n_events)]


def _mood(minute: float, base: float, events: list[tuple[float, float]]) -> float:
    mood = base
    for at, swing in events:
        if minute >= at:
            mood += swing * math.exp(-(minute - at) / 45)
    return max(-0.95, min(0.95, mood))


def _probabilities(rng: random.Random, mood: float) -> SentimentResult:
    noisy = max(-0.99, min(0.99, mood + rng.gauss(0, 0.45)))
    positive = max(0.0, noisy) * 0.9 + 0.05
    negative = max(0.0, -noisy) * 0.9 + 0.05
    neutral = max(0.0, 1.0 - positive - negative)
    total = positive + negative + neutral
    return SentimentResult(negative / total, neutral / total, positive / total)


def seed_demo(
    store: Store, fixtures: Sequence[Fixture], tweets_per_fixture: int = 600, seed: int = 7
) -> int:
    rng = random.Random(seed)
    store.upsert_fixtures(fixtures)
    total = 0
    next_id = 1_850_000_000_000_000_000
    for fixture in fixtures:
        events = _mood_curve(rng)
        base_home = rng.uniform(-0.2, 0.3)
        tweets, scores = [], []
        for _ in range(tweets_per_fixture):
            # Activity peaks around the match itself.
            minute = rng.triangular(-120, 235, 60)
            side = rng.choices(["home", "away", "both"], weights=[45, 45, 10])[0]
            home_mood = _mood(minute, base_home, events)
            mood = {"home": home_mood, "away": -home_mood * 0.8, "both": home_mood * 0.2}[side]
            result = _probabilities(rng, mood)
            team = fixture.away if side == "away" else fixture.home
            text = rng.choice(_TEMPLATES[result.label]).format(
                team=team.name, tag=fixture.match_hashtag.lstrip("#")
            )
            if side == "both":
                text += f" vs {fixture.away.name}"
            next_id += rng.randint(1, 10_000)
            tweets.append(
                RawTweet(
                    tweet_id=str(next_id),
                    created_at=fixture.kickoff + timedelta(minutes=minute),
                    text=text,
                    author_id=str(rng.randint(1, 10**9)),
                    lang="en",
                    like_count=int(rng.paretovariate(1.3)) - 1,
                    retweet_count=int(rng.paretovariate(1.8)) - 1,
                    reply_count=int(rng.paretovariate(2.0)) - 1,
                )
            )
            scores.append(result)
        total += store.insert_tweets(fixture, tweets)
        store.save_sentiment([t.tweet_id for t in tweets], scores, model="demo")
    return total
