from datetime import datetime, timedelta
from types import SimpleNamespace

import requests
import tweepy

from pl_predictor.collector import RawTweet, TweetCollector
from pl_predictor.config import Settings
from pl_predictor.pipeline import collect_fixture, collectable_fixtures, score_pending
from pl_predictor.sentiment import SentimentResult


def _settings(**overrides) -> Settings:
    base = dict(
        bearer_token="x",
        db_path=":memory:",
        fixtures_path="",
        teams_path="",
        model_name="fake",
        pre_match_minutes=120,
        post_match_minutes=120,
        max_tweets_per_fixture=50,
        max_query_length=512,
    )
    return Settings(**{**base, **overrides})


def _tweet(i: int, ts: datetime, text: str) -> tweepy.Tweet:
    return tweepy.Tweet(
        {
            # Same shape the API returns, e.g. 2026-10-17T11:30:00.000Z
            "id": str(1000 + i),
            "text": text,
            "created_at": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "edit_history_tweet_ids": [str(1000 + i)],
            "lang": "en",
            "author_id": "42",
            "public_metrics": {"like_count": i, "retweet_count": 0, "reply_count": 0},
        }
    )


class FakeClient:
    """Mimics tweepy.Client.search_recent_tweets with two pages of results."""

    def __init__(self, kickoff: datetime):
        self.calls = []
        texts = ["#Arsenal brilliant", "Man City awful", "Arsenal v Man City", "#ARSMCI"]
        self.pages = [
            [_tweet(i, kickoff + timedelta(minutes=10 * i), texts[i % 4]) for i in range(3)],
            [_tweet(i, kickoff + timedelta(minutes=10 * i), texts[i % 4]) for i in range(3, 5)],
        ]

    def search_recent_tweets(self, **kwargs):
        self.calls.append(dict(kwargs))
        token = kwargs.get("next_token")
        page = int(token.removeprefix("page")) if token else 0
        meta = {"result_count": len(self.pages[page])}
        if page + 1 < len(self.pages):
            meta["next_token"] = f"page{page + 1}"
        return tweepy.Response(data=self.pages[page], includes={}, errors=[], meta=meta)


def test_collectable_fixtures_respects_lookback(ars_mci):
    settings = _settings()
    assert collectable_fixtures([ars_mci], settings, now=ars_mci.kickoff + timedelta(hours=1))
    assert collectable_fixtures([ars_mci], settings, now=ars_mci.kickoff + timedelta(days=6))
    assert not collectable_fixtures([ars_mci], settings, now=ars_mci.kickoff + timedelta(days=8))
    assert not collectable_fixtures([ars_mci], settings, now=ars_mci.kickoff - timedelta(hours=3))


def test_search_clamps_end_time_and_paginates(ars_mci):
    client = FakeClient(ars_mci.kickoff)
    collector = TweetCollector(client)
    now = ars_mci.kickoff + timedelta(minutes=30)
    tweets = list(collector.search("q", ars_mci.kickoff, ars_mci.full_time, limit=50, now=now))

    assert len(tweets) == 5
    assert isinstance(tweets[0], RawTweet) and tweets[1].like_count == 1
    assert client.calls[0]["end_time"] < now
    assert client.calls[0]["start_time"] == ars_mci.kickoff
    assert client.calls[1]["next_token"] == "page1"


def test_search_uses_since_id_instead_of_start_time(ars_mci):
    client = FakeClient(ars_mci.kickoff)
    list(
        TweetCollector(client).search(
            "q",
            ars_mci.kickoff,
            ars_mci.full_time,
            limit=50,
            since_id="999",
            now=ars_mci.kickoff + timedelta(days=1),
        )
    )
    assert client.calls[0]["since_id"] == "999"
    assert "start_time" not in client.calls[0]


def test_search_skips_future_window(ars_mci):
    client = FakeClient(ars_mci.kickoff)
    now = ars_mci.kickoff - timedelta(days=1)
    collector = TweetCollector(client)
    assert list(collector.search("q", ars_mci.kickoff, ars_mci.full_time, 50, now=now)) == []
    assert client.calls == []


def test_collect_and_score_end_to_end(ars_mci, store, monkeypatch):
    store.upsert_fixtures([ars_mci])
    client = FakeClient(ars_mci.kickoff)
    collector = TweetCollector(client)
    # Pretend "now" is after the match so the whole window is searchable.
    real_search = collector.search
    monkeypatch.setattr(
        collector,
        "search",
        lambda *a, **k: real_search(*a, now=ars_mci.full_time + timedelta(hours=3), **k),
    )

    settings = _settings()
    assert collect_fixture(ars_mci, collector, store, settings) == 5
    assert store.latest_tweet_id(ars_mci.fixture_id) == "1004"

    scorer = SimpleNamespace(
        model_name="fake",
        score=lambda texts: [SentimentResult(0.1, 0.2, 0.7) for _ in texts],
    )
    assert score_pending(store, scorer, chunk_size=2) == 5
    assert score_pending(store, scorer) == 0

    df = store.tweets_frame()
    assert len(df) == 5
    assert set(df["side"]) == {"home", "away", "both", "neither"}
    assert set(df["phase"]) == {"live"}

    # Re-inserting the same tweets is a no-op.
    assert collect_fixture(ars_mci, collector, store, settings) == 0


def test_first_run_spreads_budget_survives_errors_then_completes(ars_mci, store):
    calls = []

    def search(query, start_time, end_time, limit, since_id=None):
        calls.append((start_time, end_time, limit, since_id))
        if len(calls) == 3:
            raise requests.ConnectionError("dropped")
        yield RawTweet(str(len(calls)), start_time, "#Arsenal", None, "en", 0, 0, 0)

    collector = SimpleNamespace(search=search)
    store.upsert_fixtures([ars_mci])
    settings = _settings(max_tweets_per_fixture=80)
    start, end = ars_mci.window(settings.pre_match_minutes, settings.post_match_minutes)
    after = end + timedelta(hours=1)

    # 8 slices of 10; the failing third slice doesn't stop the rest.
    assert collect_fixture(ars_mci, collector, store, settings, now=after) == 7
    assert len(calls) == 8 and {c[2] for c in calls} == {10}
    assert calls[0][0] == start and calls[1][0] == calls[0][1] and calls[-1][1] == end
    assert not store.is_complete(ars_mci.fixture_id)  # retried next run because of the error

    # Next run is incremental and, with the window closed, marks the fixture complete...
    collect_fixture(ars_mci, collector, store, settings, now=after)
    assert calls[-1][3] == "8" and store.is_complete(ars_mci.fixture_id)
    # ...so later runs spend no requests on it, even after the fixture list is reloaded.
    store.upsert_fixtures([ars_mci])
    assert collect_fixture(ars_mci, collector, store, settings, now=after) == 0
    assert len(calls) == 9


def test_small_budget_is_not_exceeded(ars_mci, store):
    limits = []

    def search(query, start_time, end_time, limit, since_id=None):
        limits.append(limit)
        return iter(())

    store.upsert_fixtures([ars_mci])
    settings = _settings(max_tweets_per_fixture=25)
    collect_fixture(ars_mci, SimpleNamespace(search=search), store, settings)
    assert sum(limits) <= 25 and min(limits) >= 10
