"""Pull tweets for a fixture from the X API v2 recent-search endpoint via Tweepy."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import tweepy

log = logging.getLogger(__name__)

# Recent search only reaches back 7 days, and end_time must be >= 10s in the past.
# The 20-min margin keeps start_time valid when Tweepy retries after a 15-min rate-limit wait.
SEARCH_LOOKBACK = timedelta(days=7) - timedelta(minutes=20)
END_TIME_BUFFER = timedelta(seconds=15)
TWEET_FIELDS = ["created_at", "lang", "author_id", "public_metrics"]


@dataclass(frozen=True)
class RawTweet:
    tweet_id: str
    created_at: datetime
    text: str
    author_id: str | None
    lang: str | None
    like_count: int
    retweet_count: int
    reply_count: int

    @classmethod
    def from_tweepy(cls, tweet: tweepy.Tweet) -> RawTweet:
        metrics = tweet.public_metrics or {}
        return cls(
            tweet_id=str(tweet.id),
            created_at=tweet.created_at,
            text=tweet.text,
            author_id=str(tweet.author_id) if tweet.author_id else None,
            lang=tweet.lang,
            like_count=metrics.get("like_count", 0),
            retweet_count=metrics.get("retweet_count", 0),
            reply_count=metrics.get("reply_count", 0),
        )


class TweetCollector:
    def __init__(self, client: tweepy.Client):
        self.client = client

    @classmethod
    def from_bearer_token(cls, bearer_token: str) -> TweetCollector:
        return cls(tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True))

    def search(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int,
        since_id: str | None = None,
        now: datetime | None = None,
    ) -> Iterator[RawTweet]:
        """Yield up to `limit` tweets matching `query` between start_time and end_time.

        The range is clamped to what recent search allows. Passing `since_id`
        (the newest tweet already stored) makes repeated runs incremental.
        """
        now = now or datetime.now(timezone.utc)
        start = max(start_time, now - SEARCH_LOOKBACK)
        end = min(end_time, now - END_TIME_BUFFER)
        if start >= end or limit <= 0:
            return

        params = dict(
            query=query,
            end_time=end,
            max_results=min(100, max(10, limit)),
            tweet_fields=TWEET_FIELDS,
        )
        # The API rejects start_time together with a since_id older than it, and
        # since_id alone already bounds the range from below.
        if since_id:
            params["since_id"] = since_id
        else:
            params["start_time"] = start

        log.debug("search_recent_tweets %s", params)
        paginator = tweepy.Paginator(self.client.search_recent_tweets, **params)
        for tweet in paginator.flatten(limit=limit):
            yield RawTweet.from_tweepy(tweet)
