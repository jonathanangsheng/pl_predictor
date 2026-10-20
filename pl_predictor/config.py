"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ConfigError(Exception):
    """A setup problem the user can fix; the CLI prints it without a traceback."""


@dataclass(frozen=True)
class Settings:
    bearer_token: str | None
    db_path: Path
    teams_path: Path
    model_name: str
    pre_match_minutes: int
    post_match_minutes: int
    max_tweets_per_fixture: int
    max_query_length: int
    football_data_org_token: str | None = None
    # Predictor settings (matches/model/predict/study/watch)
    sentiment_since: date | None = None
    stage2: bool = False


def get_settings() -> Settings:
    load_dotenv()
    env = os.environ.get
    return Settings(
        bearer_token=env("X_BEARER_TOKEN") or None,
        db_path=Path(env("PLS_DB_PATH", "data/pl_predictor.db")),
        teams_path=Path(env("PLS_TEAMS_PATH", "config/teams.yaml")),
        model_name=env("PLS_MODEL_NAME", DEFAULT_MODEL),
        pre_match_minutes=int(env("PLS_PRE_MATCH_MINUTES", "120")),
        post_match_minutes=int(env("PLS_POST_MATCH_MINUTES", "120")),
        max_tweets_per_fixture=int(env("PLS_MAX_TWEETS_PER_FIXTURE", "500")),
        max_query_length=int(env("PLS_MAX_QUERY_LENGTH", "512")),
        football_data_org_token=env("FOOTBALL_DATA_ORG_TOKEN") or None,
        sentiment_since=date.fromisoformat(s) if (s := env("PLS_SENTIMENT_SINCE")) else None,
        stage2=env("PLS_STAGE2", "0") == "1",
    )


def require_token(settings: Settings) -> str:
    if not settings.bearer_token:
        raise ConfigError("X_BEARER_TOKEN is not set (see .env.example)")
    return settings.bearer_token
