from datetime import datetime, timezone
from pathlib import Path

import pytest

from pl_predictor.fixtures import Fixture, load_fixtures
from pl_predictor.storage import Store
from pl_predictor.teams import load_teams

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def teams():
    return load_teams(ROOT / "config" / "teams.yaml")


@pytest.fixture(scope="session")
def fixtures(teams):
    return load_fixtures(ROOT / "config" / "fixtures.csv", teams)


@pytest.fixture
def ars_mci(teams) -> Fixture:
    return Fixture(
        fixture_id="test-ars-mci",
        matchweek=8,
        kickoff=datetime(2026, 10, 17, 11, 30, tzinfo=timezone.utc),
        home=teams["ARS"],
        away=teams["MCI"],
    )


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()
