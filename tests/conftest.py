from datetime import datetime, timezone
from pathlib import Path

import pytest

from pl_predictor.fixtures import Fixture, fpl_fixture
from pl_predictor.storage import Store
from pl_predictor.teams import load_teams

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def teams():
    return load_teams(ROOT / "config" / "teams.yaml")


@pytest.fixture(scope="session")
def fixtures(teams):
    """A few fixtures as `pl-predictor` builds them from FPL (the tests stay offline)."""
    ko = datetime(2026, 10, 17, 11, 30, tzinfo=timezone.utc)
    pairs = [("EVE", "CHE"), ("BRE", "LIV"), ("FUL", "HUL"), ("NFO", "ARS")]
    return [fpl_fixture(7, ko, home, away, teams) for home, away in pairs]


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
