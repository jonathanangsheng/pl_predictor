from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pl_predictor import pipeline
from pl_predictor.config import ConfigError, get_settings
from pl_predictor.fixtures import fpl_fixture


def test_teams_config_loads(teams):
    assert len(teams) == 20 and all(t.colour.startswith("#") for t in teams.values())


@pytest.mark.parametrize(
    ("text", "side"),
    [
        ("COYG what a goal #Arsenal", "home"),
        ("@ManCity were poor today", "away"),
        ("Arsenal v Man City is the game of the season", "both"),
        ("Great day at #ARSMCI", "neither"),
        ("Gunners all day", "home"),
        ("MAN CITY!!!", "away"),
        # Word boundaries: substrings of other words must not match.
        ("#ArsenalWomen win again", "neither"),
    ],
)
def test_side_attribution(ars_mci, text, side):
    assert ars_mci.side(text) == side


def test_phase(ars_mci):
    assert ars_mci.phase(ars_mci.kickoff - timedelta(minutes=1)) == "pre"
    assert ars_mci.phase(ars_mci.kickoff + timedelta(minutes=60)) == "live"
    assert ars_mci.phase(ars_mci.full_time + timedelta(minutes=1)) == "post"


def test_query_contains_both_clubs(ars_mci):
    query = ars_mci.build_query(512)
    assert query.startswith("(#ARSMCI OR #Arsenal OR #MCFC")
    assert query.endswith("-is:retweet lang:en")
    assert '"Man City"' in query
    assert len(query) <= 512


def test_query_is_trimmed_to_limit(ars_mci):
    query = ars_mci.build_query(80)
    assert len(query) <= 80
    assert "#ARSMCI" in query and "#MCFC" in query


def test_query_too_short_raises(ars_mci):
    with pytest.raises(ValueError):
        ars_mci.build_query(10)


def test_fpl_fixture_id_and_unknown_club(teams):
    ko = datetime(2026, 10, 24, 14, tzinfo=timezone.utc)
    assert fpl_fixture(8, ko, "ARS", "EVE", teams).fixture_id == "mw08-ars-eve"
    with pytest.raises(ConfigError, match="ZZZ"):
        fpl_fixture(8, ko, "ARS", "ZZZ", teams)


def test_season_fixtures_from_fpl(monkeypatch):
    class FakeFPL:
        def teams(self):
            return pd.DataFrame({"id": [1, 2], "short_name": ["ARS", "EVE"]})

        def fixtures(self):
            return pd.DataFrame(
                {
                    "event": [9, 8, None],
                    "team_h": [2, 1, 1],
                    "team_a": [1, 2, 2],
                    "kickoff_time": pd.to_datetime(
                        ["2026-10-31 15:00", "2026-10-24 14:00", None], utc=True
                    ),
                    "team_h_score": None,
                    "team_a_score": None,
                    "finished": False,
                    "started": False,
                }
            )

    monkeypatch.setattr(pipeline, "FPLHook", FakeFPL)
    fixtures = pipeline.season_fixtures(get_settings())
    assert [f.fixture_id for f in fixtures] == [
        "mw08-ars-eve",
        "mw09-eve-ars",
    ]  # unscheduled dropped
    now = datetime(2026, 10, 25, tzinfo=timezone.utc)
    assert [f.fixture_id for f in pipeline.near(fixtures, days=3, now=now)] == ["mw08-ars-eve"]
