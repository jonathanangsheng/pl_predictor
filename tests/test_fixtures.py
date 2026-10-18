from datetime import timedelta

import pytest


def test_config_files_load(teams, fixtures):
    assert len(teams) == 20
    assert fixtures and all(f.home.code != f.away.code for f in fixtures)
    assert fixtures == sorted(fixtures, key=lambda f: f.kickoff)


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


def test_csv_saved_with_bom_loads(tmp_path, teams):
    from pl_predictor.fixtures import load_fixtures

    path = tmp_path / "fixtures.csv"
    path.write_text(
        "fixture_id,matchweek,kickoff_utc,home,away\nx,1,2026-10-17T11:30:00Z,ARS,MCI\n",
        encoding="utf-8-sig",
    )
    assert load_fixtures(path, teams)[0].fixture_id == "x"
