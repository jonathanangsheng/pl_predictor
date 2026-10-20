import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pl_predictor import figures
from pl_predictor.dashboard import create_app
from pl_predictor.demo import seed_demo
from pl_predictor.matches import fpl_matches


@pytest.fixture
def seeded(store, fixtures):
    seed_demo(store, fixtures, tweets_per_fixture=200)
    return store.fixtures_frame(), store.tweets_frame()


def test_demo_seeds_every_fixture(seeded, fixtures):
    fx, tweets = seeded
    assert len(fx) == len(fixtures)
    assert tweets["fixture_id"].nunique() == len(fixtures)
    assert tweets["score"].between(-1, 1).all()


def test_figures_render(seeded):
    fx, tweets = seeded
    fixture = fx.iloc[0]
    df = tweets[tweets["fixture_id"] == fixture["fixture_id"]]

    for fig in (
        figures.sentiment_timeline(df, fixture, 10),
        figures.volume_timeline(df, fixture, 10),
        figures.sentiment_mix(df, fixture),
    ):
        assert isinstance(fig, go.Figure) and fig.data

    assert len(figures.kpis(df, fixture)) == 4
    assert 0 < len(figures.top_tweets(df, fixture)) <= 15
    clubs = figures.club_table(tweets, fx)
    assert len(clubs) == len(set(fx["home_name"]) | set(fx["away_name"]))


def test_figures_handle_empty(seeded):
    fx, tweets = seeded
    fixture = fx.iloc[0]
    empty = tweets.iloc[0:0]
    assert not figures.sentiment_timeline(empty, fixture, 10).data
    assert figures.top_tweets(empty, fixture) == []
    assert figures.kpis(empty, fixture)[0][1] == "0"


def test_app_builds(tmp_path):
    app = create_app(tmp_path / "db.sqlite")
    assert app.layout is not None


def test_upcoming_predictions_keeps_latest_future_row():
    ko = pd.Timestamp("2026-10-24 14:00", tz="UTC")
    hist = pd.DataFrame(
        {
            "season": 2026,
            "home": ["ARS", "ARS", "CHE"],
            "away": ["MCI", "MCI", "LIV"],
            "kickoff": [ko, ko, ko - pd.Timedelta(days=3)],
            "predicted_at": [ko - pd.Timedelta(days=2), ko - pd.Timedelta(hours=1), ko],
            "p_home": [0.3, 0.5, 0.4],
            "p_draw": 0.25,
            "p_away": [0.45, 0.25, 0.35],
            "elo_home": 0.4,
            "elo_draw": 0.3,
            "elo_away": 0.3,
            "mkt_home": [0.45, np.nan, 0.4],
            "why": "Elo +0.40",
            "mkt_draw": 0.3,
            "mkt_away": 0.3,
        }
    )
    rows = figures.upcoming_predictions(hist, ko - pd.Timedelta(days=1))
    assert [(r["fixture"], r["p_home"], r["mkt_home"]) for r in rows] == [("ARS v MCI", 50, None)]


def test_track_record_blanks_missing_metrics():
    rows = figures.track_record(
        {
            "model": {"n": 2, "log_loss": 0.95, "brier": 0.6, "accuracy": 0.5},
            "market (pre-match odds)": {
                "n": 0,
                "log_loss": np.nan,
                "brier": np.nan,
                "accuracy": np.nan,
            },
        }
    )
    assert rows[1]["predictor"] == "market (pre-match odds)" and rows[1]["log_loss"] is None


def _fpl_matches():
    ko = pd.Timestamp("2026-10-24 14:00", tz="UTC")
    # NEW: in FPL's list but not yet played (and in no teams.yaml); its game was postponed.
    teams = pd.DataFrame({"id": [1, 2, 3, 4], "short_name": ["ARS", "CHE", "TOT", "NEW"]})
    fixtures = pd.DataFrame(
        {
            "event": [1, 2, 3, 4, 4],
            "team_h": [1, 2, 3, 1, 4],
            "team_a": [2, 3, 1, 3, 2],
            "kickoff_time": [ko - pd.Timedelta(days=d) for d in (14, 7, 1)]
            + [ko, ko - pd.Timedelta(hours=1)],
            "team_h_score": [2.0, 1.0, 0.0, None, None],
            "team_a_score": [0.0, 1.0, 3.0, None, None],
            "finished": [True, True, True, False, False],
            "started": [True, True, True, False, False],
        }
    )
    return fpl_matches(teams, fixtures), ko


def test_league_table_points_and_order():
    matches, _ = _fpl_matches()
    table = figures.league_table(matches)
    assert [(r["club"], r["played"], r["points"]) for r in table] == [
        ("ARS", 2, 6),
        ("CHE", 2, 1),
        ("TOT", 2, 1),
        ("NEW", 0, 0),
    ]
    assert table[1]["gd"] == -2 and [r["pos"] for r in table] == [1, 2, 3, 4]


def test_next_match_skips_played_fixtures():
    matches, ko = _fpl_matches()
    match = figures.next_match(matches, "TOT", ko - pd.Timedelta(hours=1))
    assert (match["home"], match["away"]) == ("ARS", "TOT")
    assert figures.next_match(matches, "CHE", ko) is None


def test_central_tiles_from_demo_data(seeded, teams):
    fx, tweets = seeded
    previews = figures.fixture_previews(tweets, fx, n=2)
    assert len(previews) == 2 and previews[0]["tweets"] > 0
    top = figures.top_tweets_overall(tweets, fx)
    assert len(top) == 3 and top[0]["likes"] >= top[-1]["likes"]
    fig = figures.mood_map(tweets, fx, teams)
    assert len(fig.data) == 3 and len(fig.data[2].lat) > 0


def test_slow_results_download_does_not_block_fpl(monkeypatch):
    import threading

    from pl_predictor import dashboard

    monkeypatch.setattr(dashboard, "_fpl_cached", lambda bucket: "fpl")
    got = []
    with dashboard._RESULTS_LOCK:  # a results download in progress
        worker = threading.Thread(target=lambda: got.append(dashboard._fpl()))
        worker.start()
        worker.join(timeout=2)
    assert got == ["fpl"]


def test_bad_results_file_becomes_a_message(monkeypatch):
    from pl_predictor import dashboard
    from pl_predictor.matches import SchemaError

    def broken():
        raise SchemaError("E0 2026: missing columns ['HS']")

    monkeypatch.setattr(dashboard, "_results", broken)
    history = pd.DataFrame({"kickoff": [pd.Timestamp("2026-10-24", tz="UTC")]})
    assert "missing columns" in dashboard._scored(history)


def test_default_fixture_needs_scored_tweets(store, ars_mci):
    from datetime import timedelta

    from pl_predictor.collector import RawTweet

    store.upsert_fixtures([ars_mci])
    tweet = RawTweet("1", ars_mci.kickoff + timedelta(minutes=5), "#Arsenal", None, "en", 0, 0, 0)
    store.insert_tweets(ars_mci, [tweet])
    assert store.scored_fixture_ids() == set()


def test_xg_table_recent_form_and_season_difference():
    matches = pd.DataFrame(
        {
            "season": [2025, 2026, 2026, 2026],
            "date": pd.to_datetime(["2026-03-01", "2026-08-16", "2026-08-23", "2026-08-30"]),
            "home": ["ARS", "ARS", "CHE", "ARS"],
            "away": ["CHE", "CHE", "ARS", "TOT"],
            "home_xg": [9.0, 2.0, 1.0, np.nan],  # last season and a missing value are ignored
            "away_xg": [9.0, 0.5, 1.5, np.nan],
        }
    )
    xg = figures.xg_table(matches, last=1)
    assert xg.loc["ARS", "xgd"] == pytest.approx(2.0 + 1.5 - 0.5 - 1.0)
    assert xg.loc["ARS", ["xg_for", "xg_against"]].tolist() == [1.5, 1.0]  # last match only
    assert "TOT" not in xg.index
