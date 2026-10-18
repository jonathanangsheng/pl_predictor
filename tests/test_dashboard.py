import plotly.graph_objects as go
import pytest

from pl_predictor import figures
from pl_predictor.dashboard import create_app
from pl_predictor.demo import seed_demo


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
        figures.club_ranking(tweets, fx),
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
