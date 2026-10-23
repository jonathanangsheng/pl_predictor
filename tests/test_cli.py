import sys

import pytest

from pl_predictor.cli import main
from pl_predictor.config import ConfigError
from pl_predictor.sentiment import SentimentScorer
from pl_predictor.storage import Store


@pytest.mark.parametrize("argv", [["query", "--fixture", "nope"], ["run", "--fixture", "nope"]])
def test_unknown_fixture_exits_with_message(argv, monkeypatch, tmp_path, fixtures):
    monkeypatch.setattr("pl_predictor.pipeline.season_fixtures", lambda settings: fixtures)
    db = tmp_path / "pl.db"
    monkeypatch.setenv("PLS_DB_PATH", str(db))
    monkeypatch.setenv("X_BEARER_TOKEN", "test")
    monkeypatch.setattr(SentimentScorer, "__init__", lambda self, *a, **k: None)
    with pytest.raises(SystemExit, match="Unknown fixture id"):
        main(argv)
    if db.exists():  # `run` opens the store, but must not write fixtures for a typo
        assert Store(db).fixtures_frame().empty


def test_missing_config_file_exits_with_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PLS_TEAMS_PATH", str(tmp_path / "teams.yaml"))
    with pytest.raises(SystemExit, match="teams.yaml not found"):
        main(["fixtures"])


def test_watch_needs_token_up_front(monkeypatch):
    from pl_predictor import watch

    monkeypatch.setattr(watch, "file_logging", lambda: None)  # don't write the repo's logs/
    monkeypatch.setenv("X_BEARER_TOKEN", "")
    with pytest.raises(SystemExit, match="X_BEARER_TOKEN is not set"):
        main(["watch", "--once"])


def test_scorer_without_transformers_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ConfigError, match=r"\[sentiment\]"):
        SentimentScorer().score(["hello"])


def test_network_failure_is_one_line(monkeypatch):
    import requests

    from pl_predictor import pipeline

    class DownFPL:
        def teams(self):
            raise requests.ConnectionError("FPL down")

    monkeypatch.setattr(pipeline, "FPLHook", DownFPL)
    with pytest.raises(SystemExit, match="Could not reach a data source"):
        main(["fixtures"])


def test_run_watch_keeps_last_fixture_list_through_an_outage(monkeypatch, tmp_path, fixtures):
    import requests

    from pl_predictor import cli, pipeline

    class Stop(Exception):
        pass

    lists = iter([fixtures, requests.ConnectionError("FPL down")])

    def season(settings):
        item = next(lists)
        if isinstance(item, Exception):
            raise item
        return item

    seen, sleeps = [], []
    monkeypatch.setenv("PLS_DB_PATH", str(tmp_path / "pl.db"))
    monkeypatch.setenv("X_BEARER_TOKEN", "test")
    monkeypatch.setattr(SentimentScorer, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(pipeline, "season_fixtures", season)
    monkeypatch.setattr(pipeline, "run_once", lambda *a: seen.append(a[-1]) or (0, 0))
    monkeypatch.setattr(cli, "FIXTURES_REFRESH_S", 0)  # refresh every loop

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise Stop

    monkeypatch.setattr(cli.time, "sleep", sleep)
    with pytest.raises(Stop):
        main(["run", "--watch", "1"])
    assert seen == [fixtures, fixtures]  # second loop: refresh failed, last list reused


def test_run_fetches_fixtures_right_after_boot(monkeypatch, tmp_path, fixtures):
    from pl_predictor import cli, pipeline

    seen = []
    monkeypatch.setenv("PLS_DB_PATH", str(tmp_path / "pl.db"))
    monkeypatch.setenv("X_BEARER_TOKEN", "test")
    monkeypatch.setattr(SentimentScorer, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(pipeline, "season_fixtures", lambda settings: fixtures)
    monkeypatch.setattr(pipeline, "run_once", lambda *a: seen.append(a[-1]) or (0, 0))
    monkeypatch.setattr(cli.time, "monotonic", lambda: 5.0)  # e.g. a CI runner booted 5 s ago
    main(["run"])
    assert seen == [fixtures]
