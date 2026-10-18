from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import requests

from pl_predictor import model as m
from pl_predictor import watch
from pl_predictor.elo import elo_ratings, local_elo
from pl_predictor.features import FEATURES
from pl_predictor.matches import (
    SchemaError,
    UnmappedTeamError,
    fill_from_fdorg,
    normalise,
    season_of,
    to_codes,
)
from pl_predictor.predict import append_history, predict_upcoming, score, upcoming_fixtures
from pl_predictor.study import (
    CUTOFF,
    fixtures_needed,
    run_arm,
    sentiment_features,
)

TEAMS = [f"T{i:02d}" for i in range(20)]
NAMES = pd.DataFrame({"code": TEAMS, "fdcouk": TEAMS, "clubelo": TEAMS, "fdorg": TEAMS})


def synthetic_matches(seasons=range(2019, 2024), seed=0) -> pd.DataFrame:
    """A 20-team league where results follow hidden team strengths (380 matches a season)."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 0.5, len(TEAMS)), strict=True))
    rows = []
    for season in seasons:
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            day = pd.Timestamp(f"{season}-08-10") + pd.Timedelta(days=3 * (k // 10))
            hg = rng.poisson(1.5 * np.exp(strength[h] - strength[a]))
            ag = rng.poisson(1.1 * np.exp(strength[a] - strength[h]))
            rows.append(
                {
                    "season": season,
                    "date": day,
                    "kickoff": day.tz_localize("UTC") + pd.Timedelta(hours=14),
                    "home": h,
                    "away": a,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": "H" if hg > ag else "A" if ag > hg else "D",
                    "home_shots": hg * 3 + rng.poisson(9),
                    "away_shots": ag * 3 + rng.poisson(8),
                    "home_shots_on_target": hg + rng.poisson(3),
                    "away_shots_on_target": ag + rng.poisson(3),
                    "avg_h": 2.2,
                    "avg_d": 3.4,
                    "avg_a": 3.3,
                    "close_h": 2.2,
                    "close_d": 3.4,
                    "close_a": 3.3,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def league():
    matches = synthetic_matches()
    rows, feats, source = m.dataset(matches, elo_source="local", names=NAMES)
    return matches, rows, feats, source


# ---------------------------------------------------------------- data


def test_season_rule():
    assert season_of(pd.Timestamp("2020-07-26 15:00", tz="UTC")) == 2019
    assert season_of(pd.Timestamp("2026-08-15 11:30", tz="UTC")) == 2026
    assert season_of(pd.Timestamp("2027-01-01")) == 2026


def test_unmapped_team_raises():
    with pytest.raises(UnmappedTeamError, match="Nowhere FC"):
        to_codes(pd.Series(["T00", "Nowhere FC"]), "fdcouk", NAMES)


def _raw(new_format: bool) -> pd.DataFrame:
    raw = {
        "Date": [pd.Timestamp("2024-08-17")],
        "HomeTeam": ["T00"],
        "AwayTeam": ["T01"],
        "FTHG": [2],
        "FTAG": [1],
        "FTR": ["H"],
        "HS": [12],
        "AS": [8],
        "HST": [5],
        "AST": [2],
        "HC": [6],
        "AC": [3],
        "HF": [10],
        "AF": [12],
    }
    if new_format:
        raw |= {
            "Time": ["15:00"],
            "AvgH": [2.0],
            "AvgD": [3.5],
            "AvgA": [3.8],
            "AvgCH": [1.9],
            "AvgCD": [3.6],
            "AvgCA": [4.0],
        }
    else:
        raw |= {"BbAvH": [2.1], "BbAvD": [3.4], "BbAvA": [3.6]}
    return pd.DataFrame(raw)


def test_normalise_new_and_old_formats():
    new = normalise(_raw(True), 2024, NAMES)
    assert new.loc[0, ["avg_h", "close_h"]].tolist() == [2.0, 1.9]
    assert new.loc[0, "kickoff"] == pd.Timestamp("2024-08-17 14:00", tz="UTC")  # 15:00 BST
    old = normalise(_raw(False), 2010, NAMES)
    assert str(old["kickoff"].dt.tz) == "UTC"  # mixes with newer seasons
    assert (
        old.loc[0, "avg_h"] == 2.1
        and np.isnan(old.loc[0, "close_h"])
        and pd.isna(old.loc[0, "kickoff"])
    )


def test_schema_drift_is_reported():
    with pytest.raises(SchemaError, match="HST"):
        normalise(_raw(True).drop(columns=["HST"]), 2024, NAMES)


# ---------------------------------------------------------------- features and Elo


def test_features_are_as_of(league):
    matches, rows, feats, _ = league
    early = m.dataset(matches[matches["season"] <= 2020], elo_source="local", names=NAMES)[1]
    pd.testing.assert_frame_equal(early, feats.loc[early.index])


def test_new_club_gets_nan_form_but_a_prediction(league):
    matches, rows, feats, _ = league
    upcoming = pd.DataFrame(
        {
            "season": [2024],
            "date": [pd.Timestamp("2024-08-17")],
            "kickoff": [pd.Timestamp("2024-08-17 14:00", tz="UTC")],
            "home": ["NEW"],
            "away": ["T00"],
        }
    )
    rows2, feats2, _ = m.dataset(matches, upcoming, elo_source="local", names=NAMES)
    new = feats2.iloc[-1]
    assert np.isnan(new["diff_goals_for"]) and not np.isnan(new["elo_diff"])
    model = m.stage1().fit(feats.loc[rows["result"].notna(), FEATURES], rows["result"].dropna())
    assert m.proba(model, feats2.iloc[[-1]][FEATURES]).sum() == pytest.approx(1)


def test_local_elo_rewards_winners_and_carries_to_upcoming():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home": ["A", "A"],
            "away": ["B", "B"],
            "result": ["H", np.nan],
            "home_goals": [3, np.nan],
            "away_goals": [0, np.nan],
        }
    )
    elo = local_elo(rows)
    assert elo.loc[0].tolist() == [1500, 1500]
    assert elo.loc[1, "home_elo"] > 1500 > elo.loc[1, "away_elo"]


# ---------------------------------------------------------------- model


def test_walk_forward_beats_elo_on_signal_and_never_trains_on_test(league):
    _, rows, feats, _ = league
    report, y, probs, elo_probs = m.walk_forward(rows, feats)
    assert sorted(report["season"].unique()) == [2021, 2022, 2023]
    assert len(y) == len(probs) == 3 * 380 and np.allclose(probs.sum(axis=1), 1)
    summary = m.summarise(report)
    assert summary.loc["model", "log_loss"] < summary.loc["market (pre-match odds)", "log_loss"]


def test_metrics_and_demargin():
    p = m.demargin(pd.DataFrame({"h": [2.0], "d": [4.0], "a": [4.0]}))
    assert p.sum() == pytest.approx(1) and p[0, 0] == pytest.approx(0.5)
    res = m.metrics(pd.Series(["H"]), p)
    assert res["log_loss"] == pytest.approx(np.log(2)) and res["accuracy"] == 1


def test_missing_model_message(tmp_path):
    with pytest.raises(m.ModelMissing, match="model train"):
        m.load(tmp_path / "nope.joblib")


# ---------------------------------------------------------------- predictions and scoring


class FakeFPL:
    def __init__(self, fixtures):
        self._fx = fixtures

    def teams(self):
        return pd.DataFrame({"id": [1, 2, 3, 4], "short_name": ["T00", "T01", "T02", "T03"]})

    def fixtures(self):
        return self._fx


def test_upcoming_fixtures_picks_next_unplayed_gameweek():
    fx = pd.DataFrame(
        {
            "event": [7, 7, 8, None],
            "team_h": [1, 3, 1, 2],
            "team_a": [2, 4, 3, 4],
            "kickoff_time": pd.to_datetime(
                ["2026-10-17T11:30Z", "2026-10-19T19:00Z", "2026-10-24T14:00Z", None]
            ),
            "started": [True, False, False, False],
        }
    )
    rows, unscheduled = upcoming_fixtures(FakeFPL(fx).teams(), fx)
    assert rows[["home", "away", "gameweek"]].values.tolist() == [["T02", "T03", 7]]
    assert rows.loc[0, "season"] == 2026 and unscheduled == ["T01 v T03"]
    assert str(rows["kickoff"].dt.tz) == "UTC"


def test_score_uses_latest_pre_kickoff_prediction():
    ko = pd.Timestamp("2024-08-17 14:00", tz="UTC")
    hist = pd.DataFrame(
        {
            "season": 2024,
            "home": ["T00", "T00", "T00", "T02"],
            "away": ["T01", "T01", "T01", "T03"],
            "kickoff": [ko, ko, ko, ko + pd.Timedelta(days=30)],
            "predicted_at": [
                ko - pd.Timedelta(hours=5),
                ko - pd.Timedelta(hours=1),
                ko + pd.Timedelta(hours=1),
                ko,
            ],
            "p_home": [0.2, 0.6, 0.99, 0.5],
            "p_draw": [0.3, 0.2, 0.005, 0.25],
            "p_away": [0.5, 0.2, 0.005, 0.25],
            "elo_home": 0.5,
            "elo_draw": 0.25,
            "elo_away": 0.25,
        }
    )
    matches = pd.DataFrame(
        {
            "season": [2024],
            "home": ["T00"],
            "away": ["T01"],
            "result": ["H"],
            "avg_h": [2.0],
            "avg_d": [4.0],
            "avg_a": [4.0],
            "close_h": [np.nan],
            "close_d": [np.nan],
            "close_a": [np.nan],
        }
    )
    summary, done, pending = score(hist, matches)
    assert pending == 1 and len(done) == 1
    assert summary["model"]["log_loss"] == pytest.approx(-np.log(0.6))  # not the post-kickoff 0.99
    assert (
        summary["market (pre-match odds)"]["n"] == 1 and summary["closing odds (ceiling)"]["n"] == 0
    )


def test_locked_history_is_retried_then_reported(tmp_path, monkeypatch):
    calls = []

    def locked(*args, **kwargs):
        calls.append(1)
        raise PermissionError

    monkeypatch.setattr(pd.DataFrame, "to_csv", locked)
    monkeypatch.setattr("pl_predictor.predict.time.sleep", lambda s: None)
    with pytest.raises(PermissionError, match="close it"):
        append_history(pd.DataFrame({"a": [1]}), tmp_path / "h.csv", retries=2)
    assert len(calls) == 3


# ---------------------------------------------------------------- study


def _tweets(ko):
    t = lambda minutes: (ko + pd.Timedelta(minutes=minutes)).isoformat()  # noqa: E731
    return pd.DataFrame(
        {
            "fixture_id": "f1",
            "created_at": pd.to_datetime(
                [t(-60), t(-61), t(-24 * 60 - 1), t(-120), t(-90)], utc=True
            ),
            "side": ["home", "home", "home", "away", "neither"],
            "score": [0.9, 0.5, -1.0, -0.4, 1.0],
            "collected_at": pd.Timestamp(t(-58)),
        }
    )


def test_sentiment_window_sides_and_timing():
    ko = pd.Timestamp("2026-10-24 14:00", tz="UTC")
    db = pd.DataFrame(
        {"fixture_id": ["f1"], "kickoff": [ko], "home_code": ["T00"], "away_code": ["T01"]}
    )
    target = pd.DataFrame({"season": [2026], "home": ["T00"], "away": ["T01"], "kickoff": [ko]})
    out = sentiment_features(_tweets(ko), db, target).iloc[0]
    # At T excluded, T - 1 min included, before kickoff - 24h excluded, `neither` ignored.
    assert out["home_n"] == 1 and out["home_net"] == 0.5 and out["away_n"] == 1 and out["timing_ok"]
    moved = target.assign(kickoff=ko + pd.Timedelta(hours=2))
    assert sentiment_features(_tweets(ko), db, moved).iloc[0]["note"] == "kickoff mismatch"
    late = _tweets(ko).assign(collected_at=ko - CUTOFF - pd.Timedelta(hours=3))
    assert not sentiment_features(late, db, target).iloc[0]["timing_ok"]


def test_study_gate_passes_real_signal_and_rejects_noise():
    rng = np.random.default_rng(1)
    n = 600
    y = pd.Series(rng.choice(["H", "D", "A"], n, p=[0.45, 0.27, 0.28]))
    base = np.tile([0.45, 0.27, 0.28], (n, 1))
    signal = pd.DataFrame(
        {
            "home_net": np.where(y == "H", 0.6, -0.2) + rng.normal(0, 0.2, n),
            "away_net": np.where(y == "A", 0.6, -0.2) + rng.normal(0, 0.2, n),
        }
    )
    signal["net_diff"] = signal["home_net"] - signal["away_net"]
    noise = signal.apply(lambda c: rng.permutation(c.to_numpy()))
    folds = [np.arange(100 * k, 100 * (k + 1)) for k in range(1, 6)]
    assert run_arm("signal", y, base, signal, folds)["passed"]
    assert not run_arm("noise", y, base, noise, folds)["passed"]


def test_power_needs_more_fixtures_for_smaller_effects():
    need = fixtures_needed(0.3)
    assert need[0.002] > need[0.005] > need[0.01] > 0


# ---------------------------------------------------------------- watch


def _upcoming(now):
    return pd.DataFrame(
        {
            "season": 2026,
            "home": ["T00", "T02"],
            "away": ["T01", "T03"],
            "gameweek": 8,
            "kickoff": [now + timedelta(minutes=30), now - timedelta(minutes=10)],
        }
    )


def test_watch_collects_then_predicts_and_survives_failures(tmp_path, monkeypatch):
    now = datetime(2026, 10, 24, 13, 30, tzinfo=timezone.utc)
    upcoming = pd.concat(
        [
            _upcoming(now),
            pd.DataFrame(
                {
                    "season": [2026],
                    "home": ["T04"],
                    "away": ["T05"],
                    "gameweek": [8],
                    "kickoff": [now + timedelta(minutes=20)],
                }
            ),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(watch, "upcoming_fixtures", lambda teams, fixtures: (upcoming, []))
    events = []

    def collect(settings, row, now):
        if row["home"] == "T04":
            raise RuntimeError("X API down")
        events.append(("collect", row["home"]))
        return 3

    watch.run(
        None,
        once=True,
        clock=lambda: now,
        collect=collect,
        predict=lambda keys, now: events.append(("predict", next(iter(keys))[1])),
        fpl=FakeFPL(None),
        lock=tmp_path / "w.lock",
    )
    # T02 v T03 kicked off already (MISSED); T04 failed but did not stop T00.
    assert events == [("collect", "T00"), ("predict", "T00")]
    assert not (tmp_path / "w.lock").exists()


def test_watch_lock_blocks_second_instance(tmp_path):
    lock = tmp_path / "w.lock"
    watch.acquire_lock(lock)
    with pytest.raises(watch.AlreadyRunning):
        watch.acquire_lock(lock)


# ---------------------------------------------------------------- code-review regressions


class DownFPL:
    def teams(self):
        raise requests.ConnectionError("FPL down")

    def fixtures(self):
        raise requests.ConnectionError("FPL down")


def test_watch_survives_failed_first_refresh(tmp_path):
    lock = tmp_path / "w.lock"
    watch.run(None, once=True, fpl=DownFPL(), lock=lock, collect=None, predict=lambda k, n: None)
    watch.run(None, dry_run=True, fpl=DownFPL(), lock=lock, predict=lambda k, n: None)
    assert not lock.exists()


class Stop(Exception):
    pass


def test_watch_retries_failed_prediction_without_recollecting(tmp_path, monkeypatch):
    now = datetime(2026, 10, 24, 13, 30, tzinfo=timezone.utc)
    upcoming = _upcoming(now).iloc[[0]]
    monkeypatch.setattr(watch, "upcoming_fixtures", lambda teams, fixtures: (upcoming, []))
    sleeps, collects, predicts = [], [], []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise Stop

    def predict(keys, now):
        predicts.append(keys)
        if len(predicts) == 1:
            raise RuntimeError("football-data.co.uk timeout")

    monkeypatch.setattr(watch.time, "sleep", sleep)
    with pytest.raises(Stop):
        watch.run(
            None,
            clock=lambda: now,
            fpl=FakeFPL(None),
            lock=tmp_path / "w.lock",
            collect=lambda settings, row, now: collects.append(1) or 1,
            predict=predict,
        )
    assert len(collects) == 1 and len(predicts) == 2


def test_vanished_fixtures_are_reported_unless_handled():
    now = datetime(2026, 10, 24, 13, 30, tzinfo=timezone.utc)
    old = _upcoming(now)
    new = old.iloc[[1]]
    assert watch.vanished(old, new, set()) == [("T00", "T01")]
    assert watch.vanished(old, new, {(2026, "T00", "T01")}) == []
    assert watch.vanished(None, new, set()) == []


class FakeFdorg:
    def matches(self, season):
        return pd.DataFrame(
            {
                "status": ["FINISHED"],
                "utcDate": ["2026-10-17T14:00:00Z"],
                "homeTeam.tla": ["ZZZ"],
                "awayTeam.tla": ["T00"],
                "score.fullTime.home": [1],
                "score.fullTime.away": [0],
            }
        )


def test_fdorg_fill_skips_on_unmapped_team():
    table = pd.DataFrame(
        {"season": [2026], "home": ["T00"], "away": ["T01"], "date": [pd.Timestamp("2026-10-10")]}
    )
    out = fill_from_fdorg(table, FakeFdorg(), 2026, NAMES)
    pd.testing.assert_frame_equal(out, table)


def test_missing_clubelo_name_falls_back_to_local_elo():
    rows = synthetic_matches(seasons=[2023]).head(20)
    names = NAMES.assign(clubelo=None)
    elo, source = elo_ratings(rows, names, "auto")
    assert source == "local" and elo.notna().all().all()


def test_score_uses_whole_latest_row():
    ko = pd.Timestamp("2024-08-17 14:00", tz="UTC")
    hist = pd.DataFrame(
        {
            "season": 2024,
            "home": "T00",
            "away": "T01",
            "kickoff": ko,
            "predicted_at": [ko - pd.Timedelta(hours=5), ko - pd.Timedelta(hours=1)],
            "p_home": [0.2, 0.6],
            "p_draw": 0.2,
            "p_away": [0.6, 0.2],
            "elo_home": 0.5,
            "elo_draw": 0.25,
            "elo_away": 0.25,
            "home_tweets": [5.0, np.nan],
            "note": ["old run", np.nan],
        }
    )
    matches = pd.DataFrame(
        {
            "season": [2024],
            "home": ["T00"],
            "away": ["T01"],
            "result": ["H"],
            "avg_h": [2.0],
            "avg_d": [4.0],
            "avg_a": [4.0],
            "close_h": [np.nan],
            "close_d": [np.nan],
            "close_a": [np.nan],
        }
    )
    _, done, _ = score(hist, matches)
    row = done.iloc[0]
    assert row["p_home"] == 0.6 and np.isnan(row["home_tweets"]) and pd.isna(row["note"])


def test_predict_accepts_naive_now(league):
    matches, rows, feats, source = league
    bundle = m.train(rows, feats, source)
    upcoming = pd.DataFrame(
        {
            "season": [2024],
            "date": [pd.Timestamp("2024-08-17")],
            "kickoff": [pd.Timestamp("2024-08-17 14:00", tz="UTC")],
            "home": ["T00"],
            "away": ["T01"],
        }
    )
    rows2, feats2, _ = m.dataset(matches, upcoming, elo_source="local", names=NAMES)
    pred = predict_upcoming(bundle, rows2, feats2, rows2.index[-1:], datetime(2024, 8, 17, 10))
    assert str(pred.loc[0, "predicted_at"].tz) == "UTC"
    assert pred[["p_home", "p_draw", "p_away"]].sum(axis=1).iloc[0] == pytest.approx(1)
