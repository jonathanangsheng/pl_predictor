import json
from datetime import date

import pandas as pd
import pytest
import requests

from pl_predictor.hooks import ClubEloHook, FootballDataCoUkHook, FootballDataOrgHook, FPLHook


class FakeSession:
    """Stands in for requests.Session: returns canned bodies keyed by URL path suffix."""

    def __init__(self, routes: dict[str, bytes | str | dict | list]):
        self.routes = routes
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        body = next((b for suffix, b in self.routes.items() if url.endswith(suffix)), None)
        resp = requests.Response()
        resp.url = url
        if body is None:
            resp.status_code, resp._content = 404, b""
        else:
            resp.status_code = 200
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
            resp._content = body.encode() if isinstance(body, str) else body
        return resp


def test_football_data_co_uk_results():
    csv = (
        "\ufeffDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,HxG,AxG\n"
        "E0,12/10/2026,Coventry,Newcastle,4,1,1.43,1.84\n"
        ",,,,,,,\n"
    )
    session = FakeSession({"2627/E0.csv": csv.encode("utf-8")})
    df = FootballDataCoUkHook(session=session).results(2026)

    assert session.calls[0][0].endswith("/mmz4281/2627/E0.csv")
    assert session.headers["User-Agent"].startswith("pl-predictor/")
    assert list(df.columns)[0] == "Div"  # BOM stripped
    assert len(df) == 1  # blank trailing row dropped
    assert df.loc[0, "Date"] == pd.Timestamp("2026-10-12")  # day-first dates
    assert df.loc[0, "HxG"] == 1.43


def test_football_data_co_uk_old_cp1252_file():
    csv = "Div,Date,HomeTeam,AwayTeam,Referee\nE0,14/08/99,Arsenal,Leicester,G Poll \xa3\n"
    session = FakeSession({"9900/E0.csv": csv.encode("cp1252")})
    df = FootballDataCoUkHook(session=session).results(1999)
    assert df.loc[0, "Referee"].endswith("£")


def test_fpl_hook():
    bootstrap = {
        "teams": [{"id": 1, "short_name": "ARS"}],
        "elements": [{"id": 7, "expected_goals": "1.2"}],
        "events": [{"id": 7, "is_current": True}],
    }
    fixtures = [
        {"id": 61, "event": 7, "kickoff_time": "2026-10-17T11:30:00Z", "team_h": 9, "team_a": 6}
    ]
    live = {
        "elements": [
            {
                "id": 7,
                "stats": {"minutes": 90, "expected_goals": "0.4"},
                "explain": [{"fixture": 61, "stats": []}],
            }
        ]
    }
    session = FakeSession(
        {
            "bootstrap-static/": bootstrap,
            "fixtures/": fixtures,
            "element-summary/7/": {"history": [{"round": 1}]},
            "event/7/live/": live,
        }
    )
    hook = FPLHook(session=session)

    assert hook.teams().loc[0, "short_name"] == "ARS"
    assert hook.players().loc[0, "id"] == 7
    assert hook.gameweeks().loc[0, "is_current"]
    df = hook.fixtures(gameweek=7, future=True)
    assert session.calls[-1][1] == {"event": 7, "future": 1}
    assert str(df["kickoff_time"].dt.tz) == "UTC"
    hook.fixtures()
    assert session.calls[-1][1] == {}  # unset filters aren't sent
    assert hook.player_history(7).loc[0, "round"] == 1
    row = hook.live(7).iloc[0]
    assert row["minutes"] == 90 and row["fixtures"] == [61]


CLUBELO_CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Arsenal,ENG,1,2051.3,2026-10-13,2026-10-17\n"
    "2,Man City,ENG,1,2010.9,2026-10-13,2026-10-17\n"
)


def test_clubelo_ratings_and_history():
    session = FakeSession({"/2026-10-18": CLUBELO_CSV, "/ManCity": CLUBELO_CSV})
    hook = ClubEloHook(session=session)

    df = hook.ratings(date(2026, 10, 18))
    assert df.loc[0, "Club"] == "Arsenal" and df.loc[0, "Elo"] == 2051.3
    assert df["From"].dtype.kind == "M"
    assert hook.ratings("2026-10-18").shape == df.shape
    hook.club_history("Man City")
    assert session.calls[-1][0] == "http://api.clubelo.com/ManCity"


def test_football_data_org_hook():
    standings = {
        "standings": [
            {"type": "HOME", "table": [{"position": 9, "team": {"tla": "XXX"}}]},
            {"type": "TOTAL", "table": [{"position": 1, "team": {"tla": "ARS"}, "points": 19}]},
        ]
    }
    matches = {
        "matches": [{"id": 1, "utcDate": "2026-10-17T11:30:00Z", "homeTeam": {"tla": "EVE"}}]
    }
    session = FakeSession(
        {"PL/standings": standings, "PL/matches": matches, "PL/teams": {"teams": [{"tla": "ARS"}]}}
    )
    hook = FootballDataOrgHook(token="t0ken", session=session)

    assert session.headers["X-Auth-Token"] == "t0ken"
    table = hook.standings()
    assert table.loc[0, "team.tla"] == "ARS" and len(table) == 1  # TOTAL, not HOME
    df = hook.matches(season=2026)
    assert session.calls[-1][1] == {"season": 2026}
    assert df.loc[0, "homeTeam.tla"] == "EVE" and str(df["utcDate"].dt.tz) == "UTC"
    assert hook.teams().loc[0, "tla"] == "ARS"


def test_football_data_org_requires_token(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_ORG_TOKEN", "")
    with pytest.raises(ValueError, match="FOOTBALL_DATA_ORG_TOKEN"):
        FootballDataOrgHook(session=FakeSession({}))


def test_http_errors_raise():
    with pytest.raises(requests.HTTPError):
        FPLHook(session=FakeSession({})).teams()
