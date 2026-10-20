"""Hooks: thin clients for external football data sources, each returning pandas DataFrames.

- FootballDataCoUkHook  football-data.co.uk  results, stats, xG, odds; upcoming odds (CSV)
- FPLHook               Fantasy Premier League API  fixtures, table, player stats and xG
- FootballDataOrgHook   api.football-data.org v4  fixtures, results, standings (needs a token)
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from . import __version__
from .config import get_settings


class _Hook:
    base_url: str

    def __init__(self, session: requests.Session | None = None, timeout: float = 30):
        self.session = session or requests.Session()
        # football-data.co.uk redirects requests that have no User-Agent.
        self.session.headers["User-Agent"] = f"pl-predictor/{__version__}"
        self.timeout = timeout

    def _get(self, path: str, **params) -> requests.Response:
        params = {k: v for k, v in params.items() if v is not None}
        resp = self.session.get(f"{self.base_url}/{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp


class FootballDataCoUkHook(_Hook):
    base_url = "https://www.football-data.co.uk"

    def results(self, season: int, division: str = "E0") -> pd.DataFrame:
        """One row per played match. `season` is the starting year (2026 = 2026/27); E0 = PL.

        Column guide: https://www.football-data.co.uk/notes.txt
        """
        code = f"{season % 100:02d}{(season + 1) % 100:02d}"
        return self._csv(f"mmz4281/{code}/{division}.csv")

    def upcoming(self, division: str = "E0") -> pd.DataFrame:
        """Next round of fixtures with bookmaker odds (AvgH/AvgD/AvgA, ...); updated midweek."""
        df = self._csv("fixtures.csv")
        return df[df["Div"] == division].reset_index(drop=True)

    def _csv(self, path: str) -> pd.DataFrame:
        raw = self._get(path).content
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:  # older seasons are Windows-1252
            text = raw.decode("cp1252")
        df = pd.read_csv(io.StringIO(text)).dropna(how="all")
        # Older seasons use two-digit years, so parse each date on its own.
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        return df


def fpl_codes(teams: pd.DataFrame) -> dict[int, str]:
    """FPLHook.teams() id -> club code."""
    return dict(zip(teams["id"], teams["short_name"], strict=True))


class FPLHook(_Hook):
    """Unofficial, undocumented API behind fantasy.premierleague.com. Current season only."""

    base_url = "https://fantasy.premierleague.com/api"

    def _bootstrap(self, key: str) -> pd.DataFrame:
        return pd.DataFrame(self._get("bootstrap-static/").json()[key])

    def teams(self) -> pd.DataFrame:
        """Clubs with table `position` and `strength_overall_home`/`_away` (1-5).

        FPL leaves played/win/draw/loss/points and the attack/defence strengths at 0;
        build the table from `fixtures()` or use FootballDataOrgHook.standings().
        """
        return self._bootstrap("teams")

    def players(self) -> pd.DataFrame:
        """Season totals per player: minutes, goals, xG, xA, price, injury news, ..."""
        return self._bootstrap("elements")

    def gameweeks(self) -> pd.DataFrame:
        return self._bootstrap("events")

    def fixtures(self, gameweek: int | None = None, future: bool = False) -> pd.DataFrame:
        """Fixtures with kickoff, score and difficulty. Team ids match `teams()["id"]`."""
        rows = self._get("fixtures/", event=gameweek, future=1 if future else None).json()
        df = pd.DataFrame(rows)
        if not df.empty:
            df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
        return df

    def player_history(self, player_id: int) -> pd.DataFrame:
        """One row per match this season for a player, including xG and xA."""
        return pd.DataFrame(self._get(f"element-summary/{player_id}/").json()["history"])

    def live(self, gameweek: int) -> pd.DataFrame:
        """Every player's stats for a gameweek, with the fixture ids they played in."""
        rows = [
            {"id": e["id"], **e["stats"], "fixtures": [x["fixture"] for x in e["explain"]]}
            for e in self._get(f"event/{gameweek}/live/").json()["elements"]
        ]
        return pd.DataFrame(rows)


class FootballDataOrgHook(_Hook):
    """football-data.org v4. The free tier covers the PL at 10 requests/minute."""

    base_url = "https://api.football-data.org/v4"

    def __init__(self, token: str | None = None, **kwargs):
        super().__init__(**kwargs)
        token = token or get_settings().football_data_org_token
        if not token:
            raise ValueError("FOOTBALL_DATA_ORG_TOKEN is not set (see .env.example)")
        self.session.headers["X-Auth-Token"] = token

    def matches(self, season: int | None = None, competition: str = "PL") -> pd.DataFrame:
        """All matches in a season (starting year; default current), flattened."""
        data = self._get(f"competitions/{competition}/matches", season=season).json()
        df = pd.json_normalize(data["matches"])
        if not df.empty:
            df["utcDate"] = pd.to_datetime(df["utcDate"], utc=True)
        return df

    def standings(self, season: int | None = None, competition: str = "PL") -> pd.DataFrame:
        """The overall league table (not the home/away splits)."""
        data = self._get(f"competitions/{competition}/standings", season=season).json()
        table = next(s["table"] for s in data["standings"] if s["type"] == "TOTAL")
        return pd.json_normalize(table)

    def teams(self, season: int | None = None, competition: str = "PL") -> pd.DataFrame:
        data = self._get(f"competitions/{competition}/teams", season=season).json()
        return pd.json_normalize(data["teams"])
