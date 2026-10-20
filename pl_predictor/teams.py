"""Club metadata and the text matching used to attribute a tweet to a side."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

from .config import ConfigError


@dataclass(frozen=True)
class Team:
    name: str
    code: str
    hashtags: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    keywords: tuple[str, ...] = field(default=())
    colour: str = "#898781"
    lat: float | None = None
    lon: float | None = None

    @cached_property
    def _pattern(self) -> re.Pattern[str]:
        terms = [rf"#{re.escape(h)}\b" for h in self.hashtags]
        terms += [rf"@{re.escape(h)}\b" for h in self.handles]
        terms += [rf"\b{re.escape(k)}\b" for k in self.keywords]
        return re.compile("|".join(terms), re.IGNORECASE)

    def is_mentioned(self, text: str) -> bool:
        return bool(self._pattern.search(text))

    def query_terms(self) -> list[str]:
        """X search operators for this club, most specific first."""
        terms = [f"#{h}" for h in self.hashtags]
        terms += [f"@{h}" for h in self.handles]
        terms += [f'"{k}"' if " " in k else k for k in self.keywords]
        return terms


def load_teams(path: str | Path) -> dict[str, Team]:
    """Load teams.yaml into a dict keyed by club code."""
    if not Path(path).exists():
        raise ConfigError(f"{path} not found: run from the project folder or set PLS_TEAMS_PATH")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    teams = {}
    for entry in raw["teams"]:
        team = Team(
            name=entry["name"],
            code=entry["code"].upper(),
            hashtags=tuple(entry.get("hashtags", [])),
            handles=tuple(entry.get("handles", [])),
            keywords=tuple(entry.get("keywords", [])),
            **{k: entry[k] for k in ("colour", "lat", "lon") if k in entry},
        )
        teams[team.code] = team
    return teams
