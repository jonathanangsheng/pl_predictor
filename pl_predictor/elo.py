"""Team strength: Elo ratings computed from match results."""

from __future__ import annotations

import pandas as pd

HOME_ADVANTAGE = 60
K = 20
START_RATING = 1500.0
LEAGUE_SIZE = 20  # once a full league is rated, newcomers are promoted clubs
NEWCOMER_POOL = 3  # a new club starts at the mean of this many weakest clubs (also Dixon-Coles)


def local_elo(rows: pd.DataFrame) -> pd.DataFrame:
    """Pre-match ratings for every row (played or upcoming), oldest first.

    World-football-Elo style: K=20 scaled by goal margin, 60-point home advantage. A club seen for
    the first time starts at the mean of the three lowest current ratings (it is almost always
    promoted).
    """
    ratings: dict[str, float] = {}
    home_elo, away_elo = [], []
    ordered = rows.sort_values("date", kind="stable")
    cols = ["home", "away", "result", "home_goals", "away_goals"]
    for home, away, result, home_goals, away_goals in ordered[cols].itertuples(index=False):
        for club in (home, away):
            if club not in ratings:
                lowest = sorted(ratings.values())[:NEWCOMER_POOL]
                full = len(ratings) >= LEAGUE_SIZE
                ratings[club] = sum(lowest) / len(lowest) if full else START_RATING
        h, a = ratings[home], ratings[away]
        home_elo.append(h)
        away_elo.append(a)
        if pd.isna(result):
            continue
        expected = 1 / (1 + 10 ** ((a - h - HOME_ADVANTAGE) / 400))
        actual = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        margin = abs(home_goals - away_goals)
        scale = 1 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
        delta = K * scale * (actual - expected)
        ratings[home] = h + delta
        ratings[away] = a - delta
    elo = pd.DataFrame({"home_elo": home_elo, "away_elo": away_elo}, index=ordered.index)
    return elo.loc[rows.index]
