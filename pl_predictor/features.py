"""As-of features: every value for a match uses only that team's earlier matches.

Walk-forward testing (2023/24-2025/26) kept only what beat the Elo baseline: Elo difference plus
10-match form differences in goals, shots and shots on target. Corners, fouls, table position,
rest days and gradient boosting all made log loss worse.
"""

from __future__ import annotations

import pandas as pd

ROLLING = ["goals_for", "goals_against", "shots_for", "shots_against", "sot_for", "sot_against"]
WINDOW = 10
MIN_MATCHES = 5  # fewer recent matches than this -> form is NaN (the model imputes it)
LOOKBACK = pd.Timedelta(days=400)

FEATURES = ["elo_diff"] + [f"diff_{stat}" for stat in ROLLING]


def _side(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    """Each match from one side's point of view: goals/shots "for" and the opponent's "against"."""
    other = "away" if side == "home" else "home"
    return pd.DataFrame(
        {
            "row": rows.index,
            "side": side,
            "team": rows[side],
            "date": rows["date"],
            "goals_for": rows[f"{side}_goals"],
            "goals_against": rows[f"{other}_goals"],
            "shots_for": rows[f"{side}_shots"],
            "shots_against": rows[f"{other}_shots"],
            "sot_for": rows[f"{side}_shots_on_target"],
            "sot_against": rows[f"{other}_shots_on_target"],
        }
    )


def team_form(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, side): mean of each stat over the team's previous 10 matches."""
    long = pd.concat([_side(rows, "home"), _side(rows, "away")], ignore_index=True)
    long = long.sort_values(["team", "date", "row"], kind="stable").reset_index(drop=True)
    by_team = long.groupby("team", sort=False)
    # shift(1): only earlier matches count. Matches without stats (football-data.org fill) are
    # skipped by the mean.
    prior = by_team[ROLLING].shift(1)
    rolled = prior.groupby(long["team"]).rolling(WINDOW, min_periods=1).mean()
    long[ROLLING] = rolled.reset_index(level=0, drop=True)
    # Too little recent history (promoted clubs, long absences): NaN rather than dropping the row.
    fifth_prior = by_team["date"].shift(MIN_MATCHES)
    sparse = fifth_prior.isna() | (long["date"] - fifth_prior > LOOKBACK)
    long.loc[sparse, ROLLING] = float("nan")
    return long


def build_features(rows: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix aligned to `rows` (played and upcoming), plus each side's Elo for display."""
    long = team_form(rows)
    home, away = (
        long[long["side"] == side].set_index("row")[ROLLING].reindex(rows.index)
        for side in ("home", "away")
    )
    feats = (home - away).add_prefix("diff_")
    feats["home_elo"], feats["away_elo"] = elo["home_elo"], elo["away_elo"]
    feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]
    return feats[FEATURES + ["home_elo", "away_elo"]].astype(float)
