"""Dixon-Coles goal model (Dixon & Coles 1997): Poisson goals per side plus a low-score correction.

Stage 1 averages it 50/50 with the logistic regression (model.blend). Walk-forward over
2020/21-2025/26 the blend beat the logistic regression in 5 of 6 seasons (mean log loss -0.0047,
95% CI -0.0090 to -0.0003); Dixon-Coles on its own was worse than either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .elo import NEWCOMER_POOL

XI = 0.0019  # time decay per day (the paper's 0.0065 per half-week)
WINDOW = pd.Timedelta(days=3 * 365)
RIDGE = 1e-3  # identifies the ratings and shrinks clubs with few matches
MAX_GOALS = 10


@dataclass(frozen=True)
class Ratings:
    attack: dict[str, float]
    defence: dict[str, float]
    home: float
    rho: float
    base: float
    newcomer: tuple[float, float]  # (attack, defence) for a club with no matches in the window


@dataclass(frozen=True)
class Design:
    """One fit's training data: home/away team indices (h, a), goals (hg, ag), recency weights
    (w) and the four low-score masks, which stay fixed while the optimiser runs."""

    teams: list[str]
    h: np.ndarray
    a: np.ndarray
    hg: np.ndarray
    ag: np.ndarray
    w: np.ndarray
    m00: np.ndarray
    m01: np.ndarray
    m10: np.ndarray
    m11: np.ndarray


def _design(played: pd.DataFrame, asof: pd.Timestamp) -> Design:
    """Matches in the WINDOW before `asof`, weighted by recency."""
    train = played[(played["date"] < asof) & (played["date"] >= asof - WINDOW)]
    if train.empty:
        raise ValueError(f"No matches before {asof:%Y-%m-%d} to fit Dixon-Coles on")
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {club: i for i, club in enumerate(teams)}
    hg = train["home_goals"].to_numpy(float)
    ag = train["away_goals"].to_numpy(float)
    return Design(
        teams,
        train["home"].map(index).to_numpy(),
        train["away"].map(index).to_numpy(),
        hg,
        ag,
        np.exp(-XI * (asof - train["date"]).dt.days.to_numpy()),
        (hg == 0) & (ag == 0),
        (hg == 0) & (ag == 1),
        (hg == 1) & (ag == 0),
        (hg == 1) & (ag == 1),
    )


def _nll(p: np.ndarray, d: Design) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood (constant terms dropped) and its gradient."""
    n = len(d.teams)
    attack, defence, (home, rho, base) = p[:n], p[n : 2 * n], p[2 * n :]
    log_lam = base + home + attack[d.h] - defence[d.a]  # home side's expected goals (log)
    log_mu = base + attack[d.a] - defence[d.h]
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    lam_mu = lam * mu

    # Low-score correction tau and its derivatives by log(lam), log(mu) and rho, case by case.
    t = np.ones_like(lam)
    d_lam, d_mu, d_rho = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
    m = d.m00  # 0-0: tau = 1 - lam*mu*rho
    t[m], d_lam[m], d_mu[m] = 1 - lam_mu[m] * rho, -lam_mu[m] * rho, -lam_mu[m] * rho
    d_rho[m] = -lam_mu[m]
    m = d.m01  # 0-1: tau = 1 + lam*rho
    t[m], d_lam[m], d_rho[m] = 1 + lam[m] * rho, lam[m] * rho, lam[m]
    m = d.m10  # 1-0: tau = 1 + mu*rho
    t[m], d_mu[m], d_rho[m] = 1 + mu[m] * rho, mu[m] * rho, mu[m]
    m = d.m11  # 1-1: tau = 1 - rho
    t[m], d_rho[m] = 1 - rho, -1.0
    t = np.clip(t, 1e-10, None)

    ll = np.log(t) + d.hg * log_lam - lam + d.ag * log_mu - mu
    g_lam = d.w * (d.hg - lam + d_lam / t)  # d(weighted ll) / d log(lam)
    g_mu = d.w * (d.ag - mu + d_mu / t)
    grad = np.empty_like(p)
    grad[:n] = -(np.bincount(d.h, g_lam, n) + np.bincount(d.a, g_mu, n)) + 2 * RIDGE * attack
    grad[n : 2 * n] = np.bincount(d.a, g_lam, n) + np.bincount(d.h, g_mu, n) + 2 * RIDGE * defence
    grad[2 * n] = -g_lam.sum()  # home advantage
    grad[2 * n + 1] = -(d.w * d_rho / t).sum()
    grad[2 * n + 2] = -(g_lam.sum() + g_mu.sum())  # base rate
    penalty = RIDGE * (attack @ attack + defence @ defence)
    return -(d.w * ll).sum() + penalty, grad


def fit(played: pd.DataFrame, asof: pd.Timestamp) -> Ratings:
    """Ratings from matches in the WINDOW before `asof`, recent matches weighted more."""
    d = _design(played, asof)
    n = len(d.teams)
    x0 = np.r_[np.zeros(2 * n), 0.25, -0.05, 0.1]
    bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3), (None, None)]
    p = minimize(_nll, x0, args=(d,), jac=True, method="L-BFGS-B", bounds=bounds).x
    attack = dict(zip(d.teams, p[:n], strict=True))
    defence = dict(zip(d.teams, p[n : 2 * n], strict=True))
    weakest = sorted(d.teams, key=lambda c: attack[c] + defence[c])[:NEWCOMER_POOL]
    newcomer = (np.mean([attack[c] for c in weakest]), np.mean([defence[c] for c in weakest]))
    return Ratings(attack, defence, p[2 * n], p[2 * n + 1], p[2 * n + 2], newcomer)


def expected_goals(r: Ratings, home: str, away: str) -> np.ndarray:
    """Expected goals (home, away)."""
    att_h, def_h = (r.attack[home], r.defence[home]) if home in r.attack else r.newcomer
    att_a, def_a = (r.attack[away], r.defence[away]) if away in r.attack else r.newcomer
    return np.exp([r.base + r.home + att_h - def_a, r.base + att_a - def_h])


def probs(r: Ratings, home: str, away: str) -> np.ndarray:
    """P(home win), P(draw), P(away win)."""
    lam, mu = expected_goals(r, home, away)
    goals = np.arange(MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    grid[:2, :2] *= [[1 - lam * mu * r.rho, 1 + lam * r.rho], [1 + mu * r.rho, 1 - r.rho]]
    grid /= grid.sum()
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def predict(rows: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    """H/D/A probabilities for rows.loc[idx], refitted each week on matches played before it."""
    return _weekly(rows, idx, probs, 3)


def predict_goals(rows: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    """Expected goals (home, away) for rows.loc[idx], fitted as in predict()."""
    return _weekly(rows, idx, expected_goals, 2)


def _weekly(rows: pd.DataFrame, idx: pd.Index, per_match, width: int) -> np.ndarray:
    played = rows[rows["result"].notna()]
    target = rows.loc[idx]
    out = np.empty((len(idx), width))
    weeks = target["date"].dt.to_period("W-MON").to_numpy()
    for week in pd.unique(weeks):
        at = np.flatnonzero(weeks == week)
        ratings = fit(played, target["date"].iloc[at].min())
        pairs = zip(target["home"].iloc[at], target["away"].iloc[at], strict=True)
        out[at] = [per_match(ratings, home, away) for home, away in pairs]
    return out
