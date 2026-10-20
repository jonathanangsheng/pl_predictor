"""Stage-1 model, baselines, walk-forward evaluation and the saved model bundle."""

from __future__ import annotations

import logging
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dixon_coles
from .elo import local_elo
from .features import FEATURE_LABELS, FEATURES, build_features
from .matches import ODDS

log = logging.getLogger(__name__)

LABELS = ["H", "D", "A"]
EVAL_SEASONS = 3
FIRST_TRAIN_SEASON = 2006  # 2005/06 has no prior matches to build form from
MATCHES_PER_SEASON = 380
MODEL_PATH = Path("models/model.joblib")


class ModelMissing(FileNotFoundError):
    pass


# ---------------------------------------------------------------- data assembly


def dataset(
    matches: pd.DataFrame, upcoming: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows (played + upcoming) and their features."""
    rows = pd.concat([matches, upcoming], ignore_index=True)  # concat skips None
    return rows, build_features(rows, local_elo(rows))


# ---------------------------------------------------------------- probabilities and metrics


def demargin(odds: pd.DataFrame) -> np.ndarray:
    """Decimal odds (h, d, a) -> probabilities with the overround removed; NaN rows stay NaN."""
    inv = 1 / odds.to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def per_match_log_loss(y, probs: np.ndarray) -> np.ndarray:
    """Log loss of each row's actual result (H/D/A) under `probs` (columns in LABELS order)."""
    actual = [LABELS.index(result) for result in y]
    p_actual = np.asarray(probs)[np.arange(len(actual)), actual]
    return -np.log(np.clip(p_actual, 1e-15, 1))


def trainable(rows: pd.DataFrame) -> pd.Series:
    """Played matches with enough prior history to train on."""
    return rows["result"].notna() & (rows["season"] >= FIRST_TRAIN_SEASON)


def metrics(y: pd.Series, probs: np.ndarray) -> dict:
    """Log loss and Brier (lower is better), then accuracy. Rows with NaN probabilities skipped."""
    probs = np.asarray(probs, dtype=float)
    ok = ~np.isnan(probs).any(axis=1)
    y, probs = np.asarray(y)[ok], probs[ok]
    if not len(y):
        return {"n": 0, "log_loss": np.nan, "brier": np.nan, "accuracy": np.nan}
    onehot = (np.asarray(LABELS)[None, :] == y[:, None]).astype(float)
    return {
        "n": int(len(y)),
        "log_loss": float(per_match_log_loss(y, probs).mean()),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "accuracy": float((probs.argmax(axis=1) == onehot.argmax(axis=1)).mean()),
    }


def calibration_table(y: pd.Series, probs: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, pooling the three outcomes into probability bins."""
    probs = np.asarray(probs, dtype=float)
    onehot = (np.asarray(LABELS)[None, :] == np.asarray(y)[:, None]).astype(float)
    predicted, observed = probs.ravel(), onehot.ravel()
    edges = np.linspace(0, 1, bins + 1)
    bin_of = np.clip(np.digitize(predicted, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": bin_of, "predicted": predicted, "observed": observed})
    table = df.groupby("bin").agg(
        predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size")
    )
    table.index = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in table.index]
    return table


# ---------------------------------------------------------------- models


def stage1():
    # Logistic regression is already well calibrated; a calibration wrapper made it worse.
    return make_pipeline(
        SimpleImputer(add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000),
    )


def blend(logistic_p: np.ndarray, dixon_coles_p: np.ndarray) -> np.ndarray:
    """Stage 1: logistic regression and Dixon-Coles averaged 50/50 (weight fixed before testing)."""
    return 0.5 * logistic_p + 0.5 * dixon_coles_p


def stage1_proba(
    logistic, rows: pd.DataFrame, feats: pd.DataFrame, idx: pd.Index, features=FEATURES
) -> np.ndarray:
    """Stage-1 probabilities for rows.loc[idx]: the logistic regression blended with Dixon-Coles.

    Dixon-Coles is refitted each week on matches played before it.
    """
    return blend(proba(logistic, feats.loc[idx, features]), dixon_coles.predict(rows, idx))


def explain(logistic, X: pd.DataFrame) -> pd.DataFrame:
    """Each feature's push on the logistic regression's home-vs-away log-odds (+ favours home).

    Push = (home weight - away weight) x the scaled feature, one row per match; the imputer's
    missing-form indicators are summed into "no recent form".
    """
    prep, lr = logistic[:-1], logistic[-1]
    classes = list(lr.classes_)
    weight = lr.coef_[classes.index("H")] - lr.coef_[classes.index("A")]
    names = prep.get_feature_names_out(X.columns)
    pushes = pd.DataFrame(prep.transform(X) * weight, columns=names, index=X.index)
    labels = [
        "no recent form" if name.startswith("missingindicator_") else FEATURE_LABELS.get(name, name)
        for name in names
    ]
    return pushes.T.groupby(labels, sort=False).sum().T


def elo_baseline():
    return make_pipeline(SimpleImputer(), LogisticRegression(max_iter=1000))


def proba(model, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities in H, D, A order."""
    raw = model.predict_proba(X)
    order = [list(model.classes_).index(label) for label in LABELS]
    return raw[:, order]


def walk_forward(
    rows: pd.DataFrame, feats: pd.DataFrame, eval_seasons: int = EVAL_SEASONS
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Train on seasons before S, test on S, for the last `eval_seasons` completed seasons.

    Returns per-season metrics, pooled held-out results, and model and Elo probabilities.
    """
    played = trainable(rows)
    counts = rows.loc[played, "season"].value_counts()
    complete_seasons = sorted(counts[counts >= MATCHES_PER_SEASON].index)
    results, ys, model_ps, elo_ps = [], [], [], []
    for season in complete_seasons[-eval_seasons:]:
        train = played & (rows["season"] < season)
        test = played & (rows["season"] == season)
        y_train, y_test = rows.loc[train, "result"], rows.loc[test, "result"]
        model = stage1().fit(feats.loc[train, FEATURES], y_train)
        elo = elo_baseline().fit(feats.loc[train, ["elo_diff"]], y_train)
        logistic_p = proba(model, feats.loc[test, FEATURES])
        dixon_coles_p = dixon_coles.predict(rows, rows.index[test])
        predictors = {
            "model": blend(logistic_p, dixon_coles_p),
            "logistic regression": logistic_p,
            "dixon-coles": dixon_coles_p,
            "elo": proba(elo, feats.loc[test, ["elo_diff"]]),
            "market (pre-match odds)": demargin(rows.loc[test, ODDS["avg"]]),
            "closing odds (ceiling)": demargin(rows.loc[test, ODDS["close"]]),
        }
        for name, probs in predictors.items():
            results.append({"season": season, "predictor": name, **metrics(y_test, probs)})
        ys.append(y_test)
        model_ps.append(predictors["model"])
        elo_ps.append(predictors["elo"])
        log.info("Walk-forward season %s/%s done", season, season + 1)
    return pd.DataFrame(results), pd.concat(ys), np.vstack(model_ps), np.vstack(elo_ps)


def summarise(report: pd.DataFrame) -> pd.DataFrame:
    """Mean over evaluated seasons, best log loss first."""
    return (
        report.groupby("predictor")[["log_loss", "brier", "accuracy", "n"]]
        .agg({"log_loss": "mean", "brier": "mean", "accuracy": "mean", "n": "sum"})
        .sort_values("log_loss")
    )


# ---------------------------------------------------------------- bundle


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def train(rows: pd.DataFrame, feats: pd.DataFrame) -> dict:
    played = trainable(rows)
    X, y = feats.loc[played, FEATURES], rows.loc[played, "result"]
    now = datetime.now(timezone.utc)
    return {
        "stage1": stage1().fit(X, y),
        "elo": elo_baseline().fit(X[["elo_diff"]], y),
        "features": list(FEATURES),
        "model_version": f"{now:%Y%m%d}-{_git_sha()}",
        "trained_on": int(played.sum()),
    }


def save(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path: Path = MODEL_PATH) -> dict:
    """Load a bundle this tool wrote. It is a pickle: never load files from elsewhere."""
    if not path.exists():
        raise ModelMissing(f"No model at {path}: run `pl-predictor model train` first")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = joblib.load(path)
    if any(issubclass(w.category, InconsistentVersionWarning) for w in caught):
        log.warning(
            "Model was saved with another scikit-learn version; retrain: pl-predictor model train"
        )
    return bundle
