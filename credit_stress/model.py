"""Loan-level probability of default: two models, calibrated, and scored."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET

RANDOM_STATE = 42


def split(df: pd.DataFrame, test_size: float = 0.30):
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def _logistic() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])


def _boosted() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between defaulters and non-defaulters."""
    order = np.argsort(score)
    y = np.asarray(y_true)[order]
    s = np.asarray(score)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration_table(y_true, prob, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(prob), bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bucket": q, "y": np.asarray(y_true), "p": np.asarray(prob)})
    out = frame.groupby("bucket").agg(
        accounts=("y", "size"), observed_pd=("y", "mean"), predicted_pd=("p", "mean")
    )
    out["gap"] = out["observed_pd"] - out["predicted_pd"]
    return out.reset_index()


def fit_and_score(df: pd.DataFrame) -> dict:
    X_tr, X_te, y_tr, y_te = split(df)
    results: dict = {"models": {}}
    fitted: dict = {}

    candidates = {
        "logistic": _logistic(),
        "boosted": _boosted(),
    }
    for name, est in candidates.items():
        cal = CalibratedClassifierCV(estimator=est, method="isotonic", cv=3)
        cal.fit(X_tr, y_tr)
        p = cal.predict_proba(X_te)[:, 1]
        results["models"][name] = {
            "auc": float(roc_auc_score(y_te, p)),
            "ks": ks_statistic(y_te.values, p),
            "brier": float(brier_score_loss(y_te, p)),
            "mean_pd": float(np.mean(p)),
        }
        fitted[name] = (cal, p)

    best = max(results["models"], key=lambda k: results["models"][k]["auc"])
    cal, p = fitted[best]
    results["best_model"] = best
    results["n_train"], results["n_test"] = int(len(X_tr)), int(len(X_te))
    results["test_default_rate"] = float(y_te.mean())
    results["calibration"] = calibration_table(y_te, p).to_dict(orient="records")

    perm = permutation_importance(cal, X_te, y_te, n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    imp = (pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
           .sort_values("importance", ascending=False).head(10))
    results["importance"] = imp.to_dict(orient="records")

    results["_model"] = cal
    results["_p_test"] = p
    return results


def calibrate_level(p: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    """Shift log-odds so the portfolio mean PD hits a stated central tendency.

    The source file carries a one-month default label, so its own default rate
    is not a portfolio central tendency. Ranking comes from the model; the
    level is an explicit assumption, which is how a scorecard is actually
    calibrated in a bank.
    """
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(z + mid)))))
        if mean > target_mean:
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2.0
    return 1.0 / (1.0 + np.exp(-(z + shift))), shift


def score_population(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES].astype(float))[:, 1]
