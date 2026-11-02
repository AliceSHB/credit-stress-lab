"""Satellite model: the macro cycle translated into a default-rate multiplier.

Card delinquency is close to a random walk, so the model is estimated on the
quarterly *change* rather than the level. That keeps the coefficients
interpretable (a change in unemployment moves the default rate, not the other
way round) and avoids the near-unit-root behaviour a lagged level produces on
this sample.

A scenario is a 12-quarter path for unemployment, GDP growth and house price
growth. The fitted equation is integrated forward, the scenario path is
compared with the baseline quarter by quarter, and the largest ratio becomes
the PD multiplier used in the exposure-level calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEPENDENT = "card_delinquency_rate"
REGRESSORS = ["unemployment_rate", "gdp_growth", "hpi_growth"]
HORIZON = 12


def fit_satellite(frame: pd.DataFrame, start: str = "1991-01-01", end: str = "2019-12-31") -> dict:
    data = frame.loc[start:end].copy()
    data["d_rate"] = data[DEPENDENT]
    data["delta_rate"] = data["d_rate"].diff()
    data["delta_unemployment"] = data["unemployment_rate"].diff()
    data = data.dropna(subset=["delta_rate", "delta_unemployment", "gdp_growth", "hpi_growth"])

    names = ["delta_unemployment", "gdp_growth", "hpi_growth"]
    X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy(float) for c in names])
    y = data["delta_rate"].to_numpy(float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())

    labels = ["const"] + names
    return {
        "coefficients": {n: float(b) for n, b in zip(labels, beta)},
        "std_errors": {n: float(s) for n, s in zip(labels, se)},
        "t_stats": {n: float(b / s) for n, b, s in zip(labels, beta, se)},
        "r2": r2,
        "n_obs": int(len(y)),
        "sample": [str(data.index[0].date()), str(data.index[-1].date())],
        "beta": beta,
    }


def _ramp(shock: float) -> np.ndarray:
    """Shock builds over three quarters, holds, then fades by quarter twelve."""
    shape = np.array([0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15])
    return shock * shape


SCENARIOS = {
    "baseline": {"unemployment_rate": 0.0, "gdp_growth": 0.0, "hpi_growth": 0.0},
    "adverse": {"unemployment_rate": 3.0, "gdp_growth": -2.5, "hpi_growth": -8.0},
    "severe": {"unemployment_rate": 5.5, "gdp_growth": -4.5, "hpi_growth": -15.0},
}


def _integrate(fit: dict, start: float, base_macro: pd.Series, shock: dict) -> np.ndarray:
    beta = fit["beta"]
    paths = {k: _ramp(v) for k, v in shock.items()}
    level = start
    out = []
    prev_u = 0.0
    for q in range(HORIZON):
        u_shock = paths["unemployment_rate"][q]
        du = u_shock - prev_u
        prev_u = u_shock
        x = np.array([
            1.0,
            du,
            base_macro["gdp_growth"] + paths["gdp_growth"][q],
            base_macro["hpi_growth"] + paths["hpi_growth"][q],
        ])
        level = level + float(x @ beta)
        out.append(level)
    return np.array(out)


def scenario_multipliers(fit: dict, frame: pd.DataFrame, cap: float = 3.0) -> pd.DataFrame:
    base_macro = frame[REGRESSORS].dropna().iloc[-1]
    start = float(frame[DEPENDENT].dropna().iloc[-1])
    base_path = _integrate(fit, start, base_macro, SCENARIOS["baseline"])

    rows = []
    for name, shock in SCENARIOS.items():
        path = _integrate(fit, start, base_macro, shock)
        ratio = path / np.clip(base_path, 1e-9, None)
        mult = 1.0 if name == "baseline" else float(np.clip(ratio.max(), 1.0, cap))
        rows.append({
            "scenario": name,
            "unemployment_peak": float(base_macro["unemployment_rate"] + max(_ramp(shock["unemployment_rate"]))),
            "gdp_growth_trough": float(base_macro["gdp_growth"] + min(_ramp(shock["gdp_growth"]))),
            "hpi_growth_trough": float(base_macro["hpi_growth"] + min(_ramp(shock["hpi_growth"]))),
            "projected_delinquency_peak": float(path.max()) / 100.0,
            "pd_multiplier": mult,
            "path": [round(v / 100.0, 5) for v in path.tolist()],
        })
    return pd.DataFrame(rows)
