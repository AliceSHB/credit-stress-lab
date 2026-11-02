"""From a macro scenario to expected loss, staging and overlay triggers."""
from __future__ import annotations

import numpy as np
import pandas as pd

LGD = 0.65          # unsecured retail assumption, stated not estimated
HORIZON = 12        # months used to annualise a one-month PD
STAGE2_PD_FLOOR = 0.035
TRIGGER_DECL = {"ecl_increase": 0.25, "coverage": 0.08, "stage2_share": 0.30}


def run_stress(df: pd.DataFrame, pd_12m: np.ndarray, multipliers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[["EAD"]].copy()
    work["pd_12m"] = np.clip(pd_12m, 1e-6, 0.99)
    work["lgd"] = LGD

    rows = []
    for _, row in multipliers.iterrows():
        m = float(row["pd_multiplier"])
        pd_s = np.clip(work["pd_12m"] * m, 0.0, 0.99)
        ecl = pd_s * LGD * work["EAD"]
        stage2 = (pd_s >= STAGE2_PD_FLOOR) | (work["pd_12m"] * m >= work["pd_12m"].mean() * 3)
        rows.append({
            "scenario": row["scenario"],
            "pd_multiplier": m,
            "mean_pd_12m": float(pd_s.mean()),
            "ead": float(work["EAD"].sum()),
            "ecl": float(ecl.sum()),
            "coverage": float(ecl.sum() / max(work["EAD"].sum(), 1.0)),
            "stage2_share": float(work.loc[stage2, "EAD"].sum() / max(work["EAD"].sum(), 1.0)),
        })
    results = pd.DataFrame(rows)
    base = results.loc[results["scenario"] == "baseline"].iloc[0]
    results["ecl_delta"] = results["ecl"] - base["ecl"]
    results["ecl_delta_pct"] = results["ecl_delta"] / max(base["ecl"], 1e-9)
    results["pd_delta"] = results["mean_pd_12m"] - base["mean_pd_12m"]

    adverse = results.loc[results["scenario"] == "adverse"].iloc[0]
    triggers = pd.DataFrame([
        {"trigger": "Expected loss increase under the adverse scenario",
         "threshold": f">= {TRIGGER_DECL['ecl_increase']:.0%} of baseline ECL",
         "observed": f"{adverse['ecl_delta_pct']:.1%}",
         "fired": bool(adverse["ecl_delta_pct"] >= TRIGGER_DECL["ecl_increase"])},
        {"trigger": "Stressed coverage (ECL / EAD)",
         "threshold": f">= {TRIGGER_DECL['coverage']:.0%}",
         "observed": f"{results['coverage'].max():.1%}",
         "fired": bool(results["coverage"].max() >= TRIGGER_DECL["coverage"])},
        {"trigger": "Stage 2 share of exposure",
         "threshold": f">= {TRIGGER_DECL['stage2_share']:.0%}",
         "observed": f"{results['stage2_share'].max():.1%}",
         "fired": bool(results["stage2_share"].max() >= TRIGGER_DECL["stage2_share"])},
    ])
    return results, triggers
