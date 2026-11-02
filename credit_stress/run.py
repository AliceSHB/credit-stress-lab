"""End-to-end: loans -> PD model -> satellite -> stress -> report."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from . import report
from .data import load_loans, load_macro
from .model import calibrate_level, fit_and_score, score_population
from .satellite import fit_satellite, scenario_multipliers
from .stress import run_stress

OUT = pathlib.Path(__file__).resolve().parent.parent / "output"
DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
TARGET_PD = 0.035   # 12-month portfolio central tendency, stated assumption


def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)

    print("1/4  loading data")
    loans = load_loans()
    macro = load_macro()
    print(f"     {len(loans):,} accounts | {len(macro)} macro quarters")

    print("2/4  fitting PD models")
    metrics = fit_and_score(loans)
    model = metrics.pop("_model")
    p_test = metrics.pop("_p_test")
    for name, m in metrics["models"].items():
        print(f"     {name:9s} AUC {m['auc']:.4f}  KS {m['ks']:.3f}  Brier {m['brier']:.4f}")

    print("3/4  satellite model")
    sat = fit_satellite(macro)
    print(f"     R2 {sat['r2']:.3f} on quarterly change | {sat['n_obs']} quarters | t(unemployment) {sat['t_stats']['delta_unemployment']:.1f}")
    mult = scenario_multipliers(sat, macro)

    print("4/4  stress")
    raw = score_population(model, loans)
    pd_all, shift = calibrate_level(raw, TARGET_PD)
    metrics["pd_calibration"] = {
        "target_mean_pd_12m": TARGET_PD,
        "log_odds_shift": shift,
        "mean_pd_after_calibration": float(pd_all.mean()),
    }
    results, triggers = run_stress(loans, pd_all, mult)

    # persist
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(metrics["calibration"]).to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(metrics["importance"]).to_csv(OUT / "feature_importance.csv", index=False)
    pd.DataFrame({
        "term": list(sat["coefficients"]),
        "coefficient": list(sat["coefficients"].values()),
        "std_error": list(sat["std_errors"].values()),
        "t_stat": list(sat["t_stats"].values()),
    }).to_csv(OUT / "satellite_coefficients.csv", index=False)
    mult.to_csv(OUT / "scenarios.csv", index=False)
    results.to_csv(OUT / "stress_results.csv", index=False)
    triggers.to_csv(OUT / "overlay_triggers.csv", index=False)

    text = report.commentary(metrics, sat, results, triggers)
    (OUT / "commentary.md").write_text(text)
    (DOCS / "index.html").write_text(report.html(metrics, sat, results, triggers, text))
    print("     wrote output/ and docs/index.html")
    print()
    print(text.split("## Stress results")[1].split("## Overlay")[0].strip())


if __name__ == "__main__":
    main()
