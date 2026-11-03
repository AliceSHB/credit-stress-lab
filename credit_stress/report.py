"""Reporting layer: commentary a risk committee could read, plus a one-pager."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent.parent / "output"
DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"


def commentary(metrics: dict, sat: dict, results: pd.DataFrame, triggers: pd.DataFrame) -> str:
    best = metrics["best_model"]
    m = metrics["models"][best]
    base = results.set_index("scenario").loc["baseline"]
    adv = results.set_index("scenario").loc["adverse"]
    sev = results.set_index("scenario").loc["severe"]
    fired = triggers[triggers["fired"]]["trigger"].tolist()

    lines = [
        "# Credit stress test - portfolio note",
        "",
        f"**Portfolio.** {metrics['n_train'] + metrics['n_test']:,} accounts, "
        f"{base['ead']/1e6:,.0f}m local-currency exposure at default, observed default rate "
        f"{metrics['test_default_rate']:.2%}.",
        "",
        "## Probability of default",
        f"The {best} model scores AUC {m['auc']:.3f} and KS {m['ks']:.3f} out of sample, with a "
        f"Brier score of {m['brier']:.4f}. Ranking comes from the model; the level is shifted in "
        f"log-odds to a {metrics['pd_calibration']['target_mean_pd_12m']:.1%} twelve-month central "
        f"tendency, because the source file's one-month label rate "
        f"({metrics['test_default_rate']:.1%}) is not a portfolio central tendency.",
        "",
        "## Satellite model",
        f"The quarterly change in card delinquency is regressed on the change in unemployment, "
        f"GDP growth and house price growth over {sat['sample'][0]} to {sat['sample'][1]} "
        f"({sat['n_obs']} quarters), R-squared {sat['r2']:.3f}. The unemployment term carries a "
        f"t-statistic of {sat['t_stats']['delta_unemployment']:.1f}; the GDP and house price terms "
        f"are not significant individually, which is what this data says and not what a textbook "
        f"would predict.",
        "",
        "## Stress results",
        "",
        "| Scenario | PD multiplier | Mean 12m PD | ECL | Coverage | Stage 2 share |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['scenario']} | {r['pd_multiplier']:.2f}x | {r['mean_pd_12m']:.2%} | "
            f"{r['ecl']/1e6:,.1f}m | {r['coverage']:.2%} | {r['stage2_share']:.0%} |"
        )
    lines += [
        "",
        f"Under the adverse scenario expected loss rises by {adv['ecl_delta']/1e6:,.1f}m "
        f"({adv['ecl_delta_pct']:+.0%}); under the severe scenario by "
        f"{sev['ecl_delta']/1e6:,.1f}m ({sev['ecl_delta_pct']:+.0%}). Figures are in the source "
        f"file's local currency and are not converted.",
        "",
        "## Overlay triggers",
    ]
    for _, t in triggers.iterrows():
        lines.append(f"- {'**FIRED**' if t['fired'] else 'not fired'} - {t['trigger']}: "
                     f"observed {t['observed']} (threshold {t['threshold']})")
    lines += [
        "",
        "## Judgement",
        ("The adverse scenario breaches the overlay threshold, so the result is not a "
         "forecast but a trigger for a management overlay discussion."
         if fired else
         "No overlay threshold is breached; the result stays an analytical input rather "
         "than a call to action."),
        "",
        "## Limitations",
        "- Loan-level data is a single-period retail book; there is no genuine vintage "
        "dimension, so the PD model is not a through-the-cycle model.",
        "- PD levels are calibrated to an assumed central tendency; the model's discrimination "
        "is estimated, its level is a judgement.",
        "- LGD is assumed, not estimated; EAD is current balance with no amortisation path.",
        "- The satellite is fitted on US aggregate delinquency and applied to a Taiwanese "
        "loan file. It demonstrates the translation step, not a calibrated portfolio view.",
    ]
    return "\n".join(lines) + "\n"


def html(metrics: dict, sat: dict, results: pd.DataFrame, triggers: pd.DataFrame, text: str) -> str:
    best = metrics["best_model"]
    m = metrics["models"][best]
    rows = "".join(
        f"<tr><td>{r['scenario']}</td><td>{r['pd_multiplier']:.2f}x</td>"
        f"<td>{r['mean_pd_12m']:.2%}</td><td>{r['ecl']/1e6:,.1f}m</td>"
        f"<td>{r['coverage']:.2%}</td><td>{r['stage2_share']:.0%}</td></tr>"
        for _, r in results.iterrows()
    )
    trig = "".join(
        f"<li class='{ 'hot' if t['fired'] else '' }'>{t['trigger']} - {t['observed']} "
        f"(threshold {t['threshold']})</li>"
        for _, t in triggers.iterrows()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Credit stress test</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:900px;margin:48px auto;color:#1c2230;line-height:1.55}}
h1{{font-size:26px;border-bottom:3px solid #1F3864;padding-bottom:10px}}
h2{{font-size:16px;color:#1F3864;margin-top:34px;text-transform:uppercase;letter-spacing:.06em}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:10px}}
th,td{{border-bottom:1px solid #e3e6ec;padding:8px 10px;text-align:left}}
th{{background:#f4f6fa;font-weight:600}}
.kpi{{display:flex;gap:18px;margin:18px 0;flex-wrap:wrap}}
.kpi div{{background:#f4f6fa;padding:14px 18px;border-radius:6px;min-width:150px}}
.kpi b{{display:block;font-size:22px;color:#1F3864}}
li.hot{{color:#a4262c;font-weight:600}}
</style></head><body>
<nav style="font-size:13px;margin-bottom:6px"><a href="index.html" style="color:#1F3864;margin-right:14px">Stress test</a><a href="limits.html" style="color:#1F3864">Risk appetite &amp; limits</a></nav>\n<h1>Credit stress test - portfolio note</h1>
<div class="kpi">
<div><span>Model</span><b>{best}</b></div>
<div><span>AUC</span><b>{m['auc']:.3f}</b></div>
<div><span>KS</span><b>{m['ks']:.3f}</b></div>
<div><span>Satellite R&sup2;</span><b>{sat['r2']:.3f}</b></div>
</div>
<h2>Stress results</h2>
<table><tr><th>Scenario</th><th>PD multiplier</th><th>Mean 12m PD</th><th>ECL</th>
<th>Coverage</th><th>Stage 2 share</th></tr>{rows}</table>
<h2>Overlay triggers</h2><ul>{trig}</ul>
<h2>Method</h2>
<p>Loan-level PD from a calibrated gradient-boosted model; a macro satellite model maps the
cycle into a PD multiplier; expected loss, coverage and staging follow. Numbers are on a
public retail credit file and are a demonstration of method, not a portfolio view.</p>
</body></html>"""
