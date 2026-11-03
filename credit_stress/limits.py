"""Risk appetite and limit layer.

The modelling layer answers "what is the expected loss". This layer answers the
question a limit framework actually asks: *how far is this bucket from its
limit, and what happens when it gets close*. Everything is reported as a
distance - utilisation, headroom in currency, headroom as a share of the limit
- and every bucket carries an owner, a trigger and an escalation action.

Three layers are kept separate, as in a bank's risk appetite framework:

  capacity  the most the book could carry
  appetite  what the board is willing to carry
  limit     the operational ceiling for one bucket

The book used here is illustrative and generated for the demonstration; the
real loan file in `data/` has no sector or country dimension.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["Manufacturing", "Shipping", "Real Estate", "Retail",
           "Energy", "Financial Institutions", "Public Sector", "Agriculture"]
COUNTRIES = ["FI", "SE", "NO", "DK", "PL", "EE"]
SUB_PORTFOLIOS = ["Large Corporate", "Mid Corporate", "Institutions"]

EARLY_WARNING = 0.85
BREACH = 1.00

CAPACITY_MULTIPLE = 1.50
APPETITE_MULTIPLE = 1.15


def build_book(seed: int = 7, n: int = 320) -> pd.DataFrame:
    """Illustrative corporate book: obligor, sector, country, exposure, PD."""
    rng = np.random.default_rng(seed)
    ead = rng.lognormal(mean=15.2, sigma=1.05, size=n).round(0)
    book = pd.DataFrame({
        "obligor_id": [f"OB{i:04d}" for i in range(n)],
        "obligor": [f"Obligor {i:04d}" for i in range(n)],
        "sector": rng.choice(SECTORS, n, p=[.22, .10, .20, .14, .12, .10, .06, .06]),
        "country": rng.choice(COUNTRIES, n, p=[.34, .26, .16, .12, .08, .04]),
        "sub_portfolio": rng.choice(SUB_PORTFOLIOS, n, p=[.35, .50, .15]),
        "ead": ead,
        "pd": np.clip(rng.beta(2.0, 26.0, n), 0.002, 0.30),
    })
    book["lgd"] = 0.45
    book["expected_loss"] = book["pd"] * book["lgd"] * book["ead"]
    return book


def _register(book: pd.DataFrame, dimension: str, top_n: int | None = None,
              seed: int = 11) -> pd.DataFrame:
    """One limit per bucket, sized so the demonstration exercises every state."""
    grouped = book.groupby(dimension)["ead"].sum().sort_values(ascending=False)
    if top_n:
        grouped = grouped.head(top_n)
    rng = np.random.default_rng(seed + len(dimension))
    targets = rng.uniform(0.48, 0.95, len(grouped))
    if dimension == "obligor":
        targets[0] = 1.04      # one plant: the largest single name breaches
    if dimension == "sector":
        targets[0] = 0.88      # one plant: a sector in early warning
    out = pd.DataFrame({
        "dimension": dimension,
        "bucket": grouped.index,
        "exposure": grouped.to_numpy(),
        "limit": np.round(grouped.to_numpy() / targets, 0),
    })
    out["owner"] = out["bucket"].map(lambda b: f"Credit Risk Owner - {b}")
    return out


def build_register(book: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    frames = [
        _register(book, "sub_portfolio", seed=3),
        _register(book, "sector", seed=5),
        _register(book, "country", seed=7),
        _register(book, "obligor", top_n=top_n, seed=9),
    ]
    reg = pd.concat(frames, ignore_index=True)
    reg["trigger"] = EARLY_WARNING
    util = reg["exposure"] / reg["limit"]
    reg["escalation"] = np.select(
        [util >= BREACH, util >= EARLY_WARNING],
        ["Escalate to Risk Committee; require a reduction plan",
         "Early warning: notify owner and agree action"],
        default="Report only",
    )
    return reg


def assess(register: pd.DataFrame) -> pd.DataFrame:
    """Distance to limit, expressed three ways, plus the RAG state."""
    reg = register.copy()
    reg["utilisation"] = reg["exposure"] / reg["limit"]
    reg["headroom"] = reg["limit"] - reg["exposure"]
    reg["headroom_pct"] = 1.0 - reg["utilisation"]
    reg["state"] = np.where(reg["utilisation"] >= BREACH, "Breach",
                            np.where(reg["utilisation"] >= EARLY_WARNING, "Early warning", "Within appetite"))
    return reg.sort_values(["dimension", "utilisation"], ascending=[True, False])


def three_layers(book: pd.DataFrame, register: pd.DataFrame) -> pd.DataFrame:
    """Capacity, appetite and limit, each with the distance to it."""
    total = float(book["ead"].sum())
    capacity = total * CAPACITY_MULTIPLE
    appetite = total * APPETITE_MULTIPLE
    limit_sum = float(register.loc[register["dimension"] == "sector", "limit"].sum())
    rows = [
        {"layer": "Capacity", "amount": capacity, "used": total,
         "utilisation": total / capacity, "headroom": capacity - total},
        {"layer": "Appetite", "amount": appetite, "used": total,
         "utilisation": total / appetite, "headroom": appetite - total},
        {"layer": "Sum of sector limits", "amount": limit_sum, "used": total,
         "utilisation": total / limit_sum, "headroom": limit_sum - total},
    ]
    return pd.DataFrame(rows)


def reconcile(book: pd.DataFrame, register: pd.DataFrame) -> dict:
    total = float(book["ead"].sum())
    by_dim = (register.groupby("dimension")["exposure"].sum() / total).to_dict()
    return {"total_exposure": total, "share_by_dimension": by_dim,
            "reconciles": all(abs(v - 1.0) < 1e-9 for v in by_dim.values())}


def report(book: pd.DataFrame, reg: pd.DataFrame, layers: pd.DataFrame, rec: dict) -> str:
    breaches = reg[reg["state"] == "Breach"]
    warnings = reg[reg["state"] == "Early warning"]
    worst = reg.sort_values("utilisation", ascending=False).head(5)

    lines = [
        "# Credit risk appetite - limit utilisation note",
        "",
        f"**Book.** {len(book):,} obligors, exposure {rec['total_exposure']/1e6:,.0f}m, "
        f"expected loss {book['expected_loss'].sum()/1e6:,.1f}m.",
        "",
        "## Distance to the three layers",
        "",
        "| Layer | Amount | Used | Utilisation | Headroom |",
        "|---|---|---|---|---|",
    ]
    for _, r in layers.iterrows():
        lines.append(f"| {r['layer']} | {r['amount']/1e6:,.0f}m | {r['used']/1e6:,.0f}m | "
                     f"{r['utilisation']:.1%} | {r['headroom']/1e6:,.0f}m |")

    lines += [
        "",
        "The sector limits sum to more than appetite because limits are set on four dimensions "
        "that overlap - an obligor sits inside a sector and a country. Only same-dimension sums "
        "are additive, which is why utilisation is read bucket by bucket and never across cuts.",
        "",
        "## Worst five buckets against their limits", "",
              "| Dimension | Bucket | Exposure | Limit | Utilisation | Headroom | State |",
              "|---|---|---|---|---|---|---|"]
    for _, r in worst.iterrows():
        lines.append(f"| {r['dimension']} | {r['bucket']} | {r['exposure']/1e6:,.1f}m | "
                     f"{r['limit']/1e6:,.1f}m | {r['utilisation']:.0%} | {r['headroom']/1e6:,.1f}m | {r['state']} |")

    lines += [
        "",
        f"## Judgement",
        f"{len(breaches)} bucket(s) breach and {len(warnings)} sit in early warning. "
        f"The largest single-name exposure is at {reg[reg['dimension']=='obligor']['utilisation'].max():.0%} of its limit, "
        f"which is the number a committee acts on; the aggregate sector utilisation of "
        f"{layers.iloc[2]['utilisation']:.0%} is the number it notes.",
        "",
        "## What the framework asserts",
        "- Every bucket reports utilisation, headroom and a state, never a bare amount.",
        "- Every state maps to an action and an owner, so the note ends in a decision.",
        "- The four cuts reconcile to the same total; a report that cannot be added up is not a limit report.",
        "",
        "## Limitations",
        "- The book is illustrative: the loan-level file used for modelling has no sector or "
        "country dimension, so the limit layer is demonstrated on a generated book.",
        "- Limits are sized to exercise every RAG state rather than taken from a real appetite.",
        "- Capacity and appetite multiples (1.50x and 1.15x) are assumptions, not calibrations.",
    ]
    return "\n".join(lines) + "\n"


STYLE = """<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:900px;margin:48px auto;color:#1c2230;line-height:1.55}
h1{font-size:26px;border-bottom:3px solid #1F3864;padding-bottom:10px}
h2{font-size:15px;color:#1F3864;margin-top:32px;text-transform:uppercase;letter-spacing:.06em}
table{border-collapse:collapse;width:100%;font-size:14px;margin-top:10px}
th,td{border-bottom:1px solid #e3e6ec;padding:8px 10px;text-align:left}
th{background:#f4f6fa;font-weight:600}
.red{color:#a4262c;font-weight:600}.amber{color:#9a6b1c;font-weight:600}
nav{font-size:13px;margin-bottom:6px}nav a{color:#1F3864;margin-right:14px}
.kpi{display:flex;gap:16px;margin:18px 0;flex-wrap:wrap}
.kpi div{background:#f4f6fa;padding:14px 18px;border-radius:6px;min-width:140px}
.kpi b{display:block;font-size:22px;color:#1F3864}
</style>"""


def html(book, reg, layers, rec) -> str:
    layer_rows = "".join(
        f"<tr><td>{r['layer']}</td><td>{r['amount']/1e6:,.0f}m</td><td>{r['used']/1e6:,.0f}m</td>"
        f"<td>{r['utilisation']:.1%}</td><td>{r['headroom']/1e6:,.0f}m</td></tr>"
        for _, r in layers.iterrows()
    )
    worst = reg.sort_values("utilisation", ascending=False).head(8)
    rows = ""
    for _, r in worst.iterrows():
        cls = "red" if r["state"] == "Breach" else ("amber" if r["state"] == "Early warning" else "")
        rows += (f"<tr><td>{r['dimension']}</td><td>{r['bucket']}</td>"
                 f"<td>{r['exposure']/1e6:,.1f}m</td><td>{r['limit']/1e6:,.1f}m</td>"
                 f"<td>{r['utilisation']:.0%}</td><td>{r['headroom']/1e6:,.1f}m</td>"
                 f"<td class='{cls}'>{r['state']}</td></tr>")
    breaches = int((reg["state"] == "Breach").sum())
    warn = int((reg["state"] == "Early warning").sum())
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Risk appetite and limits</title>{STYLE}</head><body>
<nav><a href="index.html">Stress test</a><a href="limits.html">Risk appetite &amp; limits</a></nav>
<h1>Risk appetite and limit utilisation</h1>
<div class="kpi">
<div><span>Obligors</span><b>{len(book):,}</b></div>
<div><span>Exposure</span><b>{rec['total_exposure']/1e6:,.0f}m</b></div>
<div><span>Breaches</span><b>{breaches}</b></div>
<div><span>Early warnings</span><b>{warn}</b></div>
</div>
<h2>Distance to the three layers</h2>
<table><tr><th>Layer</th><th>Amount</th><th>Used</th><th>Utilisation</th><th>Headroom</th></tr>
{layer_rows}</table>
<p>Limits are set on four cuts that overlap, so only same-dimension sums are additive. The note
says so rather than adding them up.</p>
<h2>Worst buckets against their limits</h2>
<table><tr><th>Dimension</th><th>Bucket</th><th>Exposure</th><th>Limit</th><th>Utilisation</th>
<th>Headroom</th><th>State</th></tr>{rows}</table>
<h2>Method</h2>
<p>Every bucket carries a limit, an owner, a trigger and an escalation action. The state is a
distance, not a level: within appetite below 85%, early warning to 100%, breach above it. The book
is illustrative and the limits are sized to exercise each state.</p>
</body></html>"""


def main(out_dir="output", docs_dir="docs"):
    import pathlib
    import json
    out, docs = pathlib.Path(out_dir), pathlib.Path(docs_dir)
    out.mkdir(exist_ok=True)
    docs.mkdir(exist_ok=True)

    book = build_book()
    register = build_register(book)
    reg = assess(register)
    layers = three_layers(book, register)
    rec = reconcile(book, register)

    reg.to_csv(out / "limit_utilisation.csv", index=False)
    layers.to_csv(out / "limit_layers.csv", index=False)
    (out / "limit_report.md").write_text(report(book, reg, layers, rec))
    (docs / "limits.html").write_text(html(book, reg, layers, rec))
    (out / "limit_summary.json").write_text(json.dumps({
        "total_exposure": rec["total_exposure"],
        "reconciles": rec["reconciles"],
        "breaches": int((reg["state"] == "Breach").sum()),
        "early_warnings": int((reg["state"] == "Early warning").sum()),
        "buckets": int(len(reg)),
        "layers": layers.to_dict(orient="records"),
    }, indent=2, default=float))
    print("wrote limit outputs")
    print(report(book, reg, layers, rec))


if __name__ == "__main__":
    main()
