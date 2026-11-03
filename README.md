# Credit stress testing: PD model, macro satellite, expected loss

A small, end-to-end demonstration of how a credit stress test is built: a
loan-level probability of default model, a macro satellite model that turns a
scenario into a PD multiplier, and the expected-loss, staging and overlay
consequences.

Everything runs from two public sources with one command.

```bash
pip install -r requirements.txt
python -m credit_stress.run     # stress test -> output/ and docs/index.html
python -m credit_stress.limits  # limits     -> output/ and docs/limits.html
pytest -q
```

Both sources are committed, so the pipeline runs offline: the FRED series as
CSV, and the UCI loan file as the original workbook.

## The chain

```
loan file ──► PD model (calibrated) ──► PD multiplier ◄── satellite model ◄── macro series
                                              │
                                              ▼
                              ECL, coverage, staging, overlay triggers
```

### 1. Loan-level PD

30,000 accounts from the UCI *default of credit card clients* file. A logistic
regression and a gradient-boosted classifier are both fitted on a 70/30
stratified split, isotonic-calibrated, and compared on out-of-sample AUC and KS.

The score ranking comes from the model. The **level** does not: the source file
carries a one-month default label whose base rate is not a portfolio central
tendency. PDs are therefore shifted in log-odds so the portfolio mean sits at a
stated 3.5% (12-month), which is how a scorecard is actually calibrated in a
bank. That step is an assumption and is labelled as one.

### 2. Macro satellite

The quarterly change in the US card delinquency rate is regressed on the change
in unemployment, GDP growth and house price growth (1991-2019, 115 quarters,
R-squared 0.199). A change specification is used because the delinquency level
is close to a random walk; a lagged level produces a near-unit root and
coefficients with the wrong sign.

Scenarios are 12-quarter paths that build, hold and fade. The fitted equation is
integrated forward and compared with the baseline path quarter by quarter; the
largest ratio is the PD multiplier.

### 3. Stress, staging and triggers

PD x multiplier, LGD of 65% (assumed), EAD at current balance. Stage 2 is
flagged above a 3.5% PD floor. Three overlay triggers are evaluated and the
result is reported as a trigger for a management overlay, not as a forecast.

## Results

| Scenario | PD multiplier | Mean PD | ECL | Coverage | Stage 2 share |
|---|---|---|---|---|---|
| Baseline | 1.00x | 3.50% | 33.9m | 2.20% | 19% |
| Adverse | 1.35x | 4.71% | 45.6m | 2.97% | 22% |
| Severe | 1.64x | 5.73% | 55.4m | 3.60% | 25% |

Adverse expected loss is +35% against baseline and breaches the overlay
threshold; severe is +64%. Amounts are in the source file's local currency
(TWD) and are not converted.

## Risk appetite and limits

`python -m credit_stress.limits` adds the layer a limit framework actually asks
for: **how far is each bucket from its limit**. A register is set on four cuts
(single name, sector, country, sub-portfolio), and every bucket reports
utilisation, headroom in currency, headroom as a share of the limit, and a state
(within appetite / early warning at 85% / breach at 100%) that maps to an owner
and an action.

The three layers are kept apart - capacity, appetite, limit - so the reader can
see which one binds. Limits set on different dimensions overlap, so only
same-dimension sums are additive; the note says so rather than adding them up.

| Layer | Amount | Used | Utilisation | Headroom |
|---|---|---|---|---|
| Capacity (1.50x) | 2,602m | 1,735m | 66.7% | 867m |
| Appetite (1.15x) | 1,995m | 1,735m | 87.0% | 260m |

One bucket breaches and four sit in early warning, which is what the register is
sized to demonstrate.

## What this is not

- **Not a through-the-cycle PD model.** The loan file is a single period with no
  genuine vintage dimension.
- **Not a calibrated portfolio view.** The loan file is Taiwanese retail credit
  and the macro satellite is fitted on US aggregates. The two links are each
  estimated on real data; joining them for a specific portfolio is a separate
  exercise.
- **Not production ECL.** One-year horizon, constant LGD, no discounting, no
  amortisation, no cure or recovery modelling.
- **No IFRS 9 staging logic** beyond a PD floor, and no climate module.

The value is in the chain and in the documentation of where it breaks, which is
what a stress-testing team spends most of its time on.

## Layout

```
credit_stress/
  data.py       loan file and macro series, feature construction
  model.py      PD models, calibration, AUC/KS, permutation importance
  satellite.py  macro regression, scenario paths, PD multipliers
  stress.py     expected loss, coverage, staging, overlay triggers
  limits.py     risk appetite register, utilisation, headroom and escalation
  report.py     commentary and the one-page HTML note
  run.py        the pipeline
tests/          calibration, ordering and reconciliation checks
```
