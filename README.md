# Credit stress testing: PD model, macro satellite, expected loss

A small, end-to-end demonstration of how a credit stress test is built: a
loan-level probability of default model, a macro satellite model that turns a
scenario into a PD multiplier, and the expected-loss, staging and overlay
consequences.

Everything runs from two public sources with one command.

```bash
pip install -r requirements.txt
python -m credit_stress.run     # writes output/ and docs/index.html
pytest -q
```

The macro series (`data/fred_*.csv`) are committed so the satellite model is
reproducible offline. The loan file is not committed; download it once from the
UCI repository into `data/`:

```bash
curl -L -o data/uci.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
unzip -o data/uci.zip -d data/
```

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
  report.py     commentary and the one-page HTML note
  run.py        the pipeline
tests/          calibration, ordering and reconciliation checks
```
