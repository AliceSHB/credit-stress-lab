# Credit stress test - portfolio note

**Portfolio.** 30,000 accounts, 1,537m local-currency exposure at default, observed default rate 22.12%.

## Probability of default
The boosted model scores AUC 0.781 and KS 0.421 out of sample, with a Brier score of 0.1350. Ranking comes from the model; the level is shifted in log-odds to a 3.5% twelve-month central tendency, because the source file's one-month label rate (22.1%) is not a portfolio central tendency.

## Satellite model
The quarterly change in card delinquency is regressed on the change in unemployment, GDP growth and house price growth over 1991-06-30 to 2019-12-31 (115 quarters), R-squared 0.199. The unemployment term carries a t-statistic of 4.4; the GDP and house price terms are not significant individually, which is what this data says and not what a textbook would predict.

## Stress results

| Scenario | PD multiplier | Mean 12m PD | ECL | Coverage | Stage 2 share |
|---|---|---|---|---|---|
| baseline | 1.00x | 3.50% | 33.9m | 2.20% | 19% |
| adverse | 1.35x | 4.71% | 45.6m | 2.97% | 22% |
| severe | 1.64x | 5.73% | 55.4m | 3.60% | 25% |

Under the adverse scenario expected loss rises by 11.8m (+35%); under the severe scenario by 21.5m (+64%). Figures are in the source file's local currency and are not converted.

## Overlay triggers
- **FIRED** - Expected loss increase under the adverse scenario: observed 34.7% (threshold >= 25% of baseline ECL)
- not fired - Stressed coverage (ECL / EAD): observed 3.6% (threshold >= 8%)
- not fired - Stage 2 share of exposure: observed 24.6% (threshold >= 30%)

## Judgement
The adverse scenario breaches the overlay threshold, so the result is not a forecast but a trigger for a management overlay discussion.

## Limitations
- Loan-level data is a single-period retail book; there is no genuine vintage dimension, so the PD model is not a through-the-cycle model.
- PD levels are calibrated to an assumed central tendency; the model's discrimination is estimated, its level is a judgement.
- LGD is assumed, not estimated; EAD is current balance with no amortisation path.
- The satellite is fitted on US aggregate delinquency and applied to a Taiwanese loan file. It demonstrates the translation step, not a calibrated portfolio view.
