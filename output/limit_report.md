# Credit risk appetite - limit utilisation note

**Book.** 320 obligors, exposure 1,735m, expected loss 58.3m.

## Distance to the three layers

| Layer | Amount | Used | Utilisation | Headroom |
|---|---|---|---|---|
| Capacity | 2,602m | 1,735m | 66.7% | 867m |
| Appetite | 1,995m | 1,735m | 87.0% | 260m |
| Sum of sector limits | 2,541m | 1,735m | 68.3% | 807m |

The sector limits sum to more than appetite because limits are set on four dimensions that overlap - an obligor sits inside a sector and a country. Only same-dimension sums are additive, which is why utilisation is read bucket by bucket and never across cuts.

Three cuts are complete and reconcile to the same 1,735m: sub-portfolio, sector and country. The single-name register holds the ten largest names, which cover 15% of the book - a limit is not set on every small obligor, so that cut is deliberately partial.

## Worst five buckets against their limits

| Dimension | Bucket | Exposure | Limit | Utilisation | Headroom | State |
|---|---|---|---|---|---|---|
| obligor | Obligor 0103 | 42.2m | 40.5m | 104% | -1.6m | Breach |
| sector | Shipping | 166.6m | 181.8m | 92% | 15.2m | Early warning |
| obligor | Obligor 0305 | 21.5m | 24.1m | 89% | 2.6m | Early warning |
| country | DK | 168.7m | 190.8m | 88% | 22.1m | Early warning |
| obligor | Obligor 0291 | 20.7m | 23.5m | 88% | 2.8m | Early warning |

## Judgement
1 bucket(s) breach and 7 sit in early warning. The largest single-name exposure is at 104% of its limit, which is the number a committee acts on; the aggregate sector utilisation of 68% is the number it notes.

## What the framework asserts
- Every bucket reports utilisation, headroom and a state, never a bare amount.
- Every state maps to an action and an owner, so the note ends in a decision.
- The four cuts reconcile to the same total; a report that cannot be added up is not a limit report.

## Limitations
- The book is illustrative: the loan-level file used for modelling has no sector or country dimension, so the limit layer is demonstrated on a generated book.
- Limits are sized to exercise every RAG state rather than taken from a real appetite.
- Capacity and appetite multiples (1.50x and 1.15x) are assumptions, not calibrations.
