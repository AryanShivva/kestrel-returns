# Evidence that it works, and how often it does not

**Validation design.** Rolling-origin, out-of-time: train on everything before a quarter, score that quarter (Oct-Dec 2025, Jan-Mar 2026, Apr-Jun 2026). This mimics the real use (train on the past, score next month). The test file is Jul-Sep 2026, so a random split would have flattered us. 6,334 out-of-time orders, 11.3% returned.
Data cleaned first: 651 duplicate partner-feed rows removed (10,504 unique orders), 700 October-2025 order values divided by 100.

## Headline numbers (`cv_folds.csv`, `summary.json`)
| Metric | Result |
|---|---|
| ROC-AUC, mean of 3 quarters | **0.775** (0.764 / 0.774 / 0.787); pooled 95% bootstrap CI 0.755-0.792 |
| PR-AUC (random = 0.113) | 0.383 |
| Accuracy at 0.5 cut | 89.4% vs **88.7% by never flagging anything** |
| Calibration | predicted vs actual within about 1-2 points in every decile (`calibration.csv`) |
| Gradient-boosted trees (tried) | AUC 0.738-0.775, no better, thrown away |

## Why 95% accuracy is not available
Only 11.3% of orders are returned, so "no returns" is already 88.7% accurate. 95% needs an error rate of 5% while the model is wrong on most of the returns it can see.
The only way to hit 95%+ is leakage: adding `last_service_event_type` / `pickup_scheduled_at` gives AUC 0.998 and accuracy 99.2% in the same validation, but those fields only exist once a return has started (REVERSE_PICKUP = 789 of 789 returned; pickup booked on 94% of returns; INSTALL_DONE/DEMO_DONE = 0 returns, because installed products do not come back). In the test file these columns are empty or `INSTALL_BOOKED`, a value that never occurs in training, so a model using them would fail on exactly the orders that matter.

## How often it is wrong (`lift_table.csv`)
| Flag the riskiest... | Orders (of 6,334) | Precision | Share of all returns caught | Wrongly flagged |
|---|---|---|---|---|
| 5% | 316 | 53% | 23% | 47% |
| 10% | 633 | 40% | 36% | 60% |
| 20% | 1,266 | 31% | 54% | 69% |
| 30% | 1,900 | 25% | 66% | 75% |

At the call break-even (risk >= 11.2%) about a third of orders are flagged, around 66% of returns are inside the flagged group, and roughly 3 in 4 flagged orders are fine. **Every flag list is mostly false alarms**, which is why the action must be cheap (a call), not costly (a hold).

## Kinds of case it gets wrong (`segments.csv`)
- Robot vacuums (AUC 0.73, return rate 20%) and room heaters (0.73): returns look driven by things not in the data (fit, noise, expectation).
- Cash on delivery (AUC 0.73): high base rate, but the model separates good from bad orders less well.
- First-time buyers with no history: nothing to go on except product/payment/promise.
- Shield members: return rate is double (19% vs 9%) and the model ranks them fine (AUC 0.76), but they generate 568 of ~1,540 false flags in this sample. Handle with a friendly call, never a hold.

## Economics on out-of-time data (`call_economics.csv`, `hold_economics.csv`, `charts.png`)
Policy inputs: return Rs 1,150; call Rs 45; a call prevents 35% of returns; a hold makes 12% of customers cancel.
- **Call break-even risk = 45 / (0.35 x 1,150) = 11.2%.** Call every order above it: net **+Rs 11,500 per 700 orders** (224 calls, Rs 10,100 cost, Rs 21,600 saved). Calling everyone: +Rs 400 (about zero).
- **Hold loses money** unless margin lost on cancelled orders is below about 4%: with a 10% margin, holding the top 10% loses about Rs 6,300 a month; with 20%, about Rs 16,400. Holding does not stop a return, it only gives the 12% who cancel the chance to.
