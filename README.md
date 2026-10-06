# Kestrel Home: returns-risk before dispatch

A small, free-to-run service that scores one order for return risk and tells a Kestrel employee what to do and why.
No paid API, no network calls, no API key. The model is a logistic regression stored as plain JSON (`model.json`).

**Headline:** ranking quality (ROC-AUC) is about **0.775** out-of-time. Accuracy is about 89.4% against an 88.7% "predict no returns" baseline. **The 95% target is not reachable honestly** (see `evidence/EVIDENCE.md`).
It supports a **confirmation-call** decision (about +Rs 11,500/month at 700 orders), not a **hold** decision (loses money).

## Run it (clean machine, Python 3.10+)
```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install flask pandas numpy
python app.py                                            # http://localhost:8000
```
Open http://localhost:8000 for the screen. `model.json` is committed, so no training is needed to run.

### Endpoint
```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
 "order_placed_at":"2026-07-14 23:05","customer_id":"KC100001","shield_member":"N","sku":"KH-RV-02",
 "sales_channel":"marketplace","payment_mode":"cod","discount_pct":22,"qty":1,"order_value_inr":17159,
 "promised_delivery_days":9,"delivery_pincode":"400151","is_gift":"Y",
 "customer_prior_orders":4,"customer_prior_returns":2,"delivery_note":""}'
```
Returns `return_risk` (0-1), `risk_band`, `action` (`CALL_BEFORE_DISPATCH` or `SHIP`), a plain-English `message`,
`expected_saving_rs`, `reasons` (top drivers, up/down), `data_notes` (e.g. "value looked 100x too large, corrected") and a `caveat`.
Bad input gives a 4xx JSON error, never a stack trace. `GET /health` for a liveness check.
`shield_member` is looked up from `data/customers.csv` if that file exists, otherwise pass it in the JSON (default N, with a note).

## Retrain / regenerate the evidence (optional)
Put the client files in `data/` (`train.csv`, `test_unlabelled.csv`, `customers.csv`, `sample_submission.csv`). **They are git-ignored on purpose: policy section 10 forbids publishing customer data.**
```bash
pip install scikit-learn matplotlib
python train.py --data data --out .     # validates, writes model.json, predictions.csv, evidence/*.csv
python make_charts.py                   # evidence/charts.png
```

## What is in the repo
| File | Purpose |
|---|---|
| `app.py`, `templates/index.html` | endpoint + one screen |
| `features.py` | cleaning (de-dup, x100 fix) + features, shared by training and service |
| `train.py`, `make_charts.py` | out-of-time validation, economics, final fit, predictions |
| `model.json` | trained model (coefficients) |
| `predictions.csv` | scores for every row of `test_unlabelled.csv` |
| `evidence/` | `EVIDENCE.md`, fold metrics, lift, calibration, economics, charts |
| `memo.md`, `submission-form.md`, `recording-script.md` | deliverables |

## Decisions I made (the brief was ambiguous)
1. **Dropped `last_service_event_type` and `pickup_scheduled_at`.** They are written *after* a return starts. With them AUC is 0.998; without them 0.775. At dispatch they are empty in the test file anyway.
2. **Kept the metric honest:** AUC / PR-AUC for ranking, plus precision at the top of the list, rather than accuracy.
3. **Recommended calls, not holds**, using the policy's own numbers (section 4 and 7).
4. **Return cost = Rs 1,150** (policy, Finance), not Rs 600 (Ritu's estimate).
5. **Scores are calibrated probabilities**, so "0.40" means roughly 40 in 100 such orders come back.
