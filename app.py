"""Kestrel returns-risk service.  python app.py  ->  http://localhost:8000
No API key, no network calls, no paid model. Pure logistic regression in numpy."""
import json, os
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, render_template
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = json.load(open(os.path.join(HERE, "model.json")))
CUST = None
for p in (os.path.join(HERE, "data", "customers.csv"),):
    if os.path.exists(p):
        CUST = pd.read_csv(p)[["customer_id", "shield_member"]].set_index("customer_id")["shield_member"].to_dict()

REQUIRED = ["order_placed_at", "sku", "sales_channel", "payment_mode", "discount_pct", "qty",
            "order_value_inr", "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"]
DEFAULTS = {"is_gift": "N", "delivery_pincode": 0, "delivery_note": "", "customer_id": ""}

# human wording for each model feature: (label, formatter of raw value)
LABELS = {
 "promised_delivery_days": "Promised delivery time", "prior_returns": "Customer's earlier returns",
 "shield": "Kestrel Shield member (free returns)", "discount_pct": "Discount depth", "cod": "Cash on delivery",
 "pay_cod": "Cash on delivery", "is_gift": "Gift order", "prior_orders": "Customer's order history",
 "list_price": "Product price band", "log_value": "Order value", "ch_marketplace": "Sold via marketplace",
 "ch_partner_outlet": "Sold at partner outlet", "pay_prepaid_upi": "Paid by UPI", "warranty_months": "Warranty length",
 "prior_return_rate": "Customer's past return rate",
}
def explain(feat, x, direction):
    f = {
     "promised_delivery_days": lambda: (f"Promised delivery is {int(x)} days (typical is 5); longer promises are returned more" if x > 5 else f"Short promise of {int(x)} days (typical is 5); fast delivery is returned less"),
     "prior_returns": lambda: (f"Customer has returned {int(x)} earlier order(s); repeat returners are the strongest signal" if x else "No earlier returns by this customer"),
     "shield": lambda: "Shield member: free 30-day returns make returns cheap for the customer" if x else "Not a Shield member (Shield members return about twice as often)",
     "discount_pct": lambda: (f"{int(x)}% discount; deeply discounted orders come back more" if x > 10 else f"Low discount ({int(x)}%)"),
     "cod": lambda: "Cash on delivery: roughly twice the return rate of prepaid" if x else "Prepaid order: lower return rate",
     "pay_cod": lambda: "Cash on delivery" if x else "Prepaid/EMI payment",
     "is_gift": lambda: "Marked as a gift" if x else "Not a gift",
     "prior_orders": lambda: f"Customer has {int(x)} earlier orders (frequent buyers return more here)",
     "list_price": lambda: f"Product list price Rs {int(x):,}",
     "log_value": lambda: f"Order value Rs {int(np.expm1(x)):,}",
     "ch_marketplace": lambda: "Marketplace channel returns more than app/web" if x else "Not a marketplace order",
     "ch_partner_outlet": lambda: "Partner-outlet purchases return less" if x else "Not a partner-outlet order",
     "pay_prepaid_upi": lambda: "UPI prepayment: lowest return rate" if x else "Not UPI",
     "warranty_months": lambda: f"{int(x)}-month warranty line ({'fewer' if x >= 24 else 'more'} returns)",
     "prior_return_rate": lambda: f"Past return rate {x:.0%}",
    }.get(feat)
    return f() if f else feat

def score_one(rec):
    notes = []
    missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
    if missing:
        raise ValueError("Missing required field(s): " + ", ".join(missing))
    r = {**DEFAULTS, **rec}
    if r["sku"] not in F.PRODUCTS:
        raise ValueError(f"Unknown sku '{r['sku']}'. Known: {', '.join(sorted(F.PRODUCTS))}")
    if r["sales_channel"] not in F.CHANNELS: raise ValueError("sales_channel must be one of " + ", ".join(F.CHANNELS))
    if r["payment_mode"] not in F.PAYMENTS: raise ValueError("payment_mode must be one of " + ", ".join(F.PAYMENTS))
    try:
        for k in ("discount_pct", "qty", "order_value_inr", "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"):
            r[k] = float(r[k])
        pd.to_datetime(r["order_placed_at"])
    except Exception:
        raise ValueError("Numeric fields must be numbers and order_placed_at a date/time (e.g. 2026-07-01 10:30)")
    if "shield_member" not in r or r.get("shield_member") in (None, ""):
        if CUST is not None and r["customer_id"] in CUST: r["shield_member"] = CUST[r["customer_id"]]
        else:
            r["shield_member"] = "N"; notes.append("Shield status unknown (customer not found / no customer file): assumed NOT a member. Pass shield_member=Y/N to override.")
    df = pd.DataFrame([r])
    X, d = F.build(df)
    if bool(d.value_fixed.iloc[0]): notes.append("Order value looked 100x too large (new payment gateway, paise): divided by 100.")
    if float(r["delivery_pincode"] or 0) == 0: notes.append("No delivery pincode captured (default 000000); pincode is not used by the model.")
    z = (X.iloc[0].values - np.array(MODEL["mean"])) / np.array(MODEL["scale"])
    contrib = z * np.array(MODEL["coef"])
    logit = MODEL["intercept"] + contrib.sum()
    p = float(1 / (1 + np.exp(-logit)))
    order = np.argsort(-np.abs(contrib))
    reasons = []
    for i in order[:12]:
        f = MODEL["features"][i]
        if abs(contrib[i]) < 0.08 or f.startswith("fam_") or f in ("note_flag", "has_pincode", "hour", "dow", "log_value") and abs(contrib[i]) < 0.15: continue
        txt = explain(f, X.iloc[0, i], contrib[i])
        reasons.append({"text": txt, "pushes": "up" if contrib[i] > 0 else "down", "weight": round(float(contrib[i]), 2)})
        if len(reasons) == 5: break
    thr = MODEL["breakeven"]
    shield = r["shield_member"] == "Y"
    if p >= thr:
        action = "CALL_BEFORE_DISPATCH"
        msg = (f"Risk {p:.0%} is above the Rs {45}-call break-even ({thr:.0%}). Make the confirmation call, then ship as normal. "
               "Do not hold the order.")
    else:
        action = "SHIP"; msg = f"Risk {p:.0%} is below the call break-even ({thr:.0%}). Ship as normal."
    if shield and action == "CALL_BEFORE_DISPATCH": msg += " Shield member: keep the call friendly; never hold a Shield order."
    return {"return_risk": round(p, 4), "risk_band": "high" if p >= 0.25 else "elevated" if p >= thr else "normal",
            "typical_risk": round(MODEL["base_rate"], 4), "action": action, "message": msg,
            "expected_saving_rs": round(max(0.0, 0.35 * p * 1150 - 45), 0),
            "reasons": reasons, "data_notes": notes,
            "caveat": "About 6 in 10 orders flagged in the top decile are NOT returned. Ranking quality (AUC) is about 0.78, not 95% accuracy."}

app = Flask(__name__)

@app.get("/")
def home(): return render_template("index.html")

@app.get("/health")
def health(): return {"ok": True, "model_trained_rows": MODEL["trained_rows"], "customer_file_loaded": CUST is not None}

@app.post("/predict")
def predict():
    rec = request.get_json(silent=True)
    if not isinstance(rec, dict):
        return jsonify(error="Send one order as a JSON object, e.g. see README."), 400
    try:
        return jsonify(score_one(rec))
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception:
        return jsonify(error="Could not score this record. Check the field values."), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
