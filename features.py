"""Feature engineering shared by training and the service.

Deliberately EXCLUDED (see evidence/leakage.md): last_service_event_type, pickup_scheduled_at,
source, free-text delivery_note body. Those are written AFTER a return starts (or are
re-import artefacts) and are not known at dispatch.
"""
import re
import numpy as np
import pandas as pd

CHANNELS = ["app", "web", "marketplace", "partner_outlet"]
PAYMENTS = ["prepaid_upi", "prepaid_card", "cod", "emi"]
FAMILIES = ["Air Fryer", "Mixer Grinder", "Water Purifier", "Robot Vacuum",
            "Induction Cooktop", "Ceiling Fan", "Room Heater"]

# Catalogue (public product data, from products.csv)
PRODUCTS = {
 "KH-AF-01": ("Air Fryer", 5199, 12, "2023-11-04"), "KH-AF-02": ("Air Fryer", 6499, 12, "2023-09-07"),
 "KH-AF-03": ("Air Fryer", 8773, 12, "2024-02-02"), "KH-MG-01": ("Mixer Grinder", 3199, 24, "2023-05-27"),
 "KH-MG-02": ("Mixer Grinder", 3999, 24, "2023-09-20"), "KH-MG-03": ("Mixer Grinder", 5398, 24, "2024-12-09"),
 "KH-WP-01": ("Water Purifier", 11999, 12, "2024-09-03"), "KH-WP-02": ("Water Purifier", 14999, 12, "2023-11-19"),
 "KH-WP-03": ("Water Purifier", 20248, 12, "2023-10-16"), "KH-RV-01": ("Robot Vacuum", 17599, 12, "2024-03-06"),
 "KH-RV-02": ("Robot Vacuum", 21999, 12, "2024-01-27"), "KH-RV-03": ("Robot Vacuum", 29698, 12, "2023-10-26"),
 "KH-IC-01": ("Induction Cooktop", 2399, 12, "2024-05-16"), "KH-IC-02": ("Induction Cooktop", 2999, 12, "2023-11-09"),
 "KH-IC-03": ("Induction Cooktop", 4048, 12, "2025-01-28"), "KH-CF-01": ("Ceiling Fan", 2799, 24, "2024-09-27"),
 "KH-CF-02": ("Ceiling Fan", 3499, 24, "2023-10-15"), "KH-CF-03": ("Ceiling Fan", 4723, 24, "2024-10-12"),
 "KH-RH-01": ("Room Heater", 1999, 12, "2023-07-29"), "KH-RH-02": ("Room Heater", 2499, 12, "2024-02-21"),
 "KH-RH-03": ("Room Heater", 3373, 12, "2024-12-23"),
}
PROD_DF = pd.DataFrame(
    [(k,) + v for k, v in PRODUCTS.items()],
    columns=["sku", "family", "list_price_inr", "warranty_months", "launch_date"])
PROD_DF["launch_date"] = pd.to_datetime(PROD_DF["launch_date"])

FEATURES = [
    "discount_pct", "qty", "log_value", "promised_delivery_days", "is_gift", "prior_orders",
    "prior_returns", "prior_return_rate", "shield", "cod", "list_price", "warranty_months",
    "has_pincode", "hour", "dow", "note_flag",
] + [f"ch_{c}" for c in CHANNELS] + [f"pay_{p}" for p in PAYMENTS] + [f"fam_{f}" for f in FAMILIES]


def fix_order_value(df: pd.DataFrame) -> pd.DataFrame:
    """Oct-2025 orders came through the new gateway at x100 (paise). Detect by comparing with
    list_price*qty*(1-discount); if ratio is ~100 divide by 100. Works per row, no date hard-coding."""
    exp = df["list_price_inr"] * df["qty"] * (1 - df["discount_pct"] / 100.0)
    ratio = df["order_value_inr"] / exp.replace(0, np.nan)
    df = df.copy()
    df["value_fixed"] = ratio.between(90, 110).fillna(False)
    df.loc[df["value_fixed"], "order_value_inr"] = df.loc[df["value_fixed"], "order_value_inr"] / 100.0
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Partner-outlet orders appear twice (crm + partner_feed). Keep the crm row."""
    if "source" in df.columns:
        df = df.assign(_p=(df["source"] == "partner_feed").astype(int)).sort_values(["order_id", "_p"])
        df = df.drop_duplicates("order_id", keep="first").drop(columns="_p")
    return df


def build(df: pd.DataFrame, customers: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: raw order rows (any subset of the README columns). Returns model matrix + helper cols."""
    df = df.copy()
    df["order_placed_at"] = pd.to_datetime(df["order_placed_at"])
    df = df.merge(PROD_DF, on="sku", how="left")
    if customers is not None and "shield_member" not in df.columns:
        df = df.merge(customers[["customer_id", "shield_member"]], on="customer_id", how="left")
    if "shield_member" not in df.columns:
        df["shield_member"] = "N"
    df["shield_member"] = df["shield_member"].fillna("N")
    df = fix_order_value(df)
    X = pd.DataFrame(index=df.index)
    X["discount_pct"] = df["discount_pct"].astype(float)
    X["qty"] = df["qty"].astype(float)
    X["log_value"] = np.log1p(df["order_value_inr"].astype(float))
    X["promised_delivery_days"] = df["promised_delivery_days"].astype(float)
    X["is_gift"] = (df["is_gift"] == "Y").astype(int)
    X["prior_orders"] = df["customer_prior_orders"].astype(float)
    X["prior_returns"] = df["customer_prior_returns"].astype(float)
    X["prior_return_rate"] = df["customer_prior_returns"] / df["customer_prior_orders"].clip(lower=1)
    X["shield"] = (df["shield_member"] == "Y").astype(int)
    X["cod"] = (df["payment_mode"] == "cod").astype(int)
    X["list_price"] = df["list_price_inr"].astype(float)
    X["warranty_months"] = df["warranty_months"].astype(float)
    X["has_pincode"] = (df["delivery_pincode"].astype(float) != 0).astype(int)
    X["hour"] = df["order_placed_at"].dt.hour
    X["dow"] = df["order_placed_at"].dt.dayofweek
    note = df.get("delivery_note", pd.Series("", index=df.index)).fillna("").astype(str)
    X["note_flag"] = note.str.contains(r"Call before delivery|HP petrol", regex=True).astype(int)
    for c in CHANNELS: X[f"ch_{c}"] = (df["sales_channel"] == c).astype(int)
    for p in PAYMENTS: X[f"pay_{p}"] = (df["payment_mode"] == p).astype(int)
    for f in FAMILIES: X[f"fam_{f}"] = (df["family"] == f).astype(int)
    return X[FEATURES], df
