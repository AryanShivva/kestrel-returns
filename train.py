"""Train, validate, and write model.json + predictions.csv + evidence/.
Usage: python train.py --data data/ --out .
Needs data/train.csv, data/test_unlabelled.csv (or test.csv), data/customers.csv.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss, brier_score_loss
import features as F

RETURN_COST, CALL_COST, CALL_PREVENT = 1150.0, 45.0, 0.35   # ops-policy s4, s7
BREAKEVEN = CALL_COST / (CALL_PREVENT * RETURN_COST)          # ~0.112
C = 0.3
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))

def find(d, *names):
    for n in names:
        if os.path.exists(os.path.join(d, n)): return os.path.join(d, n)
    raise FileNotFoundError(names)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default=".")
    a = ap.parse_args(); os.makedirs(f"{a.out}/evidence", exist_ok=True)
    cu = pd.read_csv(find(a.data, "customers.csv"))
    raw = pd.read_csv(find(a.data, "train.csv"))
    te = pd.read_csv(find(a.data, "test_unlabelled.csv", "test.csv"))
    n_raw = len(raw); tr = F.dedupe(raw).sort_values("order_placed_at").reset_index(drop=True)
    X, d = F.build(tr, cu); y = tr["returned"].values; t = pd.to_datetime(tr["order_placed_at"])
    print(f"rows {n_raw} -> {len(tr)} after de-dup; x100 values repaired: {int(d.value_fixed.sum())}")

    # ---- rolling-origin (out-of-time) validation: train on the past, score the next quarter
    folds = [("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01")]
    oot = []; rows = []
    for lo, hi in folds:
        tm = t < lo; vm = (t >= lo) & (t < hi)
        p = mk().fit(X[tm], y[tm]).predict_proba(X[vm])[:, 1]
        pg = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=40,
                                            l2_regularization=1.0).fit(X[tm], y[tm]).predict_proba(X[vm])[:, 1]
        yy = y[vm]
        rows.append(dict(fold=f"{lo}..{hi}", n=int(vm.sum()), base_rate=float(yy.mean()),
                         auc=roc_auc_score(yy, p), pr_auc=average_precision_score(yy, p),
                         acc_at_0_5=accuracy_score(yy, p > .5), acc_majority=1 - float(yy.mean()),
                         logloss=log_loss(yy, p), brier=brier_score_loss(yy, p), gbm_auc=roc_auc_score(yy, pg)))
        oot.append(pd.DataFrame({"p": p, "y": yy, "value": d.loc[vm, "order_value_inr"].values,
                                 "shield": X.loc[vm, "shield"].values, "family": d.loc[vm, "family"].values,
                                 "pay": d.loc[vm, "payment_mode"].values}))
    oot = pd.concat(oot, ignore_index=True)
    cvdf = pd.DataFrame(rows); cvdf.to_csv(f"{a.out}/evidence/cv_folds.csv", index=False)
    print(cvdf.round(4).to_string())

    # bootstrap CI for pooled OOT AUC
    rng = np.random.default_rng(0); bs = []
    for _ in range(500):
        i = rng.integers(0, len(oot), len(oot)); bs.append(roc_auc_score(oot.y.values[i], oot.p.values[i]))
    auc_pool = roc_auc_score(oot.y, oot.p)

    # ---- decision tables on pooled out-of-time predictions
    def table(df):
        out = []
        for k in (0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
            n = int(len(df) * k); top = df.sort_values("p", ascending=False).head(n)
            tp = top.y.sum(); prec = tp / n; rec = tp / df.y.sum()
            out.append(dict(top_share=k, orders=n, precision=prec, recall=rec, lift=prec / df.y.mean(),
                            false_flags_share=1 - prec))
        return pd.DataFrame(out)
    tbl = table(oot); tbl.to_csv(f"{a.out}/evidence/lift_table.csv", index=False)
    print(tbl.round(3).to_string())

    # economics per 700 orders/month: call every order with p >= break-even
    def econ(df, thr, scale=700):
        c = df[df.p >= thr]; n = len(c)
        saved = (CALL_PREVENT * c.y * RETURN_COST).sum(); cost = CALL_COST * n
        return dict(thr=thr, share_called=n / len(df), calls_per_700=n / len(df) * scale,
                    saved_per_700=saved / len(df) * scale, cost_per_700=cost / len(df) * scale,
                    net_per_700=(saved - cost) / len(df) * scale)
    ec = pd.DataFrame([econ(oot, th) for th in (0.08, BREAKEVEN, 0.15, 0.20, 0.25, 0.30, 0.40)])
    ec.to_csv(f"{a.out}/evidence/call_economics.csv", index=False); print(ec.round(1).to_string())
    # call everyone
    all_call = dict(net_per_700=(CALL_PREVENT * oot.y.mean() * RETURN_COST - CALL_COST) * 700)

    # Hold economics (per 700 orders) with margin assumption sweep. Hold -> 12% cancel (policy s7).
    holds = []
    for margin in (0.0, 0.10, 0.20, 0.30):
        for k in (0.05, 0.10, 0.20):
            n = int(len(oot) * k); top = oot.sort_values("p", ascending=False).head(n)
            saved = (0.12 * top.y * RETURN_COST).sum()          # cancelled would-be returns avoid 1,150
            lost = (0.12 * top.value * margin).sum()            # every cancellation loses the sale margin
            holds.append(dict(margin=margin, top_share=k, net_per_700=(saved - lost) / len(oot) * 700))
    hd = pd.DataFrame(holds); hd.to_csv(f"{a.out}/evidence/hold_economics.csv", index=False); print(hd.round(0).to_string())

    # calibration
    oot["bin"] = pd.qcut(oot.p, 10, duplicates="drop")
    cal = oot.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    cal.to_csv(f"{a.out}/evidence/calibration.csv", index=False); print(cal.round(3).to_string())

    # where is it wrong?
    oot["flag"] = oot.p >= BREAKEVEN
    seg = []
    for col in ("shield", "pay", "family"):
        g = oot.groupby(col).apply(lambda s: pd.Series(dict(n=len(s), return_rate=s.y.mean(),
                                   auc=roc_auc_score(s.y, s.p) if s.y.nunique() > 1 else np.nan,
                                   missed_returns=int(((~s.flag) & (s.y == 1)).sum()),
                                   false_flags=int((s.flag & (s.y == 0)).sum()))), include_groups=False)
        g.index = [f"{col}={i}" for i in g.index]; seg.append(g)
    seg = pd.concat(seg); seg.to_csv(f"{a.out}/evidence/segments.csv"); print(seg.round(3).to_string())

    # ---- final fit on ALL labelled data
    final = mk().fit(X, y)
    sc, lr = final[0], final[-1]
    model = dict(features=F.FEATURES, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                 coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                 breakeven=BREAKEVEN, base_rate=float(y.mean()), trained_rows=len(tr),
                 trained_through=str(t.max()))
    json.dump(model, open(f"{a.out}/model.json", "w"), indent=1)

    # ---- predictions
    te_d = F.dedupe(te); assert te_d.order_id.is_unique and len(te_d) == len(te)
    Xt, dt = F.build(te_d.reset_index(drop=True), cu)
    pt = final.predict_proba(Xt)[:, 1]
    sub = pd.DataFrame({"order_id": te_d.order_id.values, "score": np.round(pt, 6)})
    ss = pd.read_csv(find(a.data, "sample_submission.csv"))
    sub = ss[["order_id"]].merge(sub, on="order_id", how="left"); assert sub.score.notna().all() and len(sub) == len(ss)
    sub.to_csv(f"{a.out}/predictions.csv", index=False)
    print("predictions", sub.shape, "mean score", sub.score.mean().round(4), "share>=breakeven", (sub.score >= BREAKEVEN).mean().round(3))

    summ = dict(auc_mean=float(cvdf.auc.mean()), auc_pooled=float(auc_pool), auc_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                pr_auc_mean=float(cvdf.pr_auc.mean()), acc_mean=float(cvdf.acc_at_0_5.mean()), acc_majority=float(cvdf.acc_majority.mean()),
                breakeven=BREAKEVEN, call_all_net_per_700=all_call["net_per_700"],
                test_share_above_breakeven=float((sub.score >= BREAKEVEN).mean()), test_mean_score=float(sub.score.mean()),
                dedup_removed=n_raw - len(tr), x100_fixed=int(d.value_fixed.sum()), n_train=len(tr))
    json.dump(summ, open(f"{a.out}/evidence/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
    coefs = pd.Series(lr.coef_[0], index=F.FEATURES).sort_values(); coefs.to_csv(f"{a.out}/evidence/coefficients.csv", header=["std_coef"])

if __name__ == "__main__":
    main()
