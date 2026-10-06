import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cal=pd.read_csv("evidence/calibration.csv"); ec=pd.read_csv("evidence/call_economics.csv"); hd=pd.read_csv("evidence/hold_economics.csv")
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].plot([0,.45],[0,.45],"--",c="grey"); ax[0].plot(cal.mean_pred,cal.actual,"o-"); ax[0].set(title="Calibration (out-of-time, 10 bins)",xlabel="predicted return rate",ylabel="actual return rate")
ax[1].plot(ec.thr*100,ec.net_per_700,"o-"); ax[1].axvline(11.18,ls=":",c="r"); ax[1].set(title="Confirmation call: net Rs per 700 orders",xlabel="call if risk >= x %",ylabel="Rs / month"); ax[1].text(12,ax[1].get_ylim()[0]+500,"break-even 11.2%",color="r")
for m,g in hd.groupby("margin"): ax[2].plot(g.top_share*100,g.net_per_700,"o-",label=f"margin {int(m*100)}%")
ax[2].axhline(0,c="k",lw=.5); ax[2].set(title="HOLD top x% : net Rs per 700 orders",xlabel="% of orders held",ylabel="Rs / month"); ax[2].legend()
plt.tight_layout(); plt.savefig("evidence/charts.png",dpi=110)
