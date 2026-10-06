# 3-minute screen recording script (record this yourself; no slides)
**0:00-0:25 What I tried.** Show `train.csv`; first model with everything: AUC 0.998, 99% accuracy. "Too good." Show crosstab: REVERSE_PICKUP = 789/789 returned, pickup_scheduled_at on 94% of returns.
**0:25-1:00 What I changed.** Dropped the two post-return columns and `source`. De-duplicated 651 partner-feed rows. Fixed 700 October order values that were x100 (show ratio to list price = 100.0). Switched to out-of-time validation (Oct-Jun quarters). AUC falls to 0.775.
**1:00-1:25 What I threw away.** Gradient boosting (no better than logistic), the 95% accuracy target, the hold plan (show `hold_economics.csv` negative), pincode/signup date/delivery-note features.
**1:25-2:05 Evidence.** `evidence/charts.png`: calibration, call economics (break-even 11.2%), hold economics. Lift table: top 10% = 40% precision, 60% false flags.
**2:05-2:45 Service.** `python app.py`, open the screen, score the risky example (COD, 9-day promise, 2 prior returns), show reasons; score the Oct x100 value and show the data note; send a bad request and show the polite error. Stop the network: still works, no key.
**2:45-3:00 AI use.** Claude used for analysis/code; where it was wrong; cost of the product = Rs 0 per prediction.
