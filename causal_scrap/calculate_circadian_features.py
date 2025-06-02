#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circadian-regularity metrics (IV & CV) for selected participants
----------------------------------------------------------------
• computes CV & IV separately for morning / afternoon / evening / night
• merges with combined_participants_info.csv
• outputs circadian_metrics.csv
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0.  helpers -------------------------------------------------------
# ------------------------------------------------------------------
DAY_PARTS = ["morning", "afternoon", "evening", "night"]
ASLEEP_COL = {        # map day-part → rapids column name (asleep duration)
    "morning":   "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:morning",
    "afternoon": "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:afternoon",
    "evening":   "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:evening",
    "night":     "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:night",
}

def compute_iv_cv(df_part, min_days=3):
    """
    df_part : DataFrame indexed by date, columns = four day-parts (float minutes)
    returns 8-column Series (CV_*, IV_*)
    """
    if df_part.dropna(how='all').empty:
        return pd.Series({f"{m}_{p}": np.nan
                          for p in DAY_PARTS for m in ["CV", "IV"]})

    res = {}
    for part in DAY_PARTS:
        s = df_part[part].dropna()
        n = len(s)

        # CV
        res[f"CV_{part}"] = np.nan if n < 2 or s.mean() == 0 else s.std(ddof=0) / s.mean()

        # IV
        if n >= min_days and s.var(ddof=0) != 0:
            diffs = np.diff(s.values)
            res[f"IV_{part}"] = np.square(diffs).mean() / s.var(ddof=0)
        else:
            res[f"IV_{part}"] = np.nan
    return pd.Series(res)

# ------------------------------------------------------------------
# 1.  pick participants --------------------------------------------
# ------------------------------------------------------------------
info = pd.read_csv("combined_participants_info.csv")

sel_pids = set(info.loc[:, 'pid'].tolist())

# ------------------------------------------------------------------
# 2.  gather sleep rows --------------------------------------------
# ------------------------------------------------------------------
globem_root = "data/globem_raw"
years = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]

sleep_rows = []
for y in years:
    rap = os.path.join(globem_root, y, "FeatureData", "rapids.csv")
    if not os.path.isfile(rap):
        continue
    df = pd.read_csv(rap, usecols=['pid', 'date'] + list(ASLEEP_COL.values()))
    sleep_rows.append(df[df['pid'].isin(sel_pids)])

sleep_df = pd.concat(sleep_rows, ignore_index=True)
sleep_df['date'] = pd.to_datetime(sleep_df['date'])

# ------------------------------------------------------------------
# 3.  compute metrics per pid --------------------------------------
# ------------------------------------------------------------------
metrics = []
for pid, grp in sleep_df.groupby('pid'):
    wide = (
        grp.rename(columns={v:k for k, v in ASLEEP_COL.items()})
           .set_index('date')[DAY_PARTS]
           .astype(float)
           .sort_index()
    )
    met = compute_iv_cv(wide)
    met['pid'] = pid
    metrics.append(met)

metrics_df = pd.DataFrame(metrics)

# ------------------------------------------------------------------
# 4.  merge with participant info & ΔBDI2 ---------------------------
# ------------------------------------------------------------------
out = metrics_df.merge(info, on='pid', how='left', validate='1:1')

out['BDI_change'] = out['BDI2_POST'] - out['BDI2_PRE']

# keep only the requested columns first, then append “all the rest”
core_cols = (['pid', 'BDI2_PRE', 'BDI2_POST', 'BDI_change'] +
             [f"{m}_{p}" for p in DAY_PARTS for m in ['IV', 'CV']])
other_cols = [c for c in out.columns if c not in core_cols]
out_final = out[core_cols + other_cols]

# ------------------------------------------------------------------
# 5.  drop rows with missing BDI2 scores & save --------------------
# ------------------------------------------------------------------
out_final = out_final.dropna(subset=['BDI2_PRE', 'BDI2_POST'])

out_final.to_csv("circadian_metrics.csv", index=False)
print("Saved circadian_metrics.csv  →  shape:", out_final.shape)
