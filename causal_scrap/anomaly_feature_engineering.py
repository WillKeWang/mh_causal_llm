#!/usr/bin/env python3
"""
End-to-end anomaly-count feature builder  (HBOS or COPOD)
---------------------------------------------------------
▪ finds all pids with ≥ 7 PHQ-4 entries
▪ daily anomaly detection per feature class  (screen / steps / sleep)
▪ window-level counts & proportions per PHQ interval
▪ saves  data/processed/phq4_anomaly_counts_<MODEL>.csv
"""

# ------------------------------------------------------------------ #
# 0.  Imports & global config
# ------------------------------------------------------------------ #
import os, re, sys, time
import numpy as np, pandas as pd
from functools          import lru_cache
from sklearn.preprocessing import RobustScaler
from pyod.models.hbos   import HBOS
from pyod.models.copod  import COPOD

# ---------- paths -------------------------------------------------- #
DATA_ROOT    = "data/globem_raw"
OUT_DAILY    = "data/processed/anomaly_daily"         # intermediate
OUT_PANEL    = "data/processed"                       # final CSV location
os.makedirs(OUT_DAILY, exist_ok=True)
os.makedirs(OUT_PANEL, exist_ok=True)

INS_FOLDERS  = [f"INS-W_{i}" for i in range(2, 5)]    # 2, 3, 4
LABEL_FILE   = "dep_weekly.csv"
FEATURE_FILE = "rapids.csv"

# ---------- column patterns --------------------------------------- #
INCLUDE_RE   = r"^(?:f_slp:fitbit|f_steps:fitbit|f_screen:phone)"
EXCLUDE_SUBS = ["_norm", "_dis", "_7dhist", "_14dhist", "weekday", "weekend"]
EXCLUDE_RE   = "|".join(map(re.escape, EXCLUDE_SUBS))

BLOCK_REGEX  = {
    "screen": r"^f_screen:phone",
    "steps" : r"^f_steps:fitbit",
    "sleep" : r"^f_slp:fitbit",
}

# pick detector from command-line arg 0|1
MODEL = sys.argv[1].upper() if len(sys.argv) > 1 else "HBOS"
assert MODEL in ("HBOS", "COPOD"), "choose HBOS or COPOD"

# ------------------------------------------------------------------ #
# 1.  Cached file loaders
# ------------------------------------------------------------------ #
@lru_cache(maxsize=len(INS_FOLDERS))
def load_rapids(folder: str) -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, folder, "FeatureData", FEATURE_FILE)
    return pd.read_csv(path, low_memory=False)

@lru_cache(maxsize=len(INS_FOLDERS))
def load_labels(folder: str) -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, folder, "SurveyData", LABEL_FILE)
    return pd.read_csv(path, low_memory=False \
           ).assign(date=lambda d: pd.to_datetime(d.date))

# ------------------------------------------------------------------ #
# 2.  PID helpers
# ------------------------------------------------------------------ #
def find_subjects_with_phq(min_count: int = 7):
    pids = []
    for folder in INS_FOLDERS:
        cnt = (load_labels(folder)
               .groupby("pid")["phq4"]
               .apply(lambda s: s.notna().sum()))
        pids += cnt[cnt >= min_count].index.tolist()
    return pids

def find_pid_folder(pid: str):
    for f in INS_FOLDERS:
        if pid in load_labels(f)["pid"].values:
            return f
    raise FileNotFoundError(pid)

# ------------------------------------------------------------------ #
# 3.  Daily feature matrix  (NaN-safe)
# ------------------------------------------------------------------ #
def get_daily_df(pid: str) -> pd.DataFrame:
    folder = find_pid_folder(pid)
    df     = load_rapids(folder).loc[lambda d: d.pid == pid].copy()

    good = df.columns.str.contains(INCLUDE_RE, regex=True)
    bad  = df.columns.str.contains(EXCLUDE_RE, regex=True)
    cols = df.columns[good & ~bad]
    if cols.empty:
        raise ValueError("no usable cols")

    df = df[["date"] + cols.tolist()]
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    num = df.select_dtypes("number")
    num = num.loc[:, ~num.isna().all()]                                # drop all-NaN columns
    num = num.interpolate("linear", limit_direction="both")
    num = num.fillna(num.median())

    return num                                                         # rows = days

# ------------------------------------------------------------------ #
# 4.  Detector helper
# ------------------------------------------------------------------ #
def fit_scores(df_block: pd.DataFrame):
    if df_block.shape[1] == 0:
        n = len(df_block)
        return np.full(n, np.nan), np.zeros(n, dtype=int)

    X   = RobustScaler().fit_transform(df_block)
    det = HBOS() if MODEL == "HBOS" else COPOD()
    det.fit(X)
    return det.decision_scores_, det.labels_

# ------------------------------------------------------------------ #
# 5.  Summarise one subject
# ------------------------------------------------------------------ #
def summarise_subject(pid: str):
    df_daily = get_daily_df(pid)
    dates    = df_daily.index

    # -------- daily anomaly detection ------------------------------
    result_daily = pd.DataFrame({"date": dates})
    for cls, rgx in BLOCK_REGEX.items():
        scores, flags = fit_scores(df_daily.filter(regex=rgx))
        result_daily[f"{cls}_score"] = scores
        result_daily[f"{cls}_flag"]  = flags
    result_daily.set_index("date", inplace=True)
    # save daily file for reference
    result_daily.reset_index().to_csv(
        f"{OUT_DAILY}/{pid}_{MODEL}_multiblock.csv", index=False)

    # -------- roll into PHQ windows --------------------------------
    folder = find_pid_folder(pid)
    phq = (load_labels(folder)
           .loc[lambda d: d.pid == pid, ["date", "phq4"]]
           .sort_values("date")
           .reset_index(drop=True))

    rows = []
    for i in range(1, len(phq)):
        win = result_daily.loc[phq.loc[i-1, "date"]
                               : phq.loc[i, "date"] - pd.Timedelta("1D")]
        rec = dict(pid = pid,
                   PHQ4_date = phq.loc[i, "date"].date(),
                   PHQ4      = phq.loc[i, "phq4"],
                   window_days = len(win))
        for cls in BLOCK_REGEX:
            flag = win[f"{cls}_flag"]
            rec[f"{cls}_count_flag"] = int(flag.sum())
            rec[f"{cls}_prop_flag"]  = flag.mean()
        rows.append(rec)

    return pd.DataFrame(rows)

# ------------------------------------------------------------------ #
# 6.  Pipeline runner
# ------------------------------------------------------------------ #
def main():
    pids = find_subjects_with_phq()
    print(f"{len(pids)} eligible participants · detector = {MODEL}")
    t0, dfs = time.perf_counter(), []

    for k, pid in enumerate(pids, 1):
        try:
            dfs.append(summarise_subject(pid))
            print(f"{k:3d}/{len(pids)}  {pid} ✓")
        except Exception as e:
            print(f"{k:3d}/{len(pids)}  {pid} ✗  ({e})")

    panel = pd.concat(dfs, ignore_index=True)
    out_path = f"{OUT_PANEL}/phq4_anomaly_counts_{MODEL}.csv"
    panel.to_csv(out_path, index=False)

    print(f"\nSaved {len(panel)} rows  →  {out_path}")
    print("Elapsed:", f"{time.perf_counter()-t0:0.1f}s")
    print("\nPreview:")
    print(panel.head())

if __name__ == "__main__":
    main()
