"""
GLOBEM Sleep-Mood Feature Extraction Pipeline  (auto-discover, clean names)
--------------------------------------------------------------------------
• Automatically finds all participants with ≥ 7 PHQ-4 entries
• Computes SRI, IS, σ_bed, mid-sleep, DLMO proxy per PHQ-window
• Aggregates **all** kept RAPIDS columns with window mean, std, and slope
• Clean feature names:  'f_screen:phone:foo'  ->  'f_screen_phone_foo_mean'
"""

from __future__ import annotations
import os, re, time, numpy as np, pandas as pd
from functools import lru_cache
from typing import List, Tuple

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
DATA_ROOT   = "data/globem_raw"
INS_FOLDERS = [f"INS-W_{i}" for i in range(2, 5)]                   # 2–4
LABEL_FILE  = "dep_weekly.csv"
FEATURE_FILE= "rapids.csv"

SLEEP_BED_COL  = "f_slp:fitbit_sleep_summary_rapids_firstbedtimemain:allday"
SLEEP_WAKE_COL = "f_slp:fitbit_sleep_summary_rapids_firstwaketimemain:allday"
MINUTES_PER_DAY = 1440

# -------------------------------------------------------------------------
# 0. Utilities
# -------------------------------------------------------------------------
def clean_name(col: str) -> str:
    """Return a Patsy-safe column name: drop leading 'f_' and colon→underscore."""
    col = re.sub(r"^f_", "", col)
    return col.replace(":", "_")

# -------------------------------------------------------------------------
# 1. Locate folders & cache heavy CSVs
# -------------------------------------------------------------------------
def find_pid_folder(pid: str) -> str:
    for folder in INS_FOLDERS:
        df = load_labels(folder)
        if pid in df["pid"].values:
            return folder
    raise FileNotFoundError(pid)

@lru_cache(maxsize=len(INS_FOLDERS))
def load_rapids(folder: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_ROOT, folder, "FeatureData", FEATURE_FILE),
                       low_memory=False)

@lru_cache(maxsize=len(INS_FOLDERS))
def load_labels(folder: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_ROOT, folder, "SurveyData", LABEL_FILE))

# -------------------------------------------------------------------------
# 2. Eligible subjects
# -------------------------------------------------------------------------
def find_subjects_with_phq(min_count: int = 7) -> List[str]:
    pids = []
    for folder in INS_FOLDERS:
        counts = load_labels(folder).groupby("pid")["phq4"].apply(lambda s: s.notna().sum())
        pids += counts[counts >= min_count].index.tolist()
    return pids

# -------------------------------------------------------------------------
# 3. Circadian helpers
# -------------------------------------------------------------------------
def linear_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.assign(**{c: df[c].interpolate("linear", limit_direction='both') for c in cols})

def day_vector(bed: float, wake: float) -> np.ndarray:
    vec = np.zeros(MINUTES_PER_DAY, dtype=int)
    b, w = int(round(bed*60)) % MINUTES_PER_DAY, int(round(wake*60)) % MINUTES_PER_DAY
    if w > b:
        vec[b:w] = 1
    else:
        vec[b:], vec[:w] = 1, 1
    return vec

def window_metrics(win: pd.DataFrame):
    win = win.dropna(subset=["bed_mod","wake_mod"])
    if win.empty: return (np.nan,)*5
    mat = np.stack([day_vector(b,w) for b,w in zip(win.bed_mod, win.wake_mod)])
    ser = mat.ravel()
    sri = 200*(ser[:-MINUTES_PER_DAY]==ser[MINUTES_PER_DAY:]).mean()-100
    grand = ser.mean(); is_val = ((mat.mean(0)-grand)**2).sum() / ((ser-grand)**2).sum()
    bed_sd = win.bed_mod.std()
    mids = (((win.wake_mod-win.bed_mod)%24)/2 + win.bed_mod)%24
    dlmo = (win.bed_mod.mean()-2)%24
    return sri, is_val, bed_sd, mids.mean(), dlmo

# -------------------------------------------------------------------------
# 4.  Build one subject dataframe
# -------------------------------------------------------------------------
def load_subject_tables(pid: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    folder = find_pid_folder(pid)
    feats  = load_rapids(folder)
    labels = load_labels(folder).loc[lambda d: d.pid==pid].drop(columns="Unnamed: 0", errors='ignore')

    keep = (feats.columns.str.contains(r"^(?:f_slp:fitbit|f_screen:phone|f_steps:fitbit|pid|date)")
            & ~feats.columns.str.contains(r"(?:14dhist|7dhist|_dis:|_norm:|weekday|weekend)"))
    feats = feats.loc[feats.pid==pid, feats.columns[keep]]
    return feats, labels

def build_subject_df(pid: str) -> pd.DataFrame:
    feats, labs = load_subject_tables(pid)
    for c in [SLEEP_BED_COL, SLEEP_WAKE_COL]:
        if c not in feats.columns: feats[c] = np.nan

    merged = (feats.merge(labs[["date","phq4"]], on="date", how="left")
                   .assign(date=lambda d: pd.to_datetime(d.date),
                           first_bed_hr = lambda d: d[SLEEP_BED_COL]/60,
                           first_wake_hr= lambda d: d[SLEEP_WAKE_COL]/60)
                   .sort_values("date"))

    num_cols = merged.select_dtypes("number").columns.difference(["pid","phq4"])
    merged[num_cols] = merged[num_cols].interpolate("linear", limit_direction='both')
    merged["bed_mod"]  = merged.first_bed_hr  % 24
    merged["wake_mod"] = merged.first_wake_hr % 24

    phq = merged[merged.phq4.notna()].sort_values("date")[["date","phq4"]]
    records=[]

    agg_cols = [c for c in num_cols if c not in (
        "first_bed_hr","first_wake_hr","bed_mod","wake_mod")]

    for i in range(1,len(phq)):
        win = merged[(merged.date>=phq.iloc[i-1,0]) & (merged.date<phq.iloc[i,0])]
        if len(win)<2: continue
        sri,is_val,sd,ms,dlmo = window_metrics(win)
        rec = dict(pid=pid,
                   PHQ4_date=phq.iloc[i].date.date(),
                   PHQ4=phq.iloc[i].phq4,
                   Days=len(win),
                   SRI=sri, IS=is_val, sigma_bed=sd,
                   mid_sleep=ms, dlmo_proxy=dlmo)
        # aggregate every kept sensor column
        for col in agg_cols:
            arr  = win[col].to_numpy()
            base = clean_name(col)
            rec[f"{base}_mean"]  = np.nanmean(arr)
            rec[f"{base}_std"]   = np.nanstd(arr, ddof=0)
            rec[f"{base}_slope"] = (np.nan if len(arr)<2
                                    else np.polyfit(np.arange(len(arr)), arr,1)[0])
        records.append(rec)
    return pd.DataFrame(records)

# -------------------------------------------------------------------------
# 5.  Pipeline runner
# -------------------------------------------------------------------------
def run_pipeline(min_phq:int=7) -> pd.DataFrame:
    pids = find_subjects_with_phq(min_phq)
    print(f"{len(pids)} participants with ≥{min_phq} PHQ-4 entries.")
    t0   = time.perf_counter()
    dfs  = []
    for pid in pids:
        start=time.perf_counter()
        dfs.append(build_subject_df(pid))
        print(f" {pid}: {time.perf_counter()-start:0.1f}s")
    print(f"Total: {time.perf_counter()-t0:0.1f}s")
    return pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------------------
if __name__ == "__main__":
    panel = run_pipeline(min_phq=7)
    panel.to_csv("data/processed/phq4_ts_features.csv", index=False)
    print("Columns:", panel.columns.tolist()[:15], "...")
    print(panel.head())
