"""
GLOBEM Sleep-Mood Feature Extraction Pipeline
-------------------------------------------
Modular functions for
1. locating participant folders
2. loading RAPIDS feature tables & weekly depression labels with in-memory caching
3. imputing missing bed/wake & auxiliary wearable variables using linear interpolation
4. computing window-level circadian metrics (SRI, IS, σ_bed, mid-sleep, DLMO proxy)
5. aggregating auxiliary variables (mean & std) across the full series
6. assembling one tidy subject-level dataframe

Designed for ≈200 participants with 8-10 windows each.

Dependencies
------------
    pandas >= 1.3
    numpy  >= 1.21
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple, Dict
import time

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
DATA_ROOT = "data/globem_raw"  # base directory for INS‑W_* folders
INS_FOLDERS = [f"INS-W_{i}" for i in range(2, 5)]  # INS-W_1 … INS-W_4 folders

# screen / steps columns to aggregate
EXTRA_COLS = [
    "f_screen:phone_screen_rapids_sumdurationunlock_locmap_home:night",
    "f_screen:phone_screen_rapids_sumdurationunlock:night",
    "f_screen:phone_screen_rapids_sumdurationunlock:allday",
    "f_steps:fitbit_steps_intraday_rapids_countepisodeactivebout:allday",
    "f_steps:fitbit_steps_intraday_rapids_avgdurationactivebout:allday",
    "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday",
]

SLEEP_BED_COL = "f_slp:fitbit_sleep_summary_rapids_firstbedtimemain:allday"
SLEEP_WAKE_COL = "f_slp:fitbit_sleep_summary_rapids_firstwaketimemain:allday"
LABEL_FILE = "dep_weekly.csv"
FEATURE_FILE = "rapids.csv"

MINUTES_PER_DAY = 1440

# -----------------------------------------------------------
# 1. Utilities for locating and loading data
# -----------------------------------------------------------

def find_pid_folder(pid: str, data_root: str = DATA_ROOT) -> str:
    """
    Given a pid (e.g., 'INS-W_541'), determine which INS-W folder it belongs to.

    Args:
        pid (str): The participant ID to search for.
        data_root (str): Base directory where INS-W_1, INS-W_2, etc. are stored.

    Returns:
        str: The folder name (e.g., 'INS-W_2') if found.

    Raises:
        FileNotFoundError: If the PID is not found in any folder.
    """
    for folder in ["INS-W_2", "INS-W_3", "INS-W_4"]:
        label_path = os.path.join(data_root, folder, "SurveyData", "dep_weekly.csv")
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            if pid in df["pid"].values:
                return folder
    raise FileNotFoundError(f"PID {pid} not found in any INS-W folder.")

# Cache heavy RAPIDS csv per folder (≈ hundreds of MB)
@lru_cache(maxsize=len(INS_FOLDERS))
def load_rapids(folder: str, data_root: str = DATA_ROOT) -> pd.DataFrame:
    path = os.path.join(data_root, folder, "FeatureData", FEATURE_FILE)
    df = pd.read_csv(path, low_memory=False)
    return df

@lru_cache(maxsize=len(INS_FOLDERS))
def load_labels(folder: str, data_root: str = DATA_ROOT) -> pd.DataFrame:
    path = os.path.join(data_root, folder, "SurveyData", LABEL_FILE)
    return pd.read_csv(path)

# -----------------------------------------------------------
# 2. Subject‑level extraction & preprocessing
# -----------------------------------------------------------

def load_subject_tables(pid: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_df, labels_df) filtered for *pid*."""
    folder = find_pid_folder(pid)
    feats = load_rapids(folder)
    labels = load_labels(folder)

    labels = labels.loc[labels["pid"] == pid].drop(columns=["Unnamed: 0"], errors="ignore")

    keep_cols = (
        feats.columns.str.contains(r"^(?:f_slp:fitbit|f_screen:phone|f_steps:fitbit|pid|date)")
        & ~feats.columns.str.contains(r"(?:14dhist|7dhist|_dis:|_norm:|weekday|weekend)")
    )
    pid_feats = feats.loc[feats["pid"] == pid, feats.columns[keep_cols]]
    return pid_feats, labels

# -----------------------------------------------------------
# 3. Imputation helpers
# -----------------------------------------------------------

def linear_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].interpolate("linear", limit_direction="both")
    return out

# -----------------------------------------------------------
# 4. Circadian metrics per PHQ‑4 window
# -----------------------------------------------------------

def day_vector(bed_hr: float, wake_hr: float) -> np.ndarray:
    vec = np.zeros(MINUTES_PER_DAY, dtype=int)
    b = int(round(bed_hr * 60)) % MINUTES_PER_DAY
    w = int(round(wake_hr * 60)) % MINUTES_PER_DAY
    vec[b:w] = 1 if w > b else 1
    if w <= b:  # crosses midnight
        vec[:w] = 1
    return vec

def window_metrics(win: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    mat = np.stack([day_vector(b, w) for b, w in zip(win["bed_mod"], win["wake_mod"])])
    series = mat.ravel()
    aligned = series[:-MINUTES_PER_DAY] == series[MINUTES_PER_DAY:]
    sri = 200 * aligned.mean() - 100
    grand = series.mean()
    is_val = ((mat.mean(axis=0) - grand) ** 2).sum() / ((series - grand) ** 2).sum()
    bed_sd = win["bed_mod"].std()
    mids = (((win["wake_mod"] - win["bed_mod"]) % 24) / 2 + win["bed_mod"]) % 24
    mid_sleep = float(mids.mean())
    dlmo = float((win["bed_mod"].mean() - 2) % 24)
    return sri, is_val, bed_sd, mid_sleep, dlmo

# -----------------------------------------------------------
# 5. Pipeline: build tidy dataframe for one subject
# -----------------------------------------------------------

def build_subject_df(pid: str) -> pd.DataFrame:
    feats, labels = load_subject_tables(pid)

    # add extra columns if missing (to keep downstream code simple)
    for col in [SLEEP_BED_COL, SLEEP_WAKE_COL] + EXTRA_COLS:
        if col not in feats.columns:
            feats[col] = np.nan

    # merge & convert minutes→hours
    merged = pd.merge(
        feats[["pid", "date", *[SLEEP_BED_COL, SLEEP_WAKE_COL], *EXTRA_COLS]],
        labels[["date", "phq4"]], on="date", how="left")

    merged["date"] = pd.to_datetime(merged["date"])
    merged["first_bed_hr"]  = merged[SLEEP_BED_COL]  / 60.0
    merged["first_wake_hr"] = merged[SLEEP_WAKE_COL] / 60.0

    # impute bed/wake & extras
    imp_cols = ["first_bed_hr", "first_wake_hr", *EXTRA_COLS]
    merged = merged.sort_values("date")
    merged = linear_impute(merged, imp_cols)

    # pre‑compute mod‑24 sleep times
    merged["bed_mod"]  = merged["first_bed_hr"]  % 24
    merged["wake_mod"] = merged["first_wake_hr"] % 24

    # per‑window circadian metrics
    phq = merged[~merged["phq4"].isna()].sort_values("date")[["date", "phq4"]]
    records = []
    for i in range(1, len(phq)):
        win = merged[(merged["date"] >= phq.iloc[i-1]["date"]) & (merged["date"] < phq.iloc[i]["date"])]
        if len(win) < 2:
            continue
        sri, is_val, sd, ms, dlmo = window_metrics(win)
        # aggregate extra features across *imputed* window
        agg = {}
        for col in EXTRA_COLS:
            col_mean = win[col].mean()
            col_std  = win[col].std()
            agg[f"{col}_mean"] = col_mean if not np.isnan(col_mean) else np.nan
            agg[f"{col}_std"]  = col_std  if not np.isnan(col_std)  else np.nan
        records.append({
            "pid": pid,
            "PHQ4_date": phq.iloc[i]["date"].date(),
            "PHQ4": phq.iloc[i]["phq4"],
            "Days": len(win),
            "SRI": sri,
            "IS": is_val,
            "sigma_bed": sd,
            "mid_sleep": ms,
            "dlmo_proxy": dlmo,
            **agg
        })
    return pd.DataFrame(records)

# -----------------------------------------------------------
# 6. Simple runtime unit test on a handful of IDs
# -----------------------------------------------------------

def demo_runtime(pids: List[str]) -> pd.DataFrame:
    t0 = time.perf_counter()
    dfs = []
    for pid in pids:
        start = time.perf_counter()
        df = build_subject_df(pid)
        dfs.append(df)
        print(f"{pid}: processed in {time.perf_counter() - start:0.2f} s")
    print(f"Total elapsed: {time.perf_counter() - t0:0.2f} s")
    return pd.concat(dfs, ignore_index=True)

# Run a quick check when executed directly
if __name__ == "__main__":
    SAMPLE_PIDS = ["INS-W_335", 
             "INS-W_336",
             "INS-W_344",
             "INS-W_360",
             "INS-W_427",
             "INS-W_504",
             "INS-W_508",
             "INS-W_518",
             "INS-W_537",
             "INS-W_541",
             "INS-W_698",
             "INS-W_701",
             "INS-W_744",
             "INS-W_746",
             "INS-W_751",
             "INS-W_913",
             "INS-W_974",
             "INS-W_991",
             "INS-W_1000",
             "INS-W_1031",
             "INS-W_1038",
             "INS-W_1081",
             ]
    result = demo_runtime(SAMPLE_PIDS)
    result.to_csv("data/processed/demo_output.csv", index=False)
