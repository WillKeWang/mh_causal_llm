"""
GLOBEM Sleep‑Mood Feature Extraction Pipeline (auto‑discover version)
-------------------------------------------------------------------
• Added `find_subjects_with_phq()` – scans all INS‑W_x folders and returns
  every participant who has **≥ 8 PHQ‑4 entries** (non‑NaN) in `dep_weekly.csv`.
• Main section now builds the subject list automatically instead of hard‑coding
  pids.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple
import time

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
DATA_ROOT = "data/globem_raw"
INS_FOLDERS = [f"INS-W_{i}" for i in range(2, 5)]  # INS-W_2–INS-W_4

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
# 1. Locate folders & cache heavy csvs
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

@lru_cache(maxsize=len(INS_FOLDERS))
def load_rapids(folder: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_ROOT, folder, "FeatureData", FEATURE_FILE), low_memory=False)

@lru_cache(maxsize=len(INS_FOLDERS))
def load_labels(folder: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_ROOT, folder, "SurveyData", LABEL_FILE))

# -----------------------------------------------------------
# 2. Discover eligible subjects (≥ min_count PHQ‑4)
# -----------------------------------------------------------

def find_subjects_with_phq(min_count: int = 7) -> List[str]:
    pids = []
    for folder in INS_FOLDERS:
        df = load_labels(folder)
        counts = df.groupby("pid")["phq4"].apply(lambda s: s.notna().sum())
        pids.extend(counts[counts >= min_count].index.tolist())
    return pids

# -----------------------------------------------------------
# 3. Subject‑level feature extraction
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


def linear_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.assign(**{c: df[c].interpolate("linear", limit_direction="both") for c in cols})


def day_vector(bed_hr: float, wake_hr: float) -> np.ndarray:
    vec = np.zeros(MINUTES_PER_DAY, dtype=int)
    b = int(round(bed_hr * 60)) % MINUTES_PER_DAY
    w = int(round(wake_hr * 60)) % MINUTES_PER_DAY
    if w > b:
        vec[b:w] = 1
    else:
        vec[b:] = 1
        vec[:w] = 1
    return vec


# def window_metrics(win: pd.DataFrame) -> Tuple[float, float, float, float, float]:
#     mat = np.stack([day_vector(b, w) for b, w in zip(win["bed_mod"], win["wake_mod"])])
#     series = mat.ravel()
#     aligned = series[:-MINUTES_PER_DAY] == series[MINUTES_PER_DAY:]
#     sri = 200 * aligned.mean() - 100
#     grand = series.mean()
#     is_val = ((mat.mean(axis=0) - grand) ** 2).sum() / ((series - grand) ** 2).sum()
#     bed_sd = win["bed_mod"].std()
#     mids = (((win["wake_mod"] - win["bed_mod"]) % 24) / 2 + win["bed_mod"]) % 24
#     mid_sleep = float(mids.mean())
#     dlmo = float((win["bed_mod"].mean() - 2) % 24)
#     return sri, is_val, bed_sd, mid_sleep, dlmo

def window_metrics(win: pd.DataFrame):
    """Return circadian metrics for a PHQ‑window.
    If *all* bed/wake are NaN (after interpolation) return NaN for every metric.
    """
    win = win.dropna(subset=["bed_mod", "wake_mod"])
    if win.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    mat = np.stack([day_vector(b, w) for b, w in zip(win["bed_mod"], win["wake_mod"])])
    if mat.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    ser = mat.ravel()
    sri = 200 * (ser[:-MINUTES_PER_DAY] == ser[MINUTES_PER_DAY:]).mean() - 100
    grand = ser.mean()
    is_val = ((mat.mean(axis=0) - grand) ** 2).sum() / ((ser - grand) ** 2).sum() if grand not in (0,1) else np.nan
    bed_sd = win["bed_mod"].std()
    mids = (((win["wake_mod"] - win["bed_mod"]) % 24) / 2 + win["bed_mod"]) % 24
    dlmo = (win["bed_mod"].mean() - 2) % 24
    return sri, is_val, bed_sd, float(mids.mean()), float(dlmo)


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
# 4. Runtime utility
# -----------------------------------------------------------

def run_pipeline(min_phq: int = 7):
    pids = find_subjects_with_phq(min_phq)
    print(f"Eligible participants (≥{min_phq} PHQ-4s): {len(pids)}")
    t0 = time.perf_counter()
    out_frames = []
    for pid in pids:
        start = time.perf_counter()
        out_frames.append(build_subject_df(pid))
        print(f" {pid} … {time.perf_counter() - start:0.1f}s")
    print(f"Total runtime: {time.perf_counter() - t0:0.1f}s")
    return pd.concat(out_frames, ignore_index=True)

if __name__ == "__main__":
    panel = run_pipeline(min_phq=7)
    panel.to_csv("data/processed/phq4_ts_features.csv", index=False)
    print(panel.head())
