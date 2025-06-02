#!/usr/bin/env python3
"""
Streaming inventory for the GLOBEM dataset
------------------------------------------
Outputs
--------
feature_info.csv
    column | type | missing_pct
missing_by_pid.csv
    pid | missing_pct_cells | feature_pct_available | n_days
"""

from __future__ import annotations
import argparse, pathlib, warnings
from collections import Counter, defaultdict

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ───────── helper functions ─────────
def detect_type_partial(series: pd.Series, prev: str | None) -> str:
    """Incrementally infer column type."""
    if series.name == "pid":
        return "identifier"
    if prev == "categorical_lmh":
        return prev
    samp = series.dropna().astype(str)
    if samp.empty:
        return prev or "unknown"
    if samp.isin(["l", "m", "h"]).all():
        return "categorical_lmh"
    numeric = pd.to_numeric(samp, errors="coerce")
    if numeric.notna().all():
        return "numeric"
    return "categorical_lmh"


def update_pid_stats(chunk: pd.DataFrame,
                     pid_missing: defaultdict[str, Counter],
                     pid_feat_seen: defaultdict[str, set],
                     pid_dates: defaultdict[str, set],
                     feature_cols: list[str],
                     date_cols: list[str]) -> None:
    """Update all three per‑participant tallies."""
    # 1) cell‑level missingness
    mask = chunk[feature_cols].isna()
    for pid, grp in mask.groupby(chunk["pid"]):
        total = grp.size
        missing = grp.values.sum()
        pid_missing[pid]["seen"]  += total - missing
        pid_missing[pid]["total"] += total

    # 2) feature availability (has at least one non‑missing value)
    for col in feature_cols:
        pids_with_val = chunk.loc[chunk[col].notna(), "pid"].unique()
        for pid in pids_with_val:
            pid_feat_seen[pid].add(col)

    # 3) unique days
    if date_cols:
        dcol = date_cols[0]                     # use first matching column
        dates = chunk[[ "pid", dcol ]].dropna()
        dates[dcol] = dates[dcol].astype(str).str[:10]  # YYYY‑MM‑DD slice
        for pid, grp in dates.groupby("pid"):
            pid_dates[pid].update(grp[dcol].unique())


# ───────── main routine ─────────
def main(root: pathlib.Path) -> None:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV files under {root}")

    print(f"Scanning {len(csvs)} CSVs under {root} …")

    col_missing, col_total = Counter(), Counter()
    col_type: dict[str, str] = {}

    pid_missing  : defaultdict[str, Counter] = defaultdict(Counter)
    pid_feat_seen: defaultdict[str, set]     = defaultdict(set)
    pid_dates    : defaultdict[str, set]     = defaultdict(set)

    for p in csvs:
        print("  •", p.relative_to(root))
        for chunk in pd.read_csv(
            p,
            chunksize=50_000,
            dtype=str,
            keep_default_na=True,
            na_values=["", "na", "NA"],
        ):
            # drop leading blank/Unnamed column if any
            first = chunk.columns[0]
            if first.strip() == "" or first.startswith("Unnamed"):
                chunk.drop(columns=[first], inplace=True)

            if "pid" not in chunk.columns:
                raise ValueError(f"'pid' column missing in {p}")

            feature_cols = chunk.columns.difference(["pid"])
            date_cols = [c for c in chunk.columns
                         if c.lower() in ("date", "timestamp", "datetime",
                                          "datestamp", "start_time", "utc_timestamp")]

            # per‑column stats
            nulls = chunk.isna()
            for col in chunk.columns:
                col_missing[col] += nulls[col].sum()
                col_total[col]   += len(chunk)
                col_type[col]     = detect_type_partial(chunk[col], col_type.get(col))

            # per‑participant stats
            update_pid_stats(chunk, pid_missing, pid_feat_seen,
                             pid_dates, list(feature_cols), date_cols)

    # ── write feature_info.csv ──
    pd.DataFrame({
        "column": sorted(col_total),
        "type":   [col_type[c] for c in sorted(col_total)],
        "missing_pct": [
            round(100 * col_missing[c] / max(col_total[c], 1), 4)
            for c in sorted(col_total)
        ],
    }).to_csv("feature_info.csv", index=False)
    print("✅  feature_info.csv written")

    # ── write missing_by_pid.csv ──
    total_features = len([c for c in col_total if c != "pid"])
    rows = []
    for pid in pid_missing:
        seen = pid_missing[pid]["seen"]
        total = pid_missing[pid]["total"]
        rows.append({
            "pid": pid,
            "missing_pct_cells": round(100 * (total - seen) / total, 2),
            "feature_pct_available": round(
                100 * len(pid_feat_seen[pid]) / total_features, 2
            ),
            "n_days": len(pid_dates[pid]),
        })
    pd.DataFrame(rows).to_csv("missing_by_pid.csv", index=False)
    print("✅  missing_by_pid.csv written")


# ───────── entry point ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/globem_raw",
                    help="Path to GLOBEM root (default: data/globem_raw)")
    args = ap.parse_args()
    main(pathlib.Path(args.root).expanduser())
