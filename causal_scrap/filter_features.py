#!/usr/bin/env python3
"""
build_feature_set.py – GLOBEM causal‑discovery matrix
"""

import pathlib, re, warnings, textwrap
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ── CONFIG ─────────────────────────────────────────────────────────────
GLOBEM_ROOT = pathlib.Path("/Users/kwang/Data/globem")        # edit if needed
OUT_CSV     = pathlib.Path("data/processed/globem_causal_input.csv")

OUTCOME    = "dep_bin"   # "dep_bin", "BDI2", or "dep"
BDI_THRESH = 14          # BDI‑II cutoff for mild depression → binary flag

SURVEY_COVARS = [
    "BFI10_neuroticism_PRE", "BFI10_extroversion_PRE",
    "BFI10_conscientiousness_PRE", "BRS_PRE", "PSS_10items_PRE",
]

# ── HELPERS (unchanged except variance filter) ────────────────────────
def slope(arr):
    idx, m = np.arange(len(arr), dtype=float), ~np.isnan(arr)
    return np.nan if m.sum() < 3 else float(np.polyfit(idx[m], arr[m], 1)[0])

def aggregate_subject(csv_paths, pattern):
    agg = defaultdict(lambda: defaultdict(list))
    for idx_path, csv in enumerate(csv_paths, 1):
        for chunk in pd.read_csv(csv, chunksize=40_000, dtype=str,
                                 na_values=["", "na", "NA"]):
            if "pid" not in chunk:
                continue
            first = chunk.columns[0]
            if first.strip() == "" or first.startswith("Unnamed"):
                chunk = chunk.drop(columns=[first])

            cols = [c for c in chunk.columns if pattern(c)]
            if not cols:
                continue

            feats = chunk[cols].apply(pd.to_numeric, errors="coerce")
            feats.insert(0, "pid", chunk["pid"].astype(str))

            if idx_path == 1:
                print("\n▶ First matched CSV:", csv)
                print("  Columns kept:", len(cols))

            for pid, grp in feats.groupby("pid"):
                for col in grp.columns[1:]:
                    v = grp[col].values
                    if np.isnan(v).all():
                        continue
                    agg[pid][f"{col}__mean"].append(np.nanmean(v))
                    agg[pid][f"{col}__std" ].append(np.nanstd (v))
    return pd.DataFrame.from_dict(
        {p: {k: np.nanmean(v) for k, v in d.items()} for p, d in agg.items()},
        orient="index")

def load_first_by_pid(paths, name):
    if not paths:
        return pd.DataFrame()
    df = pd.concat([
        pd.read_csv(p, dtype=str, na_values=["", "na", "NA"]).iloc[:, 1:]
        for p in paths
    ])
    df["pid"] = df["pid"].astype(str)
    num = df.columns.difference(["pid", "date"])
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    joined = df.groupby("pid").first()
    print(f"  {name} rows merged:", joined.shape[0])
    return joined

def load_dep(paths):
    if not paths:
        return pd.DataFrame()
    df = pd.concat([
        pd.read_csv(p, dtype=str, na_values=["", "na", "NA"]).iloc[:, 1:]
        for p in paths
    ])
    df["pid"] = df["pid"].astype(str)
    df["BDI2"] = pd.to_numeric(df["BDI2"], errors="coerce")
    df["dep"]  = df["dep"].map({"True": 1, "False": 0})
    dep = df.groupby("pid").agg(BDI2=("BDI2", "mean"),
                                dep =("dep", "max"))
    dep["dep_bin"] = (dep["BDI2"] >= BDI_THRESH).astype(float)
    print("  dep rows merged:", dep.shape[0])
    return dep[[OUTCOME]]

# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    # 0) gather paths ---------------------------------------------------
    ds_dirs = [d for d in GLOBEM_ROOT.iterdir()
               if d.is_dir() and d.name.startswith("INS-W_")]
    feature_csvs, pre_paths, dep_paths = [], [], []
    for d in ds_dirs:
        feature_csvs += list((d / "FeatureData").glob("*.csv"))
        if (pre := d / "SurveyData" / "pre.csv").exists():
            pre_paths.append(pre)
        dep = d / "SurveyData" / "dep_weekly.csv"
        if not dep.exists():
            dep = d / "SurveyData" / "dep_endterm.csv"
        if dep.exists():
            dep_paths.append(dep)

    print(textwrap.dedent(f"""
        ▶ Totals
          Feature CSVs : {len(feature_csvs)}
          pre.csv       : {len(pre_paths)}
          dep*.csv      : {len(dep_paths)}
    """).strip())

    # 1) aggregate features -------------------------------------------
    pat = re.compile(r"_norm\b")
    X = aggregate_subject(feature_csvs, pattern=lambda c: pat.search(c))
    print("▶ After aggregation:", X.shape)

    # 2) add covariates + labels --------------------------------------
    pre_df = load_first_by_pid(pre_paths, "pre.csv")[SURVEY_COVARS]
    dep_df = load_dep(dep_paths)
    X = X.join(pre_df, how="left").join(dep_df, how="left")
    X = X.loc[~X[OUTCOME].isna()]
    print("▶ After merge:", X.shape)

    # 3) ~~~ improved variance/mean ratio filter ~~~~~~~~~~~~~~~~~~~~~~
    min_obs = 0.05        # ≥5 % non‑missing
    var_thr = 1e-2        # absolute variance threshold
    ratio_thr = 0.15      # var / |mean| must exceed 0.15

    feats = X.columns.difference([OUTCOME])
    var   = X[feats].var(skipna=True)
    mean  = X[feats].mean(skipna=True).abs().replace(0, np.nan)
    ratio = var / mean

    ok = (
        (X[feats].notna().mean() >= min_obs) &
        (var > var_thr) &
        (ratio > ratio_thr)
    )
    kept_feats = ok[ok].index.tolist()
    X = X[kept_feats + [OUTCOME]]
    print(f"▶ After ratio filter (>{ratio_thr}):", X.shape)
    X.to_csv("data/processed/var_filtered_features.csv", index_label="pid")
    print("✅ Filtered Feature Set Saved → ", "data/processed/var_filtered_features.csv")

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
