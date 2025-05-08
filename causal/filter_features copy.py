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

# ───────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────
GLOBEM_ROOT = pathlib.Path("/Users/kwang/Data/globem")      # edit if needed
OUT_CSV     = pathlib.Path("data/processed/globem_causal_input.csv")

OUTCOME    = "dep_bin"   # "dep_bin", "BDI2", or "dep"
BDI_THRESH = 14          # BDI‑II cutoff for mild depression → binary flag

SURVEY_COVARS = [
    "BFI10_neuroticism_PRE", "BFI10_extroversion_PRE",
    "BFI10_conscientiousness_PRE", "BRS_PRE", "PSS_10items_PRE",
]

# ───────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────
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

def write_stats(df, path):
    stats = pd.DataFrame({
        "feature": df.columns,
        "mean": df.mean(skipna=True),
        "var":  df.var(skipna=True),
        "non_nan_ratio": df.notna().mean(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)
    print(f"📑  Stats → {path} ({stats.shape[0]} feats)")

# ───────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────
def main():
    # discover dataset folders
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

    # 1) aggregate features
    pat = re.compile(r"_norm\b")
    X = aggregate_subject(feature_csvs, pattern=lambda c: pat.search(c))
    print("▶ After aggregation:", X.shape)

    # 2) covariates + labels
    pre_df = load_first_by_pid(pre_paths, "pre.csv")[SURVEY_COVARS]
    dep_df = load_dep(dep_paths)

    X = X.join(pre_df, how="left").join(dep_df, how="left")
    X = X.loc[~X[OUTCOME].isna()]
    print("▶ After merge:", X.shape)

    # 3) variance filter
    var_thr, min_obs = 5e-2, 0.05
    feats = X.columns.difference([OUTCOME])
    ok = (X[feats].notna().mean() >= min_obs) & (X[feats].var(skipna=True) > var_thr)
    X = X.loc[:, ok[ok].index.tolist() + [OUTCOME]]
    print("▶ After variance filter:", X.shape)
    X_final.to_csv("data/processed/var_filtered_features.csv", index_label="pid")
    print("▶ Filtered dataframe saved. ")

    # 4) correlation clustering (|ρ|>0.8) → medoids
    corr = X.drop(columns=[OUTCOME]).corr().abs()
    dist = 1 - corr
    clust = AgglomerativeClustering(metric="precomputed",
                                    linkage="average",
                                    distance_threshold=0.2,
                                    n_clusters=None).fit(dist)
    clusters = defaultdict(list)
    for col, lab in zip(corr.columns, clust.labels_):
        clusters[lab].append(col)
    medoids = [
        cols[0] if len(cols) == 1 else corr.loc[cols, cols].mean(1).idxmax()
        for cols in clusters.values()
    ]
    X_red = X[medoids + [OUTCOME]]
    print("▶ After corr‑cluster:", X_red.shape)

    # 5) XGBoost importance + greedy decorrelation ≤25
    y = X_red.pop(OUTCOME).values
    bst = xgb.XGBClassifier(n_estimators=400, max_depth=4,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=0).fit(X_red, y)
    imp_rank = pd.Series(bst.feature_importances_, index=X_red.columns)\
                 .sort_values(ascending=False).index
    corr2 = X_red.corr().abs()
    keep = []
    for f in imp_rank:
        if all(corr2.loc[f, k] < 0.5 for k in keep):
            keep.append(f)
        if len(keep) == 25:
            break
    X_final = X_red[keep]
    X_final[OUTCOME] = y
    print("▶ After XGB importance:", X_final.shape)

    # 6) VIF pruning (keep VIF ≤5)
    scaler = StandardScaler().fit_transform(X_final[keep].fillna(0))
    vif = [variance_inflation_factor(scaler, i) for i in range(len(keep))]
    while max(vif) > 5 and len(keep) > 20:
        keep.pop(int(np.argmax(vif)))
        scaler = StandardScaler().fit_transform(X_final[keep].fillna(0))
        vif = [variance_inflation_factor(scaler, i) for i in range(len(keep))]
    X_final = X_final[keep + [OUTCOME]]
    print("▶ Final matrix:", X_final.shape, "| max VIF:", round(max(vif), 2))

    # 7) save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    X_final.to_csv(OUT_CSV, index_label="pid")
    print("✅ Saved →", OUT_CSV)

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
