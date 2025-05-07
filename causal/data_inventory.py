# step1_inventory.py ----------------------------------------------------------
import pandas as pd
import pathlib
import re

ROOT = pathlib.Path("/Users/kwang/Data/globem")   # change if needed

# --------------------------------------------------------------------- helpers
def detect_type(series: pd.Series) -> str:
    """Heuristically label a column."""
    if series.name == "dep":
        return "survey_outcome"
    if series.dropna().astype(str).isin(["l", "m", "h"]).all():
        return "categorical_lmh"
    try:
        pd.to_numeric(series.dropna().sample(min(500, len(series))), errors="raise")
        return "numeric"
    except Exception:
        return "categorical_lmh"   # fallback

def load_all_csvs():
    csv_files = ROOT.rglob("*.csv")
    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["__source__"] = str(f.relative_to(ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)

# ---------------------------------------------------------------------- load
print("📥 Loading every CSV (this may take a minute)…")
full = load_all_csvs()

# ----------------------------------------------------------------- feature info
feature_info = (
    full.drop(columns="__source__")
        .apply(detect_type)
        .to_frame(name="type")
)
feature_info["missing_pct"] = full.isna().mean().round(4) * 100

feature_info.to_csv("feature_info.csv")
print("✅ feature_info.csv written")

# -------------------------------------------------------------- missing by pid
cols_to_check = feature_info.index.tolist()
missing_by_pid = (
    full.groupby("pid")[cols_to_check]
        .apply(lambda df: df.isna().mean(axis=None))
        .rename("missing_pct_features")
        .mul(100)
        .reset_index()
)
missing_by_pid.to_csv("missing_by_pid.csv", index=False)
print("✅ missing_by_pid.csv written")

# ------------------------------------------------------------- optional heatmap
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.imshow(full.sample(min(1000, len(full))).isna(), aspect='auto', interpolation='nearest')
    plt.xlabel("Columns"); plt.ylabel("Random sample rows")
    plt.title("Missingness heat‑map (white = missing)")
    plt.tight_layout(); plt.savefig("missingness_heatmap.png", dpi=300)
    print("🖼️  missingness_heatmap.png saved")
except ImportError:
    print("matplotlib not installed – skipping heat‑map")