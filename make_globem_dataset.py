#!/usr/bin/env python3
"""
make_globem_dataset.py
======================

Generate instruction-tuning JSON files for the GLOBEM dataset
(Health-LLM / CHIL-24).

Only the “PHQ-4 *depression*” subtask is implemented for now.

Example
-------
python make_globem_dataset.py \
       --root   /groups/xx2489_gp/kw3215/Datasets/globem \
       --subtask depression \
       --out_dir datasets
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

from tqdm import tqdm


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def maybe_float(x: str):
    try:
        return float(x) if x.strip() != "" else None
    except ValueError:
        return None


def read_csv(path: Path) -> List[List[str]]:
    with path.open(newline="") as f:
        return list(csv.reader(f))


# --------------------------------------------------------------------------
# main processing
# --------------------------------------------------------------------------
def build_globem_examples(root: Path, subtask: str) -> List[Dict[str, Any]]:
    if subtask != "depression":
        print(f"[WARN] subtask '{subtask}' not implemented yet.", file=sys.stderr)
        return []

    final: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # loop over the four academic-year cohorts
    # ------------------------------------------------------------------
    for insw_idx in [1, 2, 3, 4]:
        cohort_dir = root / f"INS-W_{insw_idx}"
        if not cohort_dir.exists():
            print(f"[SKIP] folder {cohort_dir} missing")
            continue

        # -------------------- 1) labels -------------------------------
        lbl_path = cohort_dir / "SurveyData" / "dep_endterm.csv"
        if not lbl_path.exists():
            print(f"[SKIP] {lbl_path} missing")
            continue

        lbl_rows = read_csv(lbl_path)
        lbl_keys = lbl_rows[0]  # ['', pid, date, BDI2, dep]
        # create parallel lists
        labels: Dict[str, List[str]] = {k: [] for k in lbl_keys[1:]}  # drop empty first col
        for row in lbl_rows[1:]:
            for k, cell in zip(lbl_keys[1:], row[1:]):
                labels[k].append(cell)

        # -------------------- 2) RAPIDS features ----------------------
        rapids_path = cohort_dir / "FeatureData" / "rapids.csv"
        if not rapids_path.exists():
            print(f"[SKIP] {rapids_path} missing")
            continue

        rap_rows = read_csv(rapids_path)
        rap_keys = rap_rows[0]  # first row is header
        rap: Dict[str, List[str]] = {k: [] for k in rap_keys}
        for row in rap_rows[1:]:
            for k, cell in zip(rap_keys, row):
                rap[k].append(cell)

        # column names we need (14-day hist averages etc.)
        COL_STEP_AVG = (
            "f_steps:fitbit_steps_summary_rapids_avgsumsteps:14dhist"
        )
        COL_SLP_EFF = (
            "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:14dhist"
        )
        COL_SLP_AFWAKE = (
            "f_slp:fitbit_sleep_summary_rapids_avgdurationafterwakeupmain:14dhist"
        )
        COL_SLP_ASLEEP = (
            "f_slp:fitbit_sleep_summary_rapids_avgdurationasleepmain:14dhist"
        )
        COL_SLP_AWAKE = (
            "f_slp:fitbit_sleep_summary_rapids_avgdurationawakemain:14dhist"
        )
        COL_SLP_FALL = (
            "f_slp:fitbit_sleep_summary_rapids_avgdurationtofallasleepmain:14dhist"
        )

        required_cols = [
            COL_STEP_AVG,
            COL_SLP_EFF,
            COL_SLP_AFWAKE,
            COL_SLP_ASLEEP,
            COL_SLP_AWAKE,
            COL_SLP_FALL,
        ]
        for c in required_cols:
            if c not in rap:
                print(f"[WARN] column '{c}' missing in RAPIDS CSV → cohort {insw_idx}")
                continue

        # -------------------- 3) merge label ↔ features ---------------
        for idx, pid in enumerate(labels["pid"]):
            date = labels["date"][idx]

            label_raw = labels["dep"][idx]
            if label_raw.strip() == "":
                continue
            label_val = int(label_raw.lower() == "true")  # bool string → 0/1

            # find matching row in rapids
            try:
                rap_idx = next(
                    i
                    for i, (p, d) in enumerate(zip(rap["pid"], rap["date"]))
                    if p == pid and d == date
                )
            except StopIteration:
                continue  # no features for this record

            feats = {col: maybe_float(rap[col][rap_idx]) for col in required_cols}
            if any(v is None for v in feats.values()):
                continue  # incomplete features

            # build prompt
            instruction = (
                "You are a personalized healthcare agent trained to predict PHQ-4 "
                "depression which ranges from 0 to 4 based on physiological data and "
                "user information."
            )
            input_prompt = (
                "The recent 14-days sensor readings show: "
                f"[Steps] average is {feats[COL_STEP_AVG]}. "
                "Sleep efficiency, minutes after wake-up, minutes asleep, "
                "minutes awake (in bed), and minutes to fall asleep are "
                f"{feats[COL_SLP_EFF]}, {feats[COL_SLP_AFWAKE]}, "
                f"{feats[COL_SLP_ASLEEP]}, {feats[COL_SLP_AWAKE]}, "
                f"{feats[COL_SLP_FALL]} respectively. "
                "What would be the PHQ-4 depression score?"
            )
            output_prompt = f"The predicted PHQ-4 depression score is {label_val}."

            final.append(
                {
                    "instruction": instruction,
                    "input": input_prompt,
                    "output": output_prompt,
                }
            )

    return final


# --------------------------------------------------------------------------
# split & write
# --------------------------------------------------------------------------
def split_and_write(
    examples: List[dict],
    data_name: str,
    subtask: str,
    out_dir: Path,
    seed: int = 123,
):
    random.seed(seed)
    random.shuffle(examples)
    n = len(examples)
    train = examples[: n // 2]
    eval_ = examples[n // 2 :]

    # eval format
    eval_json = [
        {"no": i + 1, "question": ex["input"], "answer": ex["output"]}
        for i, ex in enumerate(eval_)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{data_name}_{subtask}_train_3.json").write_text(
        json.dumps(train[:3], indent=4)
    )
    (out_dir / f"{data_name}_{subtask}_train_10.json").write_text(
        json.dumps(train[:10], indent=4)
    )
    (out_dir / f"{data_name}_{subtask}_train_25.json").write_text(
        json.dumps(train[:25], indent=4)
    )
    (out_dir / f"{data_name}_{subtask}_train_all.json").write_text(
        json.dumps(train, indent=4)
    )

    eval_dir = out_dir / ".." / "eval" / "data" / f"{data_name.lower()}_{subtask}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "step1.json").write_text(json.dumps(eval_json, indent=4))

    print(
        f"[OK] wrote {len(train)} train & {len(eval_json)} eval examples → {out_dir}\n"
        f"[OK] evaluation file → {eval_dir / 'step1.json'}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Path to GLOBEM root")
    ap.add_argument(
        "--subtask",
        choices=["depression"],
        default="depression",
        help="PHQ-4 subscale (only depression implemented)",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("datasets"))
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    examples = build_globem_examples(args.root, args.subtask)
    if not examples:
        print("[ERR] no examples produced – check paths / data integrity", file=sys.stderr)
        sys.exit(1)

    split_and_write(examples, "GLOBEM", args.subtask, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
