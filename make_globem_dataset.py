#!/usr/bin/env python3
"""
make_globem_dataset.py
----------------------

Create instruction-tuning and evaluation JSON files for the GLOBEM dataset
as used in “Health-LLM” (CHIL 2024).

Usage
~~~~~
python make_globem_dataset.py \
       --root   /groups/xx2489_gp/kw3215/Datasets/globem \
       --subtask anxiety                                   \
       --out_dir datasets

Options
~~~~~~~
--subtask     anxiety | depression (default = anxiety)
--root        path to GLOBEM root that contains INS-W_* folders
--out_dir     where to write the resulting JSON files
--seed        random shuffle seed (default = 123)
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm


def maybe_float(x: str):
    """
    Convert to float, but return None if the cell is blank / not numeric.
    """
    try:
        return float(x)
    except ValueError:
        return None


def read_csv(path: Path) -> List[List[str]]:
    """
    Tiny wrapper that reads a CSV into list-of-rows.
    """
    with path.open(newline="") as f:
        return list(csv.reader(f))


def build_globem_examples(root: Path, subtask: str) -> List[Dict[str, Any]]:
    """
    Produce a list of {instruction,input,output} dicts for a single subtask.
    """
    final = []

    ins_template = (
        "You are a personalized healthcare agent trained to predict PHQ-4 {st} "
        "which ranges from 0 to 4 based on physiological data and user information."
    ).format(st=subtask)

    for insw_idx in [1, 2, 3, 4]:
        insw_dir = root / f"INS-W_{insw_idx}"

        # -------- survey (labels) --------
        survey = read_csv(insw_dir / "SurveyData" / "dep_weekly.csv")
        keys1 = survey[0][1:]  # skip first empty cell
        sv = {k: [] for k in keys1}
        for row in survey[1:]:
            for k, cell in zip(keys1, row[1:]):
                sv[k].append(cell)

        # -------- steps features --------
        steps = read_csv(insw_dir / "FeatureData" / "steps.csv")
        keys2 = steps[0][1:]
        st = {k: [] for k in keys2}
        for row in steps[1:]:
            for k, cell in zip(keys2, row[1:]):
                st[k].append(cell)

        # -------- sleep features --------
        sleep = read_csv(insw_dir / "FeatureData" / "sleep.csv")
        keys3 = sleep[0][1:]
        sl = {k: [] for k in keys3}
        for row in sleep[1:]:
            for k, cell in zip(keys3, row[1:]):
                sl[k].append(cell)

        # column aliases for readability
        STEP_AVG = "f_steps:fitbit_steps_summary_rapids_avgsumsteps:14dhist"
        S_EFF = "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:14dhist"
        S_AFW = "f_slp:fitbit_sleep_summary_rapids_avgdurationafterwakeupmain:14dhist"
        S_ASL = "f_slp:fitbit_sleep_summary_rapids_avgdurationasleepmain:14dhist"
        S_AWA = "f_slp:fitbit_sleep_summary_rapids_avgdurationawakemain:14dhist"
        S_FALL = "f_slp:fitbit_sleep_summary_rapids_avgdurationtofallasleepmain:14dhist"
        S_BED = "f_slp:fitbit_sleep_summary_rapids_avgdurationinbedmain:14dhist"

        for idx, pid in enumerate(sv["pid"]):
            # label row
            date = sv["date"][idx]
            try:
                dep = float(sv["feel_depressed"][idx])
                anx = float(sv["feel_anxious"][idx])
            except ValueError:
                continue  # skip if label missing

            # find matching row index in feature tables
            try:
                st_idx = next(
                    i
                    for i, (p, d) in enumerate(zip(st["pid"], st["date"]))
                    if p == pid and d == date
                )
                sl_idx = next(
                    i
                    for i, (p, d) in enumerate(zip(sl["pid"], sl["date"]))
                    if p == pid and d == date
                )
            except StopIteration:
                continue  # no matching features → skip

            avg_steps = maybe_float(st[STEP_AVG][st_idx])
            eff = maybe_float(sl[S_EFF][sl_idx])
            dura_fwake = maybe_float(sl[S_AFW][sl_idx])
            dura_sleep = maybe_float(sl[S_ASL][sl_idx])
            dura_awake = maybe_float(sl[S_AWA][sl_idx])
            dura_fall = maybe_float(sl[S_FALL][sl_idx])
            dura_bed = maybe_float(sl[S_BED][sl_idx])

            # skip if any critical feature missing
            if None in (
                avg_steps,
                eff,
                dura_fwake,
                dura_sleep,
                dura_awake,
                dura_fall,
                dura_bed,
            ):
                continue

            # prompt
            question = (
                "The recent 14-days sensor readings show: "
                f"[Steps] is {avg_steps}. "
                "[Sleep] efficiency, duration the user stayed in bed after waking up, "
                "duration the user spent to fall asleep, duration the user stayed awake "
                f"but still in bed, duration the user spent to fall asleep are "
                f"{eff}, {dura_fwake}, {dura_sleep}, {dura_awake}, {dura_fall} mins "
                f"in average; What would be the PHQ-4 {subtask} score?"
            )

            answer_val = int(dep) if subtask == "depression" else int(anx)
            answer = f"The predicted PHQ-4 {subtask} score is {answer_val}."

            final.append({"instruction": ins_template, "input": question, "output": answer})

    return final


def split_and_write(ex: List[dict], data: str, subtask: str, out_dir: Path, seed: int = 123):
    """
    Shuffle, split 50/50 into train/eval, produce 3-/10-/25-shot and full-train JSON.
    """
    random.seed(seed)
    random.shuffle(ex)
    n = len(ex)
    train, eval = ex[: n // 2], ex[n // 2 :]

    # evaluation format → no / question / answer
    eval_json = []
    for i, item in enumerate(eval, 1):
        eval_json.append(
            {
                "no": i,
                "question": item["input"],
                "answer": item["output"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{data}_{subtask}_train_3.json").write_text(json.dumps(train[:3], indent=4))
    (out_dir / f"{data}_{subtask}_train_10.json").write_text(json.dumps(train[:10], indent=4))
    (out_dir / f"{data}_{subtask}_train_25.json").write_text(json.dumps(train[:25], indent=4))
    (out_dir / f"{data}_{subtask}_train_all.json").write_text(json.dumps(train, indent=4))

    eval_dir = out_dir / ".." / "eval" / "data" / f"{data.lower()}_{subtask}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "step1.json").write_text(json.dumps(eval_json, indent=4))

    print(
        f"[OK] wrote {len(train)} train examples & {len(eval_json)} eval examples "
        f"to: {out_dir} / {eval_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Path to GLOBEM root dir")
    ap.add_argument(
        "--subtask",
        choices=["depression", "anxiety"],
        default="anxiety",
        help="Which PHQ-4 subscale to build",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("datasets"), help="output dir")
    ap.add_argument("--seed", type=int, default=123, help="shuffle seed")
    args = ap.parse_args()

    examples = build_globem_examples(args.root, args.subtask)
    if not examples:
        print("No examples produced -- please check paths / data integrity", file=sys.stderr)
        sys.exit(1)

    split_and_write(examples, "GLOBEM", args.subtask, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
