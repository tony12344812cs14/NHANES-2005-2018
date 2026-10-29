"""Step 4: how much of the AUROC comes from features that overlap with what PHQ-9 measures.

None of these features is derived from PHQ-9 (so they are not leakage), but they ask about things the
PHQ-9 items also cover. Groups:
  sleep        PHQ-9 item 3 (sleep)                    SLQ050, sleep_hours
  self_health  self-rated general health                HSD010, HUQ010, HUQ020
  function     PHQ-9 item 10 (difficulty), also fatigue PFQ049, PFQ051, PFQ054, PFQ057 (confusion/memory),
               and concentration                        PFQ090, pf_difficulty_count, pf_severe_count,
                                                        pf_notdo_count, not_working_reason

Variants (same people, folds and grids as step 1):
  all          all three groups removed; logreg and xgboost
  sleep / self_health / function   one group removed; xgboost only (the two models tie in steps 1 and 3)

Usage: python modeling/step4_phq_overlap.py [--variant all|sleep|self_health|function] [--quick]
       (no --variant: run all four)
Output: <NHANES_WORK>/modeling/step4_overlap/<variant>/ (same files as step 1)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import RESULTS  # noqa: E402
from step1_loco import FEAT, run_loco  # noqa: E402

OUT = RESULTS / "step4_overlap"
GROUPS = {
    "sleep": ["SLQ050", "sleep_hours"],
    "self_health": ["HSD010", "HUQ010", "HUQ020"],
    "function": ["PFQ049", "PFQ051", "PFQ054", "PFQ057", "PFQ090", "pf_difficulty_count", "pf_severe_count",
                 "pf_notdo_count", "not_working_reason"],
}
VARIANTS = {**{g: (cols, ("xgboost",)) for g, cols in GROUPS.items()},
            "all": (sum(GROUPS.values(), []), ("logreg", "xgboost"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = pd.read_csv(FEAT / "features_2005_2018.csv.gz")
    dic = pd.read_csv(FEAT / "feature_dictionary.csv").set_index("column").type.to_dict()
    for name in [args.variant] if args.variant else VARIANTS:
        drop, models = VARIANTS[name]
        assert set(drop) <= set(data.columns), set(drop) - set(data.columns)
        print(f"== {name}: drop {len(drop)} features, {data.shape[1] - 8 - len(drop)} left; {models}", flush=True)
        run_loco(data.drop(columns=drop), dic, OUT / name, args.quick, models=models,
                 title=f"leave-one-cycle-out, outcome PHQ-9 >= 10; without PHQ-overlap group '{name}': {drop}")


if __name__ == "__main__":
    main()
