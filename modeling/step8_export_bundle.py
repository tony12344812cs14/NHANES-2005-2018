"""Step 8a: export a self-contained bundle for the tabular foundation models (run on another machine).

The bundle holds everything step8_foundation_models.py needs, so the GPU machine needs neither NHANES
data nor this pipeline: the re-encoded 86 variables of Dang et al. (2024) (step 6), the 0/1 label, the
PHQ-9 total (step 7) and the paper's 10-fold split (the same StratifiedKFold(10, shuffle, seed 42) as
steps 6 and 7, so predictions can be paired fold by fold).

Usage: python modeling/step8_export_bundle.py PAPER1_REPRO_DIR PAPER1_CSV [OUT_DIR]
       (run step 6 first; OUT_DIR defaults to <NHANES_WORK>/modeling/step8_bundle)
Output: OUT_DIR/data.csv.gz  SEQN, fold, depression, phq9, then the 86 features
        OUT_DIR/columns.json {"categorical": [...], "continuous": [...]}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import RESULTS  # noqa: E402
import step6_paper1_recoded as s6  # noqa: E402
from step3_paper1_features import paper1_table  # noqa: E402
from step7_phq9_target import phq9_totals  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repro_dir", type=Path)
    ap.add_argument("paper_csv", type=Path)
    ap.add_argument("out_dir", type=Path, nargs="?", default=RESULTS / "step8_bundle")
    args = ap.parse_args()
    feats, dic = paper1_table(s6.OUT / "data" / "FullData_recoded.csv")
    y = pd.read_csv(args.paper_csv, usecols=["depression"]).depression.eq("Depressed").astype(int).to_numpy()
    fold = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(feats, y)):
        fold[te] = k
    phq = feats.SEQN.map(phq9_totals(args.repro_dir))
    assert phq.notna().all() and ((phq >= 10).astype(int).to_numpy() == y).all()
    cols = [c for c in feats if c != "SEQN"]
    out = pd.concat([pd.DataFrame({"SEQN": feats.SEQN, "fold": fold, "depression": y, "phq9": phq.astype(int)}),
                     feats[cols]], axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "data.csv.gz", index=False)
    (args.out_dir / "columns.json").write_text(json.dumps(
        {"categorical": [c for c in cols if dic[c] == "categorical"],
         "continuous": [c for c in cols if dic[c] != "categorical"]}, indent=1))
    print(f"{len(out)} people, {len(cols)} features, depression {y.mean():.3f}; written to {args.out_dir}")


if __name__ == "__main__":
    main()
