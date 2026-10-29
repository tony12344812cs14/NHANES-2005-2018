"""Step 3: our LOCO design with the 86 features of Dang et al. (2024) instead of ours.

Same people (our 33,498), same outcome, same folds, models and grids as step 1; only the feature set
changes. This separates "better features" from "different evaluation" when comparing with the paper
(its NHANES LR: AUROC 0.793, random 10-fold).

Input: the paper's feature table FullData_rm_dl_bt.csv (built in our reproduction of the paper; not part
of this repo). Categorical values are "v<code>" or a drug name, with "missing" as its own value; they
become integer codes with missing = NaN. Numeric columns are used as they are (the paper's imputation:
median for weight/height/BMI, 0 for skipped questions).

Usage: python modeling/step3_paper1_features.py PATH/TO/FullData_rm_dl_bt.csv [--quick]
Output: <NHANES_WORK>/modeling/step3_paper1/ (same files as step 1)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import RESULTS, WORK  # noqa: E402
from step1_loco import NOT_FEATURES, run_loco  # noqa: E402

OUT = RESULTS / "step3_paper1"


def paper1_table(path):
    p = pd.read_csv(path, low_memory=False).drop(columns="depression")
    cat = [c for c in p if p[c].dtype == object]
    for c in cat:
        s = p[c].mask(p[c] == "missing")
        codes = pd.to_numeric(s.str[1:], errors="coerce") if s.dropna().str.fullmatch(r"v-?\d+").all() else None
        p[c] = codes if codes is not None else pd.Series(pd.factorize(s, sort=True)[0], index=p.index).replace(-1, np.nan)
    return p, {c: "categorical" if c in cat else "continuous" for c in p if c != "SEQN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper1_csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    feats, dic = paper1_table(args.paper1_csv)
    ours = pd.read_csv(WORK / "features" / "features_2005_2018.csv.gz", usecols=NOT_FEATURES)
    data = ours.merge(feats, on="SEQN", how="left", validate="1:1")
    assert data[list(dic)].notna().any(axis=1).all(), "some of our participants are missing from the paper's table"
    print(f"{len(data)} people, {len(dic)} features "
          f"({sum(v == 'categorical' for v in dic.values())} categorical)", flush=True)
    run_loco(data, dic, OUT, args.quick,
             title="leave-one-cycle-out, outcome PHQ-9 >= 10; features of Dang et al. (2024), our population")


if __name__ == "__main__":
    main()
