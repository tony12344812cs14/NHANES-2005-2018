"""Step 4 of feature screening: merge variables that were only renamed between 2005-2018 cycles.
Run after step3_cycle_overlap.py; updates feature_catalog.csv and writes rename_map.csv.

Every pair below was checked against the CDC codebooks (same question text, target and value
codes) and against the data (value ranges and shares continuous across cycles). See
pending_features.txt for the candidates that were NOT pure renames.

  rename_map.csv  old, new, module, transform  (transform to apply when building the column)
  catalog         old variable -> category "renamed", is_feature False, reason "merged into NEW";
                  new variable -> n_cycles_0518 / in_all_0518 recomputed over old + new cycles
"""
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import WORK  # noqa: E402

C7 = ["2005-06", "2007-08", "2009-10", "2011-12", "2013-14", "2015-16", "2017-18"]

CLIP_16_80 = ("clip valid ages to [16, 80] in every cycle (2005-06 top-codes at 85, "
              "2017-18 bottom-codes at 16); keep 77777/99999 as refused/don't know")
ADULT_ONLY = "2005-06 also asked ages 12-19; later cycles set them missing, so drop <20 in 2005-06"
RENAMES = [
    ("DIQ", "DID070", "DIQ070", ""),
    ("DIQ", "DID340", "DID341", ""),
    *[("MCQ", f"MCQ180{x}", f"MCD180{x}", CLIP_16_80) for x in "ABCDEFGKLMN"],
    ("RHQ", "RHQ180", "RHD180", ADULT_ONLY),
    ("RHQ", "RHQ190", "RHD190", ADULT_ONLY),
    *[("RHQ", f"RHQ541{x}", f"RHQ542{x}", "") for x in "ABC"],
    *[("DBQ", f"DBD222{x}", f"DBQ223{x}", "") for x in "ABCD"],
]


def main():
    path = WORK / "feature_catalog.csv"
    cat = pd.read_csv(path)
    rmap = pd.DataFrame(RENAMES, columns=["module", "old", "new", "transform"])
    rmap.to_csv(WORK / "rename_map.csv", index=False)

    lab = pd.read_csv(WORK / "variable_labels.csv")
    lab = lab[lab.cycle.isin(C7)]
    for r in rmap.itertuples():
        old = (cat.module == r.module) & (cat.variable == r.old)
        new = (cat.module == r.module) & (cat.variable == r.new)
        assert old.sum() == 1 and new.sum() == 1, (r.old, r.new)
        cyc = set(lab[(lab.module == r.module) & lab.variable.isin([r.old, r.new])].cycle)
        cat.loc[old, ["category", "reason", "is_feature", "in_all_0518"]] = \
            ["renamed", f"merged into {r.new}", False, False]
        cat.loc[new, "n_cycles_0518"] = len(cyc)
        cat.loc[new, "in_all_0518"] = len(cyc) == 7
    cat.to_csv(path, index=False)

    fam = cat.module.replace({"PFC": "PFAS", "COTNAL": "COT"})
    all7 = cat[cat.in_all_0518].assign(family=fam).drop_duplicates(["family", "variable"])
    lines = [f"renamed pairs merged: {len(rmap)}",
             f"features present in all 7 cycles (2005-2018) after step 4: {len(all7)}", "",
             rmap.to_string(index=False)]
    (WORK / "step4_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))


if __name__ == "__main__":
    main()
