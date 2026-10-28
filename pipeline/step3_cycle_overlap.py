"""Step 3 of feature screening: which features exist in every 2005-2018 cycle (P_ excluded).
Run after step2_leakage.py; adds to feature_catalog.csv:
  n_cycles_0518  number of 2005-06 ... 2017-18 cycles the (module family, variable) appears in
  in_all_0518    feature present in all 7 cycles
A variable is matched by name only within a module family: renamed files are one family
(PFC->PFAS, COTNAL->COT), anything else is not, because names collide across modules
(LBXHCT = hematocrit in CBC but hydroxycotinine in COT).
Nothing is removed here; renamed items (e.g. INDFMINC -> INDFMIN2) are not harmonised yet.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import WORK  # noqa: E402

C7 = ["2005-06", "2007-08", "2009-10", "2011-12", "2013-14", "2015-16", "2017-18"]
FAMILY = {"PFC": "PFAS", "COTNAL": "COT"}


def main():
    path = WORK / "feature_catalog.csv"
    cat = pd.read_csv(path)
    lab = pd.read_csv(WORK / "variable_labels.csv")
    lab, cat["family"] = lab.assign(family=lab.module.replace(FAMILY)), cat.module.replace(FAMILY)
    per_var = lab[lab.cycle.isin(C7)].groupby(["family", "variable"]).cycle.agg(set)
    key = pd.MultiIndex.from_frame(cat[["family", "variable"]])
    cat["n_cycles_0518"] = [len(per_var.get(k, set())) for k in key]
    cat["in_all_0518"] = cat.is_feature & (cat.n_cycles_0518 == 7)
    cat.drop(columns="family").to_csv(path, index=False)

    f = cat[cat.is_feature & (cat.n_cycles_0518 > 0)].drop_duplicates(["family", "variable"])
    per_var = per_var[pd.MultiIndex.from_frame(f[["family", "variable"]])]
    all7 = f[f.in_all_0518]
    missing = per_var[per_var.map(len) == 6].map(lambda s: (set(C7) - s).pop())
    lines = [f"unique feature variables in >=1 of the 7 cycles: {len(f)}", "",
             "by number of cycles present:",
             f.n_cycles_0518.value_counts().sort_index(ascending=False).to_string(), "",
             f"present in all 7 cycles: {len(all7)} (long_format: {int(all7.long_format.sum())})", "",
             "all-7 features by module:",
             all7.groupby("family").size().sort_values(ascending=False).to_string(), "",
             "features in exactly 6 cycles, by the missing cycle:",
             missing.value_counts().to_string(), "",
             "features in all 6 cycles when one cycle is dropped:"]
    for drop in C7:
        keep = set(C7) - {drop}
        n = per_var.map(lambda s: keep <= s).sum()
        lines.append(f"  drop {drop}: {n}")
    (WORK / "step3_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
