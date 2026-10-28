"""Step 2 of feature screening: tag outcome leakage, assuming the outcome is depression from
PHQ-9 (DPQ). Run after step1_nonfeature.py; updates feature_catalog.csv in place.

Categories added (is_feature = False):
  outcome        the PHQ-9 items themselves (DPQ010-DPQ090) and DPQ100, which is asked only
                 when any item is endorsed
  outcome_proxy  concurrent self-reported mental-health status or mental-health care, which
                 measures the outcome rather than predicting it

Kept as features but given a `leak_note` (handle when the column is built):
  PFQ063A-E      condition codes causing limitation; code 14 = depression/anxiety/emotional
                 problem (CDC codebook PFQ_D and PFQ_J; 13 = cancer)
  RXQ_RX         drug-level file; antidepressants must be excluded when aggregating
  mixed items    limitation questions that pool physical and mental causes
"""
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import WORK  # noqa: E402


OUTCOME = {f"DPQ0{i}0" for i in range(1, 10)} | {"DPQ100"}
OUTCOME_PROXY = {
    "HSQ480": "days mental health not good (past 30 d)",
    "HSQ496": "days felt anxious (past 30 d)",
    "PFD069D": "days limited by depression/anxiety/emotional problem",
    "HUQ090": "seen mental health professional (past yr)",
    "RHQ551B": "used hormones for emotional symptoms",
}
LEAK_NOTE = {
    **{f"PFQ063{x}": "code 14 = depression/anxiety/emotional problem; drop that code"
       for x in "ABCDE"},
    "RXDDRGID": "exclude antidepressants when aggregating",
    "HSQ490": "pools physical and mental causes",
    "PFQ059": "pools physical, mental and emotional limitations",
    "PFQ010": "pools physical, mental and emotional limitations",
}


def main():
    path = WORK / "feature_catalog.csv"
    cat = pd.read_csv(path)
    cat["leak_note"] = cat.variable.map(LEAK_NOTE).fillna("")
    feat = cat.is_feature
    out = feat & cat.variable.isin(OUTCOME)
    proxy = feat & cat.variable.isin(OUTCOME_PROXY)
    cat.loc[out, ["category", "reason"]] = ["outcome", "PHQ-9 outcome item"]
    cat.loc[proxy, "category"] = "outcome_proxy"
    cat.loc[proxy, "reason"] = cat.loc[proxy, "variable"].map(OUTCOME_PROXY)
    cat["is_feature"] = cat.category == "feature"
    cat["long_format"] = cat.long_format & cat.is_feature
    cat.to_csv(path, index=False)

    lab = pd.read_csv(WORK / "variable_labels.csv").merge(
        cat[["module", "variable", "is_feature"]], on=["module", "variable"])
    rows = [{"cycle": c, "features": d[d.is_feature].variable.nunique()}
            for c, d in lab.groupby("cycle", sort=False)]
    lines = ["(module, variable) pairs by category:", cat.category.value_counts().to_string(), "",
             "unique feature variables per cycle after step 2:",
             pd.DataFrame(rows).to_string(index=False), "",
             "kept with leak_note:",
             cat[cat.leak_note != ""][["module", "variable", "leak_note"]].to_string(index=False)]
    (WORK / "step2_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
