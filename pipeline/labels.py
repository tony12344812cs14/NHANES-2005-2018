"""Collect variable labels from every XPT header -> variable_labels.csv (module, cycle, variable, label)."""
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import DATA, WORK  # noqa: E402

SUF = {"_D": "2005-06", "_E": "2007-08", "_F": "2009-10", "_G": "2011-12", "_H": "2013-14",
       "_I": "2015-16", "_J": "2017-18"}
rows = []
for p in sorted(p for p in DATA.glob("*/*") if p.suffix.lower() == ".xpt"):
    stem = p.stem.upper()
    comp, cyc = (stem[2:], "2017-20P") if stem.startswith("P_") else (stem[:-2], SUF.get(stem[-2:]))
    if cyc is None:
        continue
    for f in pd.read_sas(p, format="xport", iterator=True).fields:
        rows.append((comp, cyc, f["name"].decode().strip(), f["label"].decode("latin1").strip()))
pd.DataFrame(rows, columns=["module", "cycle", "variable", "label"]).to_csv(
    WORK / "variable_labels.csv", index=False)
