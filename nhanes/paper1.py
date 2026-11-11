"""The 86 features of Dang et al. (2024) ("paper1"), shared by modeling steps 3 and 5-8.

Moved unchanged from modeling/step3_paper1_features.py (paper1_table), step6_paper1_recoded.py
(recoded_table, frames; frames now takes the paper's CSV as an argument instead of a module global) and
step7_phq9_target.py (phq9_totals).

The paper's table FullData_rm_dl_bt.csv and its builder build_dataset.py come from our reproduction of the
paper and are not part of this repo. Categorical values there are "v<code>" or a drug name, with "missing"
as its own value; paper1_table turns them into integer codes with missing = NaN. Numeric columns are used
as they are (the paper's imputation: median for weight/height/BMI, 0 for skipped questions).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nhanes.loco import NOT_FEATURES, SEED
from nhanes.paths import WORK

ONSET = ["asthma_onset", "arthritis_onset", "heart_failure_onset", "heart_disease_onset", "angina_onset",
         "heart_attack_onset", "stroke_onset", "emphysema_onset", "bronchitis_onset", "liver_condition_onset",
         "thyroid_problem_onset", "cancer_onset"]
BODY = ["weight", "height", "BMI"]


def paper1_table(path):
    p = pd.read_csv(path, low_memory=False).drop(columns="depression")
    cat = [c for c in p if p[c].dtype == object]
    for c in cat:
        s = p[c].mask(p[c] == "missing")
        codes = pd.to_numeric(s.str[1:], errors="coerce") if s.dropna().str.fullmatch(r"v-?\d+").all() else None
        p[c] = codes if codes is not None else pd.Series(pd.factorize(s, sort=True)[0], index=p.index).replace(-1, np.nan)
    return p, {c: "categorical" if c in cat else "continuous" for c in p if c != "SEQN"}


def recoded_table(repro_dir, paper_csv, out_csv, drug_file):
    """Rebuild the paper's table without its fill step and re-encode it (see module docstring)."""
    sys.path.insert(0, str(repro_dir))
    import build_dataset as bd
    df = bd.build(fill=False)[0]
    paper = pd.read_csv(paper_csv, index_col=0, low_memory=False)
    assert df.index.equals(paper.index), "rebuilt table does not match the paper's table"
    num = [c for c in bd.NUMERICAL if c != "age"]
    assert (df[num].fillna(0).drop(columns=BODY) == paper[num].drop(columns=BODY)).all().all()
    onset = [c for c in df if c.endswith("_onset")]
    for c in onset:
        df[c.replace("_onset", "_years")] = (df.age - df[c]).clip(lower=0).fillna(-1)
    df = df.drop(columns=onset)
    other = [c for c in num if c not in BODY and c not in onset]
    df[other] = df[other].fillna(-1)
    drug = pd.read_sas(drug_file, format="xport", encoding="latin1")
    cls = dict(zip(drug.RXDDRUG.str.upper().str.strip(), drug.RXDDCN1A.fillna("unknown")))
    name = df.pop("RXDDRUG")
    df["first_drug_class"] = np.where(name == "NONE", "none", name.map(cls).fillna("unmatched"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"recoded table: {df.shape[1] - 1} features; onset -> years: {len(onset)}; "
          f"drug names matched to a class: {(df.first_drug_class != 'unmatched').mean():.3f}; "
          f"classes: {df.first_drug_class.nunique()}", flush=True)
    return out_csv


def frames(table_csv, how, quick, paper_csv=None):
    """Yield (split name, frame, inner groups) with frame.cycle = 'TEST' for test rows and the inner group
    for training rows, so step 1's tune() and step 2's run_catboost() work unchanged."""
    feats, dic = paper1_table(table_csv)
    if how == "loco":
        ours = pd.read_csv(WORK / "features" / "features_2005_2018.csv.gz", usecols=NOT_FEATURES)
        data = ours.merge(feats, on="SEQN", how="left", validate="1:1")
        cycles = sorted(data.cycle.unique())
        for c in cycles[:1] if quick else cycles:
            f = data.copy()
            f["cycle"] = np.where(f.cycle == c, "TEST", f.cycle)
            yield c, f, [x for x in cycles if x != c], dic
        return
    y = pd.read_csv(paper_csv, usecols=["depression"]).depression.eq("Depressed").astype(int)
    data = feats.assign(depression=y.values, PHQ9=np.nan, WTMEC2YR=1.0, WTINT2YR=1.0, SDMVPSU=0, SDMVSTRA=0)
    outer = StratifiedKFold(10, shuffle=True, random_state=42).split(data, data.depression)
    for k, (tr, te) in enumerate(outer):
        if quick and k:
            return
        f = data.copy()
        f["cycle"] = "TEST"
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED).split(tr, data.depression.iloc[tr])
        for g, (_, idx) in enumerate(inner):
            f.iloc[tr[idx], f.columns.get_loc("cycle")] = f"g{g}"
        yield f"fold{k}", f[NOT_FEATURES + list(dic)], ["g0", "g1", "g2"], dic


def phq9_totals(repro_dir):
    sys.path.insert(0, str(repro_dir))
    import build_dataset as bd
    dpq = pd.concat([bd.nio.read_component("DPQ", c) for c in bd.CYCLES], ignore_index=True)
    return pd.Series(bd.nio.phq9(dpq).values, index=dpq.SEQN.values)
