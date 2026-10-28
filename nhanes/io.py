"""NHANES loading helpers. Files are read from <NHANES_DATA>/<cycle>/<NAME>.xpt, as written by
nhanes/download.py (an upper-case .XPT is accepted too)."""
import numpy as np
import pandas as pd

from nhanes.paths import DATA

CYCLE_SUFFIX = {
    "2005-2006": "_D",
    "2007-2008": "_E",
    "2009-2010": "_F",
    "2011-2012": "_G",
    "2013-2014": "_H",
    "2015-2016": "_I",
    "2017-2018": "_J",
    "2017-2020": "P_",
}
CYCLES_2005_2018 = list(CYCLE_SUFFIX)[:7]
PHQ_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]


def file_name(component, cycle):
    tag = CYCLE_SUFFIX[cycle]
    return f"P_{component}" if tag == "P_" else f"{component}{tag}"


def find_file(name, cycle):
    for p in (DATA / cycle / f"{name}.xpt", DATA / cycle / f"{name}.XPT"):
        if p.exists():
            return p
    return None


def read_component(component, cycle, columns=None, required=True):
    """Read one component for one cycle. `component` may be a list of alternative
    names (e.g. ["COT", "COTNAL"]); the first one present is used."""
    names = [component] if isinstance(component, str) else list(component)
    for comp in names:
        path = find_file(file_name(comp, cycle), cycle)
        if path is not None:
            df = pd.read_sas(path, format="xport", encoding="latin1")
            if columns is not None:
                keep = ["SEQN"] + [c for c in columns if c in df.columns and c != "SEQN"]
                df = df[keep]
            df["SEQN"] = df["SEQN"].astype("int64")
            return df
    if required:
        raise FileNotFoundError(f"{names} not found for {cycle}")
    return None


def load(component, cycles, columns=None, required=True):
    """Concatenate a component over cycles, adding a `cycle` column."""
    frames = []
    for cycle in cycles:
        df = read_component(component, cycle, columns, required)
        if df is None:
            continue
        df.insert(1, "cycle", cycle)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def clean_codes(s, missing):
    """Replace the given NHANES refused / don't-know codes with NaN. The codes differ
    per variable (7/9, 77/99, 777/999, ...), so they must always be passed explicitly."""
    s = pd.to_numeric(s, errors="coerce")
    # pandas reads SAS missing as NaN and tiny floats (5.4e-79) for coded zero
    s = s.where(~np.isclose(s, 0) | (s == 0), 0.0)
    return s.where(~s.isin(missing))


def phq9(dpq):
    """PHQ-9 total; NaN if any of the nine items is missing/refused/don't know."""
    items = dpq[PHQ_ITEMS].apply(lambda c: clean_codes(c, missing=(7, 9)))
    total = items.sum(axis=1, min_count=9)
    return total.where(items.notna().all(axis=1))
