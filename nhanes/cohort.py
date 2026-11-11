"""Analysis population and cycle names shared by the pipeline notebooks (step 5 onwards) and
data_analysis.

Target population (per cycle): RIDAGEYR >= 20, not pregnant at exam (RIDEXPRG != 1) and all nine
PHQ-9 items DPQ010-DPQ090 answered 0-3.
"""
from nhanes import io as nio

CYCLES = nio.CYCLES_2005_2018
SHORT = dict(zip(CYCLES, ["2005-06", "2007-08", "2009-10", "2011-12", "2013-14", "2015-16",
                          "2017-18"]))
FAMILY = {"PFC": "PFAS", "COTNAL": "COT"}  # renamed files, read as one module


def population(cycle):
    demo = nio.read_component("DEMO", cycle, ["RIDAGEYR", "RIAGENDR", "RIDEXPRG"])
    dpq = nio.read_component("DPQ", cycle)
    items = dpq[nio.PHQ_ITEMS].apply(lambda s: nio.clean_codes(s, [7, 9]))
    ok = dpq.SEQN[items.notna().all(axis=1)]
    keep = (demo.RIDAGEYR >= 20) & (demo.RIDEXPRG != 1) & demo.SEQN.isin(ok)
    return demo[keep].reset_index(drop=True)
