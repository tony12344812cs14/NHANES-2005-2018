"""Step 5 of feature screening (analysis only, nothing removed): missing rates of the features
present in all 7 cycles 2005-2018 (in_all_0518), in the target population.

Target population (per cycle): RIDAGEYR >= 20, not pregnant at exam (RIDEXPRG != 1) and all
nine PHQ-9 items DPQ010-DPQ090 answered 0-3. "Missing" = empty cell (SAS missing) after merging onto the population; refused / don't
know codes (7/9, 77/99, ...) are NOT counted here because the codes differ per variable.
Renamed pairs (rename_map.csv) are merged first. Long-format modules (RXQ_RX) are skipped:
they need aggregation before a per-person missing rate means anything.

Output:
  missing_rates.csv   one row per feature: module, variable, label, overall rate,
                      rate per cycle, spread (max - min over cycles), likely cause,
                      checkbox (code-all-that-apply item: only one substantive code >= 10,
                      so an empty cell means "not chosen", not missing),
                      empty_cycles (# cycles with >= 99% missing)
  step5_counts.txt    population size, distribution of missing rates, per-module summary
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import WORK  # noqa: E402
from nhanes import io as nio  # noqa: E402

CYCLES = nio.CYCLES_2005_2018
SHORT = dict(zip(CYCLES, ["2005-06", "2007-08", "2009-10", "2011-12", "2013-14", "2015-16",
                          "2017-18"]))
FAMILY = {"PFC": "PFAS", "COTNAL": "COT"}
SUBSAMPLE = {"GLU": "fasting subsample", "TRIGLY": "fasting subsample", "OGTT": "fasting subsample",
             "PFAS": "1/3 subsample", "PFC": "1/3 subsample"}
SPECIAL = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999}
BINS = [-0.001, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0]
BIN_NAMES = ["<5%", "5-20%", "20-50%", "50-80%", "80-95%", ">=95%"]


def population(cycle):
    demo = nio.read_component("DEMO", cycle, ["RIDAGEYR", "RIAGENDR", "RIDEXPRG"])
    dpq = nio.read_component("DPQ", cycle)
    items = dpq[nio.PHQ_ITEMS].apply(lambda s: nio.clean_codes(s, [7, 9]))
    ok = dpq.SEQN[items.notna().all(axis=1)]
    keep = (demo.RIDAGEYR >= 20) & (demo.RIDEXPRG != 1) & demo.SEQN.isin(ok)
    return demo[keep].reset_index(drop=True)


def main():
    cat = pd.read_csv(WORK / "feature_catalog.csv")
    rmap = pd.read_csv(WORK / "rename_map.csv")
    old_of = {(r.module, r.new): r.old for r in rmap.itertuples()}
    feat = cat[cat.in_all_0518 & ~cat.long_format].copy()
    feat["family"] = feat.module.replace(FAMILY)
    feat = feat.drop_duplicates(["family", "variable"])
    modules = sorted(set(feat.module) | {m for m, f in FAMILY.items() if f in set(feat.family)})

    rates, sizes, women, values = {}, {}, {}, {}
    for cycle in CYCLES:
        pop = population(cycle)
        sizes[SHORT[cycle]] = len(pop)
        women[SHORT[cycle]] = pop.RIAGENDR.eq(2)
        frames = {}
        for m in modules:
            d = nio.read_component(m, cycle, required=False)
            if d is not None:
                frames[FAMILY.get(m, m)] = pop[["SEQN"]].merge(d, on="SEQN", how="left")
        for r in feat.itertuples():
            d = frames.get(r.family)
            col = r.variable if d is not None and r.variable in d else old_of.get((r.module, r.variable))
            if d is None or col not in d:
                raise KeyError(f"{r.variable} not found in {r.family} {cycle}")
            rates[(r.module, r.variable, SHORT[cycle])] = (d[col].isna().mean(), d[col].isna())
            v = pd.to_numeric(d[col], errors="coerce").dropna().round(3).unique()
            values.setdefault((r.module, r.variable), set()).update(v[:50])

    cyc = list(SHORT.values())
    rows = []
    for r in feat.itertuples():
        per = [rates[(r.module, r.variable, c)][0] for c in cyc]
        miss = pd.concat([rates[(r.module, r.variable, c)][1] for c in cyc], ignore_index=True)
        fem = pd.concat([women[c] for c in cyc], ignore_index=True)
        overall = miss.mean()
        if r.module in SUBSAMPLE:
            cause = SUBSAMPLE[r.module]
        elif overall > 0.4 and miss[~fem].mean() > 0.99 and miss[fem].mean() < 0.9:
            cause = "women only"
        elif overall > 0.05:
            cause = "skip pattern / not asked"
        else:
            cause = ""
        core = values.get((r.module, r.variable), set()) - SPECIAL
        rows.append({"module": r.module, "variable": r.variable, "label": r.label,
                     "missing_overall": round(overall, 4),
                     **{f"miss_{c}": round(p, 4) for c, p in zip(cyc, per)},
                     "spread": round(max(per) - min(per), 4), "likely_cause": cause,
                     "checkbox": len(core) == 1 and min(core) >= 10,
                     "empty_cycles": sum(p >= 0.99 for p in per)})
    out = pd.DataFrame(rows).sort_values(["missing_overall", "module", "variable"])
    out.to_csv(WORK / "missing_rates.csv", index=False)

    out["bin"] = pd.cut(out.missing_overall, BINS, labels=BIN_NAMES)
    out["is_diet"] = out.module.isin(["DR1TOT", "DR2TOT"])
    by_mod = (out.groupby("module")
              .agg(n=("variable", "size"), median=("missing_overall", "median"),
                   lt20=("missing_overall", lambda s: int((s < 0.2).sum())),
                   gt80=("missing_overall", lambda s: int((s > 0.8).sum())))
              .sort_values("median"))
    spread = out[out.spread > 0.2].sort_values("spread", ascending=False)
    part_empty = out[(out.empty_cycles > 0) & (out.empty_cycles < len(cyc))]
    all_empty = out[out.empty_cycles == len(cyc)]
    lines = ["population: age >= 20, not pregnant, PHQ-9 items all answered (0-3)",
             pd.Series(sizes, name="n").to_string(), f"total: {sum(sizes.values())}", "",
             f"features analysed: {len(out)} (all-7-cycle features; long-format RXQ_RX skipped)", "",
             "overall missing rate distribution (all / non-diet / diet DR1TOT+DR2TOT):",
             pd.DataFrame({"all": out.bin.value_counts(sort=False),
                           "non_diet": out[~out.is_diet].bin.value_counts(sort=False),
                           "diet": out[out.is_diet].bin.value_counts(sort=False)}).to_string(), "",
             "likely cause among features with missing >= 5%:",
             out[out.missing_overall >= 0.05].likely_cause.value_counts().to_string(), "",
             "per module (median missing rate, # features <20%, # features >80%):",
             by_mod.to_string(), "",
             f"features whose missing rate differs by >20 points across cycles: {len(spread)}",
             spread[["module", "variable", "label", *[f"miss_{c}" for c in cyc]]]
             .to_string(index=False, float_format=lambda x: f"{x:.2f}"), "",
             f"checkbox items (empty = not chosen): {int(out.checkbox.sum())}",
             out[out.checkbox].groupby("module").size().to_string(), "",
             f">= 99% missing in every cycle: {len(all_empty)}",
             all_empty.groupby("module").size().to_string(), "",
             f">= 99% missing in some cycles only: {len(part_empty)}",
             part_empty[["module", "variable", "label", "empty_cycles"]].to_string(index=False)]
    (WORK / "step5_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
