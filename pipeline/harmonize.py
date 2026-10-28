"""Cross-cycle harmonisation of the items left in pending_features.txt, the prescription file and
the PFOA / PFOS totals. Used by build_features.py. Category mappings follow the CDC code tables in
codebook_values.csv; gates were checked per cycle (see skip_pattern_plan.txt section 9).

Every function returns {name: (values, sources, rule)} like build_features.derive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes import io as nio  # noqa: E402

# variables to load that are not 7-cycle features themselves
EXTRA_LOAD = {
    "DEMO": ["INDFMINC", "INDFMIN2", "INDHHINC", "INDHHIN2", "DMDHRAGE", "DMDHRAGZ", "DMDHREDU", "DMDHREDZ",
             "DMDHSEDU", "DMDHSEDZ", "DMDHRMAR", "DMDHRMAZ", "AIALANG", "AIALANGA", "DMQMILIT", "DMQMILIZ"],
    "SLQ": ["SLD010H", "SLD012"],
    "ALQ": ["ALQ101", "ALQ110", "ALQ111", "ALQ120Q", "ALQ120U", "ALQ121", "ALQ130", "ALQ140Q", "ALQ140U",
            "ALQ141Q", "ALQ141U", "ALQ142", "ALQ150", "ALQ151"],
    "HUQ": ["HUQ030", "HUQ040", "HUQ041", "HUQ050", "HUQ051"],
    "MCQ": ["MCQ190", "MCQ191", "MCQ195"],
    "RHQ": ["RHD042", "RHD043"],
    "SMQ": ["SMQ077", "SMQ078"],
    "SMQRTU": ["SMQ680", "SMQ681", "SMQ851", "SMQ863"],
    "SMQFAM": ["SMD410", "SMD460", "SMD470"],
    "DR1TOT": ["DR1TWS", "DR1TWSZ"],
    "DBQ": ["DBD222U", "DBQ223U", "DBQ223E"],
}
ANTIDEPRESSANT = 249  # Multum level-2 category in RXQ_DRUG (RXDDCI1B-4B)
OTHER_PSYCH = {251: "antipsychotics", 79: "psychotherapeutic combinations",
               67: "anxiolytics, sedatives and hypnotics"}
# left out of rx_count (decided 10/28): mental-health treatment says the outcome is already known
PSYCH_LEVEL1 = {242}  # psychotherapeutic agents (antidepressants, antipsychotics, combinations)
PSYCH_LEVEL2 = {67}   # anxiolytics, sedatives and hypnotics
PER_YEAR = {1: 52, 2: 12, 3: 1}  # ALQ1x0U / ALQ1x1U: week, month, year


# CDC lab calibrations (analytic notes of BIOPRO_D and BIOPRO_J, read 10/29). Only analytes with a
# published equation are adjusted; CDC recommends no adjustment for the others (Na, Cl, K, Ca, HCO3,
# phosphorus, bilirubin, total protein, glucose), and CBC gives no equations for its 2013 change;
# those are left as measured (decided 10/29).
CREATININE_2005 = (0.978, -0.016)  # BIOPRO_D: Deming, standard creatinine on the original 2005-06 values
COBAS_TO_DXC = {  # BIOPRO_J backward equations: 2017-18 (Roche Cobas 6000) -> DxC 660i scale of 1999-2016
    "LBXSAL": (1.044, 0.01128), "LBXSASSI": (1.018, 3.762), "LBXSATSI": (1.013, 2.688),
    "LBXSBU": (1.001, 0.4488), "LBXSCH": (1.046, -2.203), "LBXSCR": (1.051, -0.06945),
    "LBXSGTSI": (0.8042, 2.363), "LBXSIR": (0.9776, -4.494), "LBXSLDSI": (0.8568, 2.062),
    "LBXSTR": (0.9655, -7.020), "LBXSUA": (0.9323, 0.2326),
}
COBAS_TO_DXC_LOG10 = {"LBXSAPSI": (1.001, -0.04294)}  # log10 Deming

def calibrate_labs(d):
    """Put the CDC-calibrated biochemistry on one scale, in place, with the published equations only.
    Returns {column: note} for the dictionary."""
    notes = {}
    old = d.cycle.eq("2005-06")
    a, b = CREATININE_2005
    d.loc[old, "LBXSCR"] = a * d.loc[old, "LBXSCR"] + b
    notes["LBXSCR"] = "2005-06 corrected to standard creatinine (CDC Deming: 0.978 x - 0.016)"
    new = d.cycle.eq("2017-18")
    for c, (a, b) in COBAS_TO_DXC.items():
        if c in d:
            d.loc[new, c] = a * d.loc[new, c] + b
            notes[c] = notes.get(c, "") + f"; 2017-18 Roche Cobas -> Beckman DxC scale ({a} x {b:+})".lstrip("; ")
    for c, (a, b) in COBAS_TO_DXC_LOG10.items():
        if c in d:
            d.loc[new, c] = 10 ** (a * np.log10(d.loc[new, c]) + b)
            notes[c] = f"2017-18 Roche Cobas -> Beckman DxC scale (log10: {a} log x {b:+})"
    # calculated analytes follow their inputs (CDC formulas in BIOPRO_J)
    d.loc[new, "LBXSGB"] = d.loc[new, "LBXSTP"] - d.loc[new, "LBXSAL"]
    d.loc[new, "LBXSOSSI"] = 1.86 * d.loc[new, "LBXSNASI"] + d.loc[new, "LBXSGL"] / 18 + d.loc[new, "LBXSBU"] / 2.8 + 9
    notes["LBXSGB"] = "2017-18 recalculated as total protein - calibrated albumin"
    notes["LBXSOSSI"] = "2017-18 recalculated as 1.86 Na + glucose/18 + calibrated BUN/2.8 + 9"
    return notes


def col(d, *names):
    """First of the names that holds a value, row by row (a renamed item across cycles)."""
    out = pd.Series(np.nan, index=d.index)
    for n in names:
        if n in d:
            out = out.fillna(d[n])
    return out


def to_alq121(days):
    """Days per year -> the ALQ121 / ALQ142 frequency scale (0 never ... 1 every day, 10 1-2 times)."""
    cuts = [(330, 1), (234, 2), (130, 3), (78, 4), (42, 5), (18, 6), (12, 7), (7, 8), (3, 9), (0.5, 10)]
    out = pd.Series(np.nan, index=days.index)
    out[days == 0] = 0
    for lo, code in reversed(cuts):
        out[days >= lo] = code
    return out


def harmonized(d):
    out = {}
    m = lambda s, mapping: s.map(mapping)  # noqa: E731

    inc = {**{k: k for k in range(1, 11)}, 11: 11, 14: 11, 15: 11}  # 12 / 13 (over / under $20k) -> NaN
    out["fam_income_cat"] = (m(col(d, "INDFMINC", "INDFMIN2"), inc), "INDFMINC, INDFMIN2",
                             "1-10 as coded ($0-74,999 in 10 bands), 11 = $75,000+; 'over / under $20,000' -> NaN")
    out["hh_income_cat"] = (m(col(d, "INDHHINC", "INDHHIN2"), inc), "INDHHINC, INDHHIN2", "as fam_income_cat")

    age = d.DMDHRAGE if "DMDHRAGE" in d else pd.Series(np.nan, index=d.index)
    agez = pd.cut(age, [-1, 19, 39, 59, 200], labels=[1, 2, 3, 4]).astype(float)
    out["hh_ref_age_cat"] = (agez.fillna(d.get("DMDHRAGZ")), "DMDHRAGE, DMDHRAGZ",
                             "household reference person's age: 1 <20 / 2 20-39 / 3 40-59 / 4 60+")
    edu3 = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
    marital3 = m(d.DMDHRMAR, {1: 1, 6: 1, 2: 2, 3: 2, 4: 2, 5: 3}).fillna(d.DMDHRMAZ)
    out["hh_ref_edu3"] = (m(d.DMDHREDU, edu3).fillna(d.DMDHREDZ), "DMDHREDU, DMDHREDZ",
                          "1 less than high school / 2 high school, GED, some college / 3 college graduate")
    spouse = m(d.DMDHSEDU, edu3).fillna(d.DMDHSEDZ)
    out["hh_spouse_edu3"] = (spouse.where(~marital3.isin([2, 3]), 0), "DMDHSEDU, DMDHSEDZ, hh_ref_marital3",
                             "as hh_ref_edu3; reference person has no spouse / partner -> 0")
    out["hh_ref_marital3"] = (marital3, "DMDHRMAR, DMDHRMAZ",
                              "1 married or living with partner / 2 widowed, divorced, separated / 3 never married")
    out["acasi_english"] = (m(col(d, "AIALANG", "AIALANGA"), {1: 1, 2: 0, 3: 0}).mask(d.RIDAGEYR >= 70, -1),
                            "AIALANG, AIALANGA", "self-administered interview in English (2011+ adds Asian languages); "
                            "70+ -1 in every cycle (2009+ does not give it to 70+)")
    out["military_service"] = (m(col(d, "DMQMILIT", "DMQMILIZ"), {1: 1, 2: 0}), "DMQMILIT, DMQMILIZ",
                               "served in US armed forces; NOTE 2011+ asks active duty only")

    new_sleep = np.floor(d.SLD012 + 0.5).clip(upper=12)
    out["sleep_hours"] = (d.SLD010H.fillna(new_sleep), "SLD010H, SLD012",
                          "hours of sleep, whole hours, 12 = 12+; NOTE 2015+ asks weekdays / workdays, half-hour "
                          "answers rounded, and its mean is about 1 h higher (6.8 -> 7.7-7.9)")

    # alcohol: who is asked follows ALQ101 / ALQ110 (2005-16) or ALQ111 (2017-18)
    drinker = d.ALQ101.eq(1) | d.ALQ110.eq(1) | d.ALQ111.eq(1)
    abstainer = d.ALQ110.eq(2) | d.ALQ111.eq(2)
    days = d.ALQ120Q * d.ALQ120U.map(PER_YEAR)
    days = days.where(d.ALQ120Q.ne(0), 0)
    freq = to_alq121(days).fillna(d.ALQ121)
    freq = freq.where(~abstainer, 0)
    out["alcohol_freq"] = (freq, "ALQ101, ALQ110, ALQ111, ALQ120Q/U, ALQ121",
                           "ALQ121 scale: 0 none in past year, 1 every day ... 10 1-2 times; 2005-16 days/year "
                           "binned to it; NOTE <12 drinks in lifetime (2005-16) and never drank (2017-18) -> 0")
    none = freq.eq(0)
    out["drinks_per_day"] = (d.ALQ130.clip(upper=15).where(~none, 0), "ALQ130, alcohol_freq",
                             "drinks on a drinking day, capped at 15 (2015+ top code); no drinking -> 0")
    bdays = (col(d, "ALQ140Q", "ALQ141Q") * col(d, "ALQ140U", "ALQ141U").map(PER_YEAR))
    bdays = bdays.where(col(d, "ALQ140Q", "ALQ141Q").ne(0), 0)
    binge = to_alq121(bdays).fillna(d.ALQ142).where(~none, 0)
    out["binge_freq"] = (binge, "ALQ140Q/U, ALQ141Q/U, ALQ142",
                         "days with heavy drinking, ALQ121 scale; NOTE 2005-10: 5+ drinks, 2011+: 4+ (women) / 5+ (men)")
    heavy = col(d, "ALQ150", "ALQ151").map({1: 1, 2: 0})
    out["heavy_drinking_ever"] = (heavy.where(~abstainer, 0), "ALQ150, ALQ151",
                                  "ever drank heavily almost every day; NOTE threshold as binge_freq; never drinkers -> 0")

    old_t = d.SMQ680.map({1: 1, 2: 0})
    parts = d[[c for c in ["SMQ681", "SMQ851", "SMQ863"] if c in d]]
    new_t = np.where(parts.eq(1).any(axis=1), 1, np.where(parts.eq(2).all(axis=1), 0, np.nan))
    out["nicotine_5d"] = (old_t.fillna(pd.Series(new_t, index=d.index)), "SMQ680; SMQ681, SMQ851, SMQ863",
                          "any nicotine product in past 5 days; 2013+ = smoked OR smokeless OR nicotine replacement")
    home = np.where(d.SMD460.eq(0) | d.SMD470.eq(0), 0, np.where(d.SMD470 > 0, 1, np.nan))
    out["home_smoke_inside"] = (d.SMD410.map({1: 1, 2: 0}).fillna(pd.Series(home, index=d.index)),
                                "SMD410; SMD460, SMD470", "someone smokes inside the home; 2013+ from # smokers / # smoking inside")
    not_current = d.SMQ020.eq(2) | d.SMQ040.eq(3)
    out["time_to_first_cig"] = (np.where(not_current, 5, d.SMQ077.fillna(d.SMQ078.clip(upper=4))),
                                "SMQ077, SMQ078, smoking_status",
                                "1 <=5 min / 2 6-30 min / 3 30-60 min / 4 >1 h (2013+ 4-7 collapsed) / 5 not a current smoker")

    place = d.HUQ040.fillna(d.HUQ041.replace({6: 5}))
    out["usual_care_place"] = (np.where(d.HUQ030.eq(2), 0, place), "HUQ030, HUQ040, HUQ041",
                               "0 no usual place / 1 clinic / 2 doctor's office or HMO / 3 ER / 4 hospital outpatient / "
                               "5 other (2013+ 'no one place' folded in)")
    out["healthcare_visits"] = (d.HUQ050.fillna(d.HUQ051.map({0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5})),
                                "HUQ050, HUQ051", "visits in past year: 0 / 1 / 2-3 / 4-9 / 10-12 / 13+")

    arth = (d.MCQ190.map({1: 2, 2: 1, 3: 3}).fillna(d.MCQ191.map({1: 2, 2: 1, 3: 3, 4: 3}))
            .fillna(d.MCQ195.map({1: 1, 2: 2, 3: 3, 4: 3})))
    out["arthritis_type"] = (np.where(d.MCQ160A.eq(2), 0, arth), "MCQ160A, MCQ190, MCQ191, MCQ195",
                             "0 none / 1 osteoarthritis / 2 rheumatoid / 3 other incl. psoriatic; codes 1 and 2 swap in MCQ195")
    reason = d.RHD042.map({1: 1, 2: 2, 7: 3, 8: 4, 9: 4}).fillna(d.RHD043.map({1: 1, 2: 2, 3: 3, 7: 3, 9: 4}))
    out["irregular_period_reason"] = (np.where(d.RIAGENDR.eq(1), -1, np.where(d.RHQ031.eq(1), 0, reason)),
                                      "RHQ031, RHD042, RHD043",
                                      "0 regular periods / 1 pregnancy / 2 breastfeeding / 3 menopause or hysterectomy / 4 other; men -1")
    out["tap_water_source"] = (d.DR1TWS.replace({2: 91, 3: 91}).fillna(d.DR1TWSZ), "DR1TWS, DR1TWSZ",
                               "1 community supply / 4 no tap water / 91 other incl. well, spring")
    other_milk = np.where(col(d, "DBD222U", "DBQ223U").notna() | d.DBQ223E.notna(), 1,
                          np.where(d.DBQ197.isin([0, 1, 2, 3, 4]), 0, np.nan))
    out["DBQ223U"] = (other_milk, "DBD222U, DBQ223U, DBQ223E, DBQ197",
                      "drinks another type of milk (2009+ soy milk counted here, as before 2009)")
    return out


def rx_counts(cycles, short, drug_file):
    """Per person: # distinct prescription drugs in the past month, psychotropics (PSYCH_LEVEL1 /
    PSYCH_LEVEL2) left out since they treat the outcome. Also counts per psychotropic class for the report."""
    drug = pd.read_sas(drug_file, format="xport", encoding="latin1")
    cats = drug[[f"RXDDCI{i}B" for i in range(1, 5)]]
    level1 = drug[[f"RXDDCI{i}A" for i in range(1, 5)]]
    psych = set(drug.RXDDRGID[level1.isin(PSYCH_LEVEL1).any(axis=1) | cats.isin(PSYCH_LEVEL2).any(axis=1)])
    other = {name: set(drug.RXDDRGID[cats.eq(code).any(axis=1)]) for code, name in OTHER_PSYCH.items()}
    frames = []
    for cycle in cycles:
        rx = nio.read_component("RXQ_RX", cycle, ["RXDUSE", "RXDDRGID", "RXDDRUG"])
        use = pd.to_numeric(rx.RXDUSE, errors="coerce")
        person = rx.assign(use=use).groupby("SEQN").use.first()
        taken = rx[use.eq(1)].copy()
        taken["key"] = taken.RXDDRGID.where(taken.RXDDRGID.str.strip() != "", taken.RXDDRUG)
        taken = taken[taken.key.str.strip() != ""].drop_duplicates(["SEQN", "key"])
        f = pd.DataFrame({"use": person})
        f["n_all"] = taken.groupby("SEQN").size()
        f["n_antidep"] = taken[taken.RXDDRGID.isin(set(drug.RXDDRGID[cats.eq(ANTIDEPRESSANT).any(axis=1)]))].groupby("SEQN").size()
        f["n_psych"] = taken[taken.RXDDRGID.isin(psych)].groupby("SEQN").size()
        for name, ids in other.items():
            f[name] = taken[taken.RXDDRGID.isin(ids)].groupby("SEQN").size()
        f = f.fillna({c: 0 for c in f.columns if c != "use"})
        f["rx_count"] = np.where(f.use.eq(2), 0, np.where(f.use.eq(1), f.n_all - f.n_psych, np.nan))
        f["cycle"] = short[cycle]
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True)


def pfoa_pfos(cycles, short):
    """Serum PFOA / PFOS totals (ng/mL) as in paper3: PFC 2005-12 total, SSPFAS 2013-14 and PFAS 2015-18
    linear + branched isomers. Subsample-only."""
    frames = []
    for cycle in cycles:
        if cycle <= "2011-2012":
            d = nio.read_component("PFC", cycle, ["LBXPFOA", "LBXPFOS"])
            d["pfoa"], d["pfos"] = d.LBXPFOA, d.LBXPFOS
        elif cycle == "2013-2014":
            d = nio.read_component("SSPFAS", cycle, ["SSNPFOA", "SSBPFOA", "SSNPFOS", "SSMPFOS"])
            d["pfoa"], d["pfos"] = d.SSNPFOA + d.SSBPFOA, d.SSNPFOS + d.SSMPFOS
        else:
            d = nio.read_component("PFAS", cycle, ["LBXNFOA", "LBXBFOA", "LBXNFOS", "LBXMFOS"])
            d["pfoa"], d["pfos"] = d.LBXNFOA + d.LBXBFOA, d.LBXNFOS + d.LBXMFOS
        frames.append(d[["SEQN", "pfoa", "pfos"]].assign(cycle=short[cycle]))
    return pd.concat(frames, ignore_index=True)
