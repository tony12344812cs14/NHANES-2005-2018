"""KNHANES (Korea National Health and Nutrition Examination Survey) 2014 and 2016 in the NHANES
feature definitions, for the comparison in data_analysis/03 and the external test.

KNHANES needs a registration to download, so the files are not fetched here. Put the yearly files
`hnYY_all.sas7bdat` (unzipped from HNYY_ALL(SAS).zip) in one folder and set `KNHANES_DATA` to it
(default: $NHANES_DATA/knhanes).

Population, as in NHANES: age >= 20, not pregnant (HE_prg != 1), all nine PHQ-9 items
BP_PHQ_1-9 answered 0-3. Outcome: PHQ-9 total >= 10.

Codes come from the KNHANES user guides (원시자료 이용지침서, 6th period 2013-15 and 7th period
2016-18). KNHANES codes 8 / 88 / 888 as "not applicable" (skipped by a gate question) and 9 / 99 /
999 as "don't know / no answer". Every feature follows the NHANES encoding of the feature table:
the answer or a logically certain value, -1 where the questionnaire does not ask that group, empty
when truly missing. docs/knhanes_mapping.csv says, for each feature, whether the KNHANES
question or measurement is the same as in NHANES (`exact`) or only close (`recode`), and why.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes.paths import DATA

KDATA = Path(os.environ.get("KNHANES_DATA", DATA / "knhanes"))
YEARS = [2014, 2016]
PHQ = [f"BP_PHQ_{i}" for i in range(1, 10)]
LB_PER_KG = 2.20462
# KRW per USD, purchasing power parity for GDP (OECD), to put household income on the NHANES bands
PPP = {2014: 869.5, 2016: 865.9}
# upper limits (USD per year) of NHANES income codes 1-10; 11 = $75,000 and over
INCOME_BANDS = [5000, 10000, 15000, 20000, 25000, 35000, 45000, 55000, 65000, 75000]


def read_year(year):
    """Raw KNHANES file of one year (all columns, numeric codes as floats)."""
    path = KDATA / f"hn{year % 100:02d}_all.sas7bdat"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: set KNHANES_DATA to the folder with hnYY_all.sas7bdat")
    raw = pd.read_sas(path, format="sas7bdat")  # strings stay bytes; only numeric codes are used
    return raw.rename(columns={"ID": "id"})       # 2016 spells it ID


def population(raw):
    items = raw[PHQ]
    keep = (raw.age >= 20) & (raw.HE_prg != 1) & items.isin([0, 1, 2, 3]).all(axis=1)
    return raw[keep].reset_index(drop=True)


# --- helpers -----------------------------------------------------------------------------------

def num(s, missing=()):
    """Numeric value with the KNHANES missing / not-applicable codes as NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.isin(list(missing)))


def code(s, table):
    """Map codes with `table`; codes not in it become NaN."""
    return pd.to_numeric(s, errors="coerce").map(table)


def yes_no(s):
    """KNHANES 1 yes / 2 no -> NHANES 1 yes / 2 no."""
    return code(s, {1: 1, 2: 2})


def dx(s):
    """Doctor diagnosis 0 no / 1 yes -> NHANES 1 yes / 2 no."""
    return code(s, {1: 1, 0: 2})


def flag(s, na_as=0):
    """0 / 1 item whose 8 (not applicable: gate question no) is a certain 0."""
    return code(s, {1: 1, 0: 0, 8: na_as})


def any_of(df, cols, yes=1, no=(0,), na=()):
    """1 if any column is `yes`, 0 if all are in `no` (or `na`, a certain no), else NaN."""
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    out = pd.Series(np.nan, index=df.index)
    out[x.isin(list(no) + list(na)).all(axis=1)] = 0
    out[(x == yes).any(axis=1)] = 1
    return out


def women_only(df, s):
    """-1 for men, as NHANES does for the reproductive-health items."""
    return s.where(df.sex == 2, -1)


# --- features ----------------------------------------------------------------------------------

def education(df):
    """DMDEDUC2: 1 < 9th grade, 2 9-11th grade, 3 high school, 4 some college / associate,
    5 college graduate or above. From educ (school level) and graduat (1 graduated, 2 completed,
    3 dropped out, 4 enrolled). Korean middle school ends at grade 9."""
    lvl, grad = num(df.educ, [88, 99]), num(df.graduat, [8, 9])
    done = grad.isin([1, 2])
    out = pd.Series(np.nan, index=df.index)
    out[lvl.isin([1, 2, 3])] = 1                       # 서당 / 무학 / 초등학교
    out[(lvl == 4) & ~done] = 1                        # middle school not finished
    out[(lvl == 4) & done] = 2
    out[(lvl == 5) & ~done] = 2                        # high school not finished
    out[(lvl == 5) & done] = 3
    out[lvl == 6] = 4                                  # 2- / 3-year college
    out[(lvl == 7) & ~done] = 4
    out[(lvl == 7) & done] = 5
    out[lvl == 8] = 5                                  # graduate school
    out[lvl.isin([4, 5, 7]) & grad.isna()] = np.nan
    return out


def marital(df):
    """DMDMARTL: 1 married, 2 widowed, 3 divorced, 4 separated, 5 never married
    (6 living with partner has no KNHANES counterpart)."""
    out = code(df.marri_2, {1: 1, 2: 4, 3: 2, 4: 3})
    return out.where(df.marri_1 != 2, 5)


def income_band(df, year):
    """Monthly household income (ainc, 10,000 KRW) -> USD a year at PPP -> NHANES codes 1-11."""
    usd = num(df.ainc, [999999]) * 1e4 * 12 / PPP[year]
    band = pd.Series(np.searchsorted(INCOME_BANDS, usd, side="right") + 1, index=df.index, dtype=float)
    return band.where(usd.notna())


def weight_year_ago(df):
    """WHD050 (pounds): current weight +/- the reported one-year change (BO1_1 1 none, 2 lost,
    3 gained; BO1_2 / BO1_3 bands 3-6, 6-10, 10+ kg, taken at 4.5, 8 and 12 kg)."""
    mid = {1: 4.5, 2: 8.0, 3: 12.0}
    change = pd.Series(np.nan, index=df.index)
    change[df.BO1_1 == 1] = 0
    change[df.BO1_1 == 2] = code(df.BO1_2, mid)[df.BO1_1 == 2]    # lost: weighed more a year ago
    change[df.BO1_1 == 3] = -code(df.BO1_3, mid)[df.BO1_1 == 3]
    return ((num(df.HE_wt) + change) * LB_PER_KG).round()


def home_smoke(df, year):
    """home_smoke_inside: exposed to other people's smoke inside the home in the past 7 days.
    2014 asks it only if a family member smokes indoors (BS9_1); 2016 asks everyone (BS9_2, 3 = no
    family member smokes indoors)."""
    if year == 2014:
        out = code(df.BS9_2, {1: 1, 2: 0})
        return out.where(df.BS9_1 != 2, 0)
    return code(df.BS9_2, {1: 1, 2: 0, 3: 0})


def smoking(df):
    """smoking_status 0 never (< 100 cigarettes) / 1 former / 2 current, and cigs_per_day."""
    life, now = num(df.BS1_1, [8, 9]), num(df.BS3_1, [9])
    status = pd.Series(np.nan, index=df.index)
    status[life.isin([1, 3])] = 0
    status[(life == 2) & (now == 3)] = 1
    status[(life == 2) & now.isin([1, 2])] = 2
    cigs = num(df.BS3_2, [888, 999]).where(status == 2, 0).where(status.notna())
    return status, cigs


# ALQ121 scale: 0 none in past year, 1 every day, 2 nearly every day, 3 3-4 a week, 4 2 a week,
# 5 once a week, 6 2-3 a month, 7 once a month, 8 7-11 a year, 9 3-6 a year, 10 1-2 a year
ALCOHOL_FREQ = {1: 0, 2: 9, 3: 7, 4: 6, 5: 4, 6: 2}   # BD1_11: never / <1 a month / 1 a month /
                                                     # 2-4 a month / 2-3 a week / 4+ a week
BINGE_FREQ = {1: 0, 2: 9, 3: 7, 4: 5, 5: 2}          # BD2_31: never / <1 a month / 1 a month /
                                                     # once a week / nearly every day
DRINKS = {1: 1, 2: 3, 3: 5, 4: 7, 5: 10}             # BD2_1 glasses 1-2 / 3-4 / 5-6 / 7-9 / 10+, lower
                                                     # end of each band (a Korean glass is smaller)


def drinking(df):
    never = df.BD1 == 1                                # never drank
    none_year = never | (df.BD1_11 == 1)
    freq = code(df.BD1_11, ALCOHOL_FREQ).where(~never, 0)
    per_day = code(df.BD2_1, DRINKS).where(~none_year, 0)
    binge = code(df.BD2_31, BINGE_FREQ).where(~none_year, 0)
    return freq, per_day, binge


def work(df):
    """OCD150 1 working / 3 looking for work / 4 not working; not_working_reason as NHANES
    (0 has a job, 10 looking, else OCQ380); hours_worked (EC_wht_23, weekly hours, 0 if no job)."""
    job, why = num(df.EC1_1, [8, 9]), num(df.EC1_2, [9])
    ocd = pd.Series(np.nan, index=df.index)
    ocd[job == 1] = 1
    ocd[(job == 2) & (why == 5)] = 3
    ocd[(job == 2) & why.isin([1, 2, 3, 4, 6, 7])] = 4
    reason = code(why, {1: 7, 2: 2, 3: 3, 4: 4, 5: 10, 6: 1, 7: 7}).where(job != 1, 0)
    hours = num(df.EC_wht_23, [999]).where(job == 1, 0).where(job.notna())
    hours = hours.mask(hours == 888, 0)
    return ocd, reason, hours


def family_history(df, prefix):
    """MCQ300A / C: any of father, mother, sibling diagnosed (8 = no siblings)."""
    out = any_of(df, [f"{prefix}{i}" for i in (1, 2, 3)], no=(0,), na=(8,))
    return out.map({1: 1, 0: 2})


def sleep(df):
    """sleep_hours from weekday sleep (Total_slp_wk, minutes), whole hours, 1-12 (2016 only)."""
    if "Total_slp_wk" not in df:
        return pd.Series(np.nan, index=df.index)
    h = (num(df.Total_slp_wk, [8888, 9999]) / 60).round()
    return h.clip(1, 12)


DIET = {"DR_TKCAL": "N_EN", "DR_TMOIS": "N_WATER", "DR_TPROT": "N_PROT", "DR_TTFAT": "N_FAT",
        "DR_TSFAT": "N_SFA", "DR_TMFAT": "N_MUFA", "DR_TPFAT": "N_PUFA", "DR_TCARB": "N_CHO",
        "DR_TCALC": "N_CA", "DR_TPHOS": "N_PHOS", "DR_TSODI": "N_NA", "DR_TPOTA": "N_K",
        "DR_TIRON": "N_FE", "DR_TBCAR": "N_CAROT", "DR_TRET": "N_RETIN", "DR_TVB1": "N_B1",
        "DR_TVB2": "N_B2", "DR_TNIAC": "N_NIAC", "DR_TVC": "N_VITC", "DR_TCHOL": "N_CHOL",
        "DR_TFIBE": "N_TDF", "DR_TSUGR": "N_SUGAR", "DR_TMAGN": "N_MG", "DR_TZINC": "N_ZN",
        "DR_TVARA": "N_VA_RAE", "DR_TATOC": "N_VITE", "DR_TFDFE": "N_FOLATE"}
DIRECT = {"RIDAGEYR": "age", "BMXBMI": "HE_BMI", "BMXHT": "HE_ht", "BMXWT": "HE_wt",
          "BMXWAIST": "HE_wc", "BPXSY1": "HE_sbp1", "BPXSY2": "HE_sbp2", "BPXSY3": "HE_sbp3",
          "BPXDI1": "HE_dbp1", "BPXDI2": "HE_dbp2", "BPXDI3": "HE_dbp3",
          "LBXSASSI": "HE_ast", "LBXSATSI": "HE_alt", "LBXSBU": "HE_BUN", "LBXSCR": "HE_crea",
          "LBXSCH": "HE_chol", "LBXSGL": "HE_glu", "LBXSTR": "HE_TG", "LBXSUA": "HE_Uacid",
          "URXUCR": "HE_UCREA", "LBXTC": "HE_chol", "LBDHDD": "HE_HDL_st2", "LBXGH": "HE_HbA1c",
          "LBXCOT": "HE_Ucot", "LBXWBCSI": "HE_WBC", "LBXRBCSI": "HE_RBC", "LBXHGB": "HE_HB",
          "LBXHCT": "HE_HCT", "LBXMCVSI": "HE_MCV", "LBXMCHSI": "HE_MCH", "LBXMC": "HE_MCHC",
          "LBXPLTSI": "HE_Bplt", **DIET}


def features(df, year):
    """One row per person, the NHANES feature columns that have a KNHANES counterpart."""
    f = {}
    for col, src in DIRECT.items():
        f[col] = num(df[src]) if src in df else np.nan
    f["RIAGENDR"] = code(df.sex, {1: 1, 2: 2})
    f["DMDEDUC2"] = education(df)
    f["DMDMARTL"] = marital(df)
    size = num(df.cfam, [9])
    f["DMDHHSIZ"] = size
    f["DMDFMSIZ"] = size
    f["fam_income_cat"] = f["hh_income_cat"] = income_band(df, year)
    f["HSD010"] = f["HUQ010"] = code(df.D_1_1, {1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
    f["HUQ071"] = yes_no(df.MH1_yr) if "MH1_yr" in df else np.nan
    f["HIQ031A"] = code(df.npins, {1: 1, 2: 0})
    f["htn_dx"] = code(df.DI1_dg, {0: 0, 1: 1})
    f["htn_med"] = code(df.DI1_2, {1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 8: 0})
    f["chol_dx"] = code(df.DI2_dg, {0: 0, 1: 1})
    f["chol_med"] = code(df.DI2_2, {1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 8: 0})
    f["diabetes_status"] = code(df.DE1_dg, {0: 0, 1: 2})
    f["insulin"] = flag(df.DE1_31)
    f["diabetes_pills"] = flag(df.DE1_32)
    f["MCQ010"] = dx(df.DJ4_dg)
    f["asthma_current"] = flag(df.DJ4_pr)
    f["MCQ160A"] = dx(df.DM1_dg)
    arth = pd.Series(np.nan, index=df.index)
    arth[df.DM1_dg == 0] = 0
    arth[df.DM1_dg == 1] = 3
    arth[(df.DM1_dg == 1) & (df.DM3_dg == 1)] = 2
    arth[(df.DM1_dg == 1) & (df.DM2_dg == 1)] = 1
    f["arthritis_type"] = arth
    f["MCQ160C"] = dx(df.DI4_dg)
    f["MCQ160D"] = dx(df.DI6_dg)
    f["MCQ160E"] = dx(df.DI5_dg)
    f["MCQ160F"] = dx(df.DI3_dg)
    f["MCQ160G"] = f["MCQ160K"] = dx(df.HE_PFTdr)        # COPD, asked at the lung test (40+)
    liver = ["DK4", "DK8", "DK9"]                           # cirrhosis, hepatitis B, hepatitis C
    f["MCQ160L"] = any_of(df, [f"{d}_dg" for d in liver]).map({1: 1, 0: 2})
    f["liver_current"] = any_of(df, [f"{d}_pr" for d in liver], na=(8,))
    f["MCQ160M"] = dx(df.DE2_dg)
    f["thyroid_current"] = flag(df.DE2_pr)
    cancer = [c for c in df.columns if c.startswith("DC") and c.endswith("_dg")]
    f["MCQ220"] = any_of(df, cancer).map({1: 1, 0: 2})
    f["MCQ300A"] = family_history(df, "HE_IHDfh")
    f["MCQ300C"] = family_history(df, "HE_DMfh")
    f["KIQ022"] = dx(df.DN1_dg)
    f["smoking_status"], f["cigs_per_day"] = smoking(df)
    f["home_smoke_inside"] = home_smoke(df, year)
    f["alcohol_freq"], f["drinks_per_day"], f["binge_freq"] = drinking(df)
    f["sleep_hours"] = sleep(df)
    f["OCD150"], f["not_working_reason"], f["hours_worked"] = work(df)
    f["HOQ065"] = code(df.house, {1: 2, 2: 1, 3: 1})
    f["WHQ030"] = code(df.BO1, {1: 2, 2: 2, 3: 3, 4: 1, 5: 1})
    f["WHQ070"] = code(df.BO2_1, {1: 1, 2: 2, 3: 2, 4: 2})
    f["WHD050"] = weight_year_ago(df)
    f["WHD080A"] = flag(df.BO3_03)                          # ate less
    f["WHD080D"] = flag(df.BO3_01)                          # exercised
    f["WHD080E"] = flag(df.BO3_14)                          # skipped meals
    f["WHD080I"] = flag(df.BO3_04)                          # prescription diet pills
    f["WHD080J"] = any_of(df, ["BO3_05", "BO3_07"], na=(8,))  # other diet pills / supplements
    f["RHQ010"] = women_only(df, num(df.LW_ms_a, [888, 999]))
    f["RHQ031"] = women_only(df, code(df.LW_ms, {2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}))
    f["RHQ131"] = women_only(df, yes_no(df.LW_pr))
    f["RHQ160"] = women_only(df, num(df.LW_pr_1, [99]).mask(df.LW_pr_1 == 88, 0))
    f["RHQ420"] = women_only(df, yes_no(df.LW_oc))
    f["PFQ049"] = yes_no(df.LQ4_00)
    f["BPXPULS"] = code(df.HE_rPLS, {1: 1, 2: 2})
    f["BPXPLS"] = (num(df.HE_PLS) * 4).fillna(num(df.HE_mPLS))  # 15-s count x 4; 60-s count if irregular
    f["DR_300"] = code(df.N_DUSUAL, {1: 1, 2: 2, 3: 3})
    f["DRQSDIET"] = yes_no(df.N_DIET)
    f["DR_320Z"] = num(df.N_WAT_C, [999]) * 200                   # cups of 200 mL -> g
    return pd.DataFrame(f, index=df.index)


def load(years=YEARS):
    """KNHANES population of `years` with `cycle`, `PHQ9`, `depression`, the survey weight
    `wt_itvex` and the NHANES-defined features."""
    out = []
    for y in years:
        pop = population(read_year(y))
        phq = pop[PHQ].sum(axis=1)
        head = pd.DataFrame({"id": pop["id"].str.decode("latin1"),
                             "cycle": f"KNHANES {y}", "wt_itvex": pop.wt_itvex,
                             "PHQ9": phq, "depression": (phq >= 10).astype(int)})
        out.append(pd.concat([head, features(pop, y)], axis=1))
    return pd.concat(out, ignore_index=True)

