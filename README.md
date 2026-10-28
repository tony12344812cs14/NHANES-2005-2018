# NHANES depression features, 2005–2018

Builds one person-level feature table for predicting depression (PHQ-9 ≥ 10) from seven NHANES
cycles, 2005–06 to 2017–18, with every feature defined the same way in every cycle.

NHANES renames variables, changes answer codes and reorders questions between cycles. Pooled
naively, a model can learn *which survey year* a row came from instead of anything about the
person. This pipeline checks each feature across cycles against the CDC codebooks and the data,
merges what is the same, recodes what changed, and drops what cannot be aligned.

**Output:** 33,498 adults × 317 features, no feature above 20% missing in any cycle.

**Project notes:** [Notion page](https://app.notion.com/p/NHANES-2005-2018-407d07eb533181eeb0c0cb18503ffc18)

## Quick start

Needs Python 3.10–3.13 (check with `python3 --version`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_pipeline.sh          # downloads ~0.6 GB from CDC, then builds everything (~10 min after the download)
```

If your `python3` is older, conda can provide one:
`conda create -n nhanes python=3.12 && conda activate nhanes`, then the last two lines above.

Everything is public: the script downloads the NHANES files from the CDC website, so no data
are stored in this repository. Set `NHANES_DATA` and `NHANES_WORK` to put data and outputs
somewhere other than `./data` and `./outputs`; set `THREADS` to change the CPU threads (default 4).
If the data are already downloaded, `./run_pipeline.sh --skip-download`.

## What you get (`outputs/features/`)

| File | Contents |
|---|---|
| `features_2005_2018.csv.gz` | `SEQN`, `cycle`, survey design and weights (`SDMVPSU`, `SDMVSTRA`, `WTINT2YR`, `WTMEC2YR`), targets `PHQ9` and `depression`, then the 317 features |
| `feature_dictionary.csv` | one row per feature: how it was built, source variables, rule, type, missing rate overall and per cycle |
| `subsample_labs_2005_2018.csv.gz` | labs measured on a random subsample only (fasting glucose, lipids, PFAS, PFOA / PFOS totals); join on `SEQN` |
| `build_report.txt` | per-cycle checks of every converted feature |

**Population:** age ≥ 20, not pregnant, all nine PHQ-9 items answered.

**Encoding** of every feature:

| Value | Meaning |
|---|---|
| the answer, or a logically certain 0 | e.g. a never-smoker smokes 0 cigarettes a day; an unticked checkbox is 0 |
| `-1` | not applicable: the questionnaire does not ask this age or sex (e.g. drug use above 59, reproductive history for men) |
| empty | truly missing: eligible but not answered, refused or don't know (codes taken from the CDC codebooks) |

Tree models can use the table as it is. For linear models, one-hot the categorical features
(`type` in the dictionary) and split numeric features with `-1` into the value and an indicator.

## Pipeline

`run_pipeline.sh` runs these in order; each writes to `outputs/`.

| Script | Does |
|---|---|
| `nhanes/download.py` | downloads the 322 data files and the Multum drug-category file |
| `pipeline/labels.py` | reads every variable label from the file headers |
| `pipeline/step1_nonfeature.py` | drops ids, weights, design variables, lab comment codes, unit duplicates, process variables |
| `pipeline/step2_leakage.py` | drops the PHQ-9 items and concurrent mental-health measures |
| `pipeline/step3_cycle_overlap.py` | which features exist in all seven cycles |
| `pipeline/step4_rename.py` | merges variables that were only renamed (checked against the codebooks) |
| `pipeline/step5_missing.py` | missing rates in the population, per cycle |
| `pipeline/step6_gates.py` | finds the gate question of each skip-pattern follow-up from the data |
| `pipeline/fetch_codebooks.py` | downloads the CDC codebooks and extracts each variable's code table |
| `pipeline/build_features.py` (+ `harmonize.py`) | builds the table: skip patterns, checkboxes, derived and recoded features, diet days, missing threshold |

`modeling/step0_cycle_check.py` is a check to run before modelling: it measures how well the
features alone can tell the cycles apart, to find differences between cycles that are left.

## Decisions

Every decision, with the evidence behind it, is in `docs/` (in Chinese):

- `docs/skip_pattern_plan.txt`: skip patterns, gates checked per cycle, encoding, missing threshold,
  diet days, medications.
- `docs/pending_features.txt`: variables renamed between cycles that are not pure renames, and
  how each was recoded or why it was dropped.

A few that matter when you use the table:

- **Sleep hours:** from 2015 the question asks about weekdays and the mean is about one hour higher,
  so `sleep_quintile` ranks sleep within 2005–14 and within 2015–18 (1 shortest … 5 longest).
- **Lab instrument changes:** 2005–06 serum creatinine and the 2017–18 biochemistry panel are put on one
  scale with the CDC calibration equations; analytes CDC flags as instrument-affected but gives no equation
  for (CBC: RDW, WBC, HCT, MCH, MCHC; biochemistry: Na, Cl, K, bicarbonate, osmolality) are z-scored within
  each cycle. The rule of each one is in `feature_dictionary.csv`.
- **Medications:** `rx_count` leaves out antidepressants, antipsychotics and anxiolytics / sedatives /
  hypnotics, since they mean the outcome is already being treated.
- **Questions whose threshold changed:** `binge_freq` and `heavy_drinking_ever` use 5+ drinks before
  2011 and 4+ for women from 2011; `military_service` asks about active duty only from 2011.
- **2017–March 2020 (pre-pandemic) files** are downloaded for the cross-cycle comparison but not used
  in the table: they renumber participants and include the 2017–18 sample.

## Data source

National Center for Health Statistics, National Health and Nutrition Examination Survey,
2005–2018, https://wwwn.cdc.gov/nchs/nhanes/. Drug categories: NHANES RXQ_DRUG (Multum Lexicon).
