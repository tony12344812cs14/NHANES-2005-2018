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

Every step is a notebook, saved with its tables and figures so it can be read on GitHub without
running anything. `run_pipeline.sh` executes them in order, in place (`jupyter execute --inplace`);
each writes to `outputs/`. To run one by hand, open it and Run All (from the repository root or
its own folder, with the same `NHANES_DATA` / `NHANES_WORK`).

| Step | Does |
|---|---|
| `nhanes/download.py` | downloads the 322 data files and the Multum drug-category file |
| `pipeline/labels.ipynb` | reads every variable label from the file headers |
| `pipeline/step1_nonfeature.ipynb` | drops ids, weights, design variables, lab comment codes, unit duplicates, process variables |
| `pipeline/step2_leakage.ipynb` | drops the PHQ-9 items and concurrent mental-health measures |
| `pipeline/step3_cycle_overlap.ipynb` | which features exist in all seven cycles |
| `pipeline/step4_rename.ipynb` | merges variables that were only renamed (checked against the codebooks) |
| `pipeline/step5_missing.ipynb` | missing rates in the population, per cycle |
| `pipeline/step6_gates.ipynb` | finds the gate question of each skip-pattern follow-up from the data |
| `pipeline/fetch_codebooks.ipynb` | downloads the CDC codebooks and extracts each variable's code table |
| `pipeline/build_features.ipynb` | builds the table: skip patterns, checkboxes, derived and recoded features, diet days, missing threshold |

Code used by more than one notebook lives in `nhanes/`:

| Module | Contents |
|---|---|
| `paths.py`, `io.py`, `download.py` | where data and outputs live, reading NHANES files, the download |
| `cohort.py` | the analysis population and cycle names |
| `harmonize.py` | items recoded across cycles, prescription counts, PFOA / PFOS totals, lab calibrations |
| `loco.py` | leave-one-cycle-out design matrix, elastic net, XGBoost, CatBoost, tuning, metrics |
| `knhanes.py` | KNHANES 2014 / 2016 in the NHANES feature definitions (needs the KNHANES files in `KNHANES_DATA`) |
| `paper1.py`, `vae.py` | the 86 variables of Dang et al. (2024) and the denoising VAE (modeling steps 3–8) |
| `plots.py` | result figures of the modeling notebooks |

## Modeling (`modeling/`)

One notebook per step, results and figures saved; the record of every step (in Chinese) is
`modeling/README.md`. Each notebook has `RUN = False` at the top: it then only reads the saved
results in `$NHANES_WORK/modeling/` and draws them. `RUN = True` retrains (hours for most steps;
`QUICK = True` runs one fold to check the setup). Retraining needs more packages than the
pipeline: catboost, torch, `tabm==0.0.3`, `rtdl_num_embeddings==0.0.12`; step 8b needs
`modeling/requirements-foundation.txt`.

`modeling/step0_cycle_check.ipynb` is a check to run before modelling: it measures how well the
features alone can tell the cycles apart, to find differences between cycles that are left.

## Data analysis (`data_analysis/`)

Descriptive notebooks, saved with their figures so they can be read on GitHub without running
anything. To rerun, set `NHANES_DATA` / `NHANES_WORK` as for the pipeline (03 also needs the KNHANES files, see `nhanes/knhanes.py`), then Run All; tables
and PNGs go to `$NHANES_WORK/data_analysis/`.

| Notebook | Contents |
|---|---|
| `01_cohort_and_class_distribution.ipynb` | cohort selection per cycle from the raw DEMO / DPQ files (checked against the feature table), class balance overall and per cycle |
| `02_feature_eda.ipynb` | feature types, missing and not-applicable rates per cycle, cross-cycle distributions, feature–target rates, correlations |
| `03_knhanes_vs_nhanes.ipynb` | KNHANES 2014 / 2016 (Korea) against NHANES: outcome, distribution of each shared feature, how well the two can be told apart, and whether each feature relates to depression the same way |

## Decisions

Every decision, with the evidence behind it, is in `docs/` (in Chinese):

- `docs/skip_pattern_plan.txt`: skip patterns, gates checked per cycle, encoding, missing threshold,
  diet days, medications.
- `docs/pending_features.txt`: variables renamed between cycles that are not pure renames, and
  how each was recoded or why it was dropped.
- `docs/external_data.md`: external validation, which datasets were considered (KNHANES, NHIS),
  and which features have a counterpart in each.

A few that matter when you use the table:

- **Sleep hours:** from 2015 the question asks about weekdays and the mean is about one hour higher;
  `sleep_hours` is kept as measured and the change is noted in the dictionary.
- **Lab instrument changes:** 2005–06 serum creatinine and the 2017–18 biochemistry panel are put on one
  scale with the CDC calibration equations. Analytes without a CDC equation are left as measured.
- **Medications:** `rx_count` leaves out antidepressants, antipsychotics and anxiolytics / sedatives /
  hypnotics, since they mean the outcome is already being treated.
- **Questions whose threshold changed:** `binge_freq` and `heavy_drinking_ever` use 5+ drinks before
  2011 and 4+ for women from 2011; `military_service` asks about active duty only from 2011.
- **2017–March 2020 (pre-pandemic) files** are downloaded for the cross-cycle comparison but not used
  in the table: they renumber participants and include the 2017–18 sample.

## Data source

National Center for Health Statistics, National Health and Nutrition Examination Survey,
2005–2018, https://wwwn.cdc.gov/nchs/nhanes/. Drug categories: NHANES RXQ_DRUG (Multum Lexicon).
