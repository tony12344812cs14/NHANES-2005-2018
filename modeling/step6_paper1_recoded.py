"""Step 6: the 86 features of Dang et al. (2024), re-encoded without adding information.

Question: with the same variables, can better encoding (plus CatBoost and an ensemble) beat the paper?
Variants fixed before running; all are reported.

Feature tables
  raw      the paper's table as it is (step 3)
  recoded  the same variables, re-encoded:
           1. numerical fields the paper filled with 0 (mostly skipped questions: no diagnosis, never
              used, ...) are -1 = not applicable / unknown instead, so "0" no longer mixes "none" with
              "not asked"; weight, height and BMI keep NaN instead of the median
           2. the 13 ages at onset become years since onset (age - onset age, >= 0), -1 without the
              condition
           3. RXDDRUG (name of the first non-antidepressant drug, 755 names) becomes its Multum level-1
              drug class (18 classes), "none" without a prescription
Models: logreg and xgboost as in step 1, catboost as in step 2, and ensemble = mean of the three
probabilities.
Evaluations
  loco     our 33,498 people, leave-one-cycle-out, inner LOCO over the 6 training cycles
  kfold10  the paper's 36,259 people and split (StratifiedKFold(10, shuffle, seed 42)); tuning by an inner
           3-fold stratified split of the training folds; survey weights are not available for the
           extra people, so the weighted AUROC is unweighted there

Usage: python modeling/step6_paper1_recoded.py PAPER1_REPRO_DIR PAPER1_CSV [--eval loco|kfold10]
           [--table raw|recoded] [--drug-file RXQ_DRUG.xpt] [--vae] [--quick]
       --vae: also add the 32 latent means of step 5's denoising VAE, trained per split on the training
       rows only (output directory gets a _vae suffix)
       PAPER1_REPRO_DIR holds build_dataset.py of our reproduction; PAPER1_CSV is its FullData_rm_dl_bt.csv.
Output: <NHANES_WORK>/modeling/step6_paper1_recoded/<eval>_<table>/ (metrics.csv, predictions.csv.gz,
        chosen_params.json, report.txt); the recoded table in .../data/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import DRUG_FILE, RESULTS, WORK  # noqa: E402
from step1_loco import (LOGREG_GRID, NOT_FEATURES, SEED, XGB_GRID, fit_logreg, fit_xgb, grid,  # noqa: E402
                        logreg_matrix, metrics, tune)
from step2_more_models import CAT_GRID, run_catboost  # noqa: E402
from step3_paper1_features import paper1_table  # noqa: E402

OUT = RESULTS / "step6_paper1_recoded"
ONSET = ["asthma_onset", "arthritis_onset", "heart_failure_onset", "heart_disease_onset", "angina_onset",
         "heart_attack_onset", "stroke_onset", "emphysema_onset", "bronchitis_onset", "liver_condition_onset",
         "thyroid_problem_onset", "cancer_onset"]
BODY = ["weight", "height", "BMI"]
MODELS = ["logreg", "xgboost", "catboost"]


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


def frames(table_csv, how, quick):
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
    y = pd.read_csv(PAPER_CSV, usecols=["depression"]).depression.eq("Depressed").astype(int)
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


PAPER_CSV = None


def add_vae(f, dic, device, quick):
    """Step 5's denoising VAE, trained on the training rows of this split without labels; adds its
    32 latent means as continuous features."""
    from step5_vae import VAE, encode, train_vae
    feats = [c for c in f if c not in NOT_FEATURES]
    is_tr = (f.cycle != "TEST").to_numpy()
    A, B = logreg_matrix(f.loc[is_tr, feats], f.loc[~is_tr, feats], dic)
    vae, epochs, _ = train_vae(A, device, quick)
    z = np.zeros((len(f), VAE["latent"]), dtype=np.float32)
    z[is_tr], z[~is_tr] = encode(vae, A, device), encode(vae, B, device)
    zcols = [f"z{i:02d}" for i in range(VAE["latent"])]
    return (pd.concat([f, pd.DataFrame(z, columns=zcols, index=f.index)], axis=1),
            {**dic, **{c: "continuous" for c in zcols}}, epochs)


def run(how, table, table_csv, quick, vae=False):
    out = OUT / f"{how}_{table}{'_vae' if vae else ''}"
    if vae:
        import torch
        from step2_more_models import free_gpu
        gpu = free_gpu()
        device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
        print(f"VAE on {device}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    grids = {"logreg": grid(LOGREG_GRID), "xgboost": grid(XGB_GRID)}
    if quick:
        grids = {k: v[:1] for k, v in grids.items()}
        CAT_GRID[:] = CAT_GRID[:1]
    preds, rows, chosen = [], [], {}
    for split, f, inner, dic in frames(table_csv, how, quick):
        if vae:
            t0 = time.time()
            f, dic, epochs = add_vae(f, dic, device, quick)
            print(f"{how} {table} {split} VAE: {epochs} epochs, {(time.time() - t0) / 60:.1f} min", flush=True)
        tr, te = f[f.cycle != "TEST"], f[f.cycle == "TEST"]
        Xtr, Xte = tr.drop(columns=NOT_FEATURES), te.drop(columns=NOT_FEATURES)
        probs = {}
        for model in MODELS:
            t0 = time.time()
            if model == "catboost":
                prob, params = run_catboost(f, inner, "TEST", dic)
            else:
                params, inner_auc = tune(model, f, inner, dic, grids[model])
                if model == "logreg":
                    A, B = logreg_matrix(Xtr, Xte, dic)
                    prob = fit_logreg(A, tr.depression, params).predict_proba(B)[:, 1]
                else:
                    fit_p = {k: v for k, v in params.items() if k != "n_estimators"}
                    prob = fit_xgb(Xtr, tr.depression, None, None, fit_p,
                                   n_estimators=params["n_estimators"]).predict_proba(Xte)[:, 1]
                params = {**params, "inner_auroc": round(inner_auc, 4)}
            probs[model] = prob
            chosen[f"{model}|{split}"] = params
            rows.append({"model": model, "split": split, **metrics(te.depression, prob, te.WTMEC2YR),
                         "minutes": round((time.time() - t0) / 60, 1)})
            print(f"{how} {table} {split} {model}: AUROC {rows[-1]['auroc']:.3f}  {params}  {rows[-1]['minutes']} min",
                  flush=True)
        probs["ensemble"] = np.mean([probs[m] for m in MODELS], axis=0)
        rows.append({"model": "ensemble", "split": split, **metrics(te.depression, probs["ensemble"], te.WTMEC2YR),
                     "minutes": 0.0})
        print(f"{how} {table} {split} ensemble: AUROC {rows[-1]['auroc']:.3f}", flush=True)
        preds += [pd.DataFrame({"SEQN": te.SEQN, "split": split, "model": m, "depression": te.depression,
                                "prob": p}) for m, p in probs.items()]
    met = pd.DataFrame(rows)
    pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv.gz", index=False)
    met.to_csv(out / "metrics.csv", index=False)
    (out / "chosen_params.json").write_text(json.dumps(chosen, indent=1, default=str))
    summary = met.groupby("model", sort=False)[["auroc", "auroc_weighted", "auprc", "brier"]].agg(["mean", "std"]).round(3)
    lines = [f"features of Dang et al. (2024), table '{table}'{' + VAE latent (step 5)' if vae else ''}, "
             f"evaluation '{how}'", "",
             "mean (sd) over test splits:", summary.to_string(), "",
             "AUROC per test split:", met.pivot(index="split", columns="model", values="auroc").round(3).to_string()]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    global PAPER_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("repro_dir", type=Path)
    ap.add_argument("paper_csv", type=Path)
    ap.add_argument("--eval", choices=["loco", "kfold10"])
    ap.add_argument("--table", choices=["raw", "recoded"])
    ap.add_argument("--drug-file", type=Path, default=DRUG_FILE, help="Multum RXQ_DRUG.xpt")
    ap.add_argument("--vae", action="store_true", help="add step 5's VAE latent features")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    PAPER_CSV = args.paper_csv
    rec = OUT / "data" / "FullData_recoded.csv"
    if (args.table or "recoded") == "recoded" and not rec.exists():
        recoded_table(args.repro_dir, args.paper_csv, rec, args.drug_file)
    tables = {"raw": args.paper_csv, "recoded": rec}
    for how in [args.eval] if args.eval else ["loco", "kfold10"]:
        for table in [args.table] if args.table else ["raw", "recoded"]:
            run(how, table, tables[table], args.quick, args.vae)


if __name__ == "__main__":
    main()
