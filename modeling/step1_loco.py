"""Step 1: baseline and main model, leave-one-cycle-out (LOCO).

Each of the 7 cycles is the test set once; the other 6 train the model. Hyper-parameters are chosen
by an inner LOCO over the 6 training cycles (mean inner AUROC); preprocessing is fit on training rows
only. Outcome: depression = PHQ-9 >= 10.

Models
  logreg   elastic-net logistic regression. Categorical features one-hot (-1 and missing are their
           own levels); numeric features: -1 -> 'not applicable' indicator + NaN, median imputation,
           standardised.
  xgboost  gradient-boosted trees on the table as it is (NaN and -1 handled by the trees); number
           of trees by early stopping in the inner folds.
Both use class weights for the 8.9% prevalence.

Metrics (primary: AUROC, the mean over the 7 test cycles, each cycle weighted equally)
  per test cycle: AUROC, survey-weighted AUROC (WTMEC2YR), AUPRC, Brier score; plus the AUROC of
  all out-of-cycle predictions pooled.

Usage: python modeling/step1_loco.py [--quick]   (--quick: one outer fold, smallest grids)
Output: <NHANES_WORK>/modeling/step1/ predictions.csv.gz, metrics.csv, chosen_params.json, report.txt
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import RESULTS, THREADS, WORK  # noqa: E402

FEAT = WORK / "features"
OUT = RESULTS / "step1"
NOT_FEATURES = ["SEQN", "cycle", "SDMVPSU", "SDMVSTRA", "WTINT2YR", "WTMEC2YR", "PHQ9", "depression"]
SEED = 42
LOGREG_GRID = {"C": [0.01, 0.1, 1.0], "l1_ratio": [0.0, 0.5, 1.0]}
XGB_GRID = {"max_depth": [3, 5], "min_child_weight": [1, 10]}
XGB_FIXED = dict(learning_rate=0.05, n_estimators=2000, subsample=0.8, colsample_bytree=0.5,
                 tree_method="hist", eval_metric="auc", early_stopping_rounds=100,
                 n_jobs=THREADS, random_state=SEED)


def logreg_matrix(train, test, dic):
    """Design matrices for logistic regression, fitted on the training rows."""
    cat = [c for c in train if dic.get(c) == "categorical"]
    num = [c for c in train if c not in cat]
    na_cols = [c for c in num if (train[c] == -1).any()]

    def split_na(df):
        df = df.copy()
        for c in na_cols:
            df[f"{c}__na"] = (df[c] == -1).astype(float)
            df[c] = df[c].mask(df[c] == -1)
        df[cat] = df[cat].astype("float").fillna(-99).astype(int).astype(str)  # missing = own level
        return df

    tr, te = split_na(train), split_na(test)
    nums = num + [f"{c}__na" for c in na_cols]
    ct = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat),
    ])
    return ct.fit_transform(tr), ct.transform(te)


def fit_logreg(Xtr, ytr, params):
    m = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", max_iter=300,
                           tol=1e-3, random_state=SEED, **params)
    return m.fit(Xtr, ytr)


def fit_xgb(Xtr, ytr, Xva, yva, params, n_estimators=None):
    w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    kw = dict(XGB_FIXED, scale_pos_weight=w, **params)
    if n_estimators:  # final fit: no validation set, fixed number of trees
        kw.update(n_estimators=n_estimators, early_stopping_rounds=None)
        return xgb.XGBClassifier(**kw).fit(Xtr, ytr, verbose=False)
    return xgb.XGBClassifier(**kw).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


def grid(g):
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def _logreg_auc(Xtr, ytr, Xva, yva, p):
    return roc_auc_score(yva, fit_logreg(Xtr, ytr, p).predict_proba(Xva)[:, 1])


def tune(model, data, train_cycles, dic, grid_):
    """Inner LOCO over the training cycles; returns the best params (and #trees for xgboost).
    The saga solver is single-threaded, so logistic-regression fits run THREADS at a time."""
    scores = {i: [] for i in range(len(grid_))}
    trees = {i: [] for i in range(len(grid_))}
    jobs = []
    for val_c in train_cycles:
        tr = data[data.cycle.isin(train_cycles) & (data.cycle != val_c)]
        va = data[data.cycle == val_c]
        Xtr, Xva = tr.drop(columns=NOT_FEATURES), va.drop(columns=NOT_FEATURES)
        if model == "logreg":
            Xtr, Xva = logreg_matrix(Xtr, Xva, dic)
            jobs += [(i, delayed(_logreg_auc)(Xtr, tr.depression, Xva, va.depression, p)) for i, p in enumerate(grid_)]
            continue
        for i, p in enumerate(grid_):
            m = fit_xgb(Xtr, tr.depression, Xva, va.depression, p)
            prob = m.predict_proba(Xva, iteration_range=(0, m.best_iteration + 1))[:, 1]
            trees[i].append(m.best_iteration + 1)
            scores[i].append(roc_auc_score(va.depression, prob))
    if jobs:
        for (i, _), auc in zip(jobs, Parallel(n_jobs=THREADS)(j for _, j in jobs)):
            scores[i].append(auc)
    best = max(scores, key=lambda i: np.mean(scores[i]))
    out = dict(grid_[best])
    if model == "xgboost":
        out["n_estimators"] = int(np.median(trees[best]))
    return out, float(np.mean(scores[best]))


def metrics(y, p, w):
    return {"auroc": roc_auc_score(y, p), "auroc_weighted": roc_auc_score(y, p, sample_weight=w),
            "auprc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "n": len(y), "prevalence": float(np.mean(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FEAT / "features_2005_2018.csv.gz")
    dic = pd.read_csv(FEAT / "feature_dictionary.csv").set_index("column").type.to_dict()
    cycles = sorted(data.cycle.unique())
    outer = cycles[:1] if args.quick else cycles
    grids = {"logreg": grid(LOGREG_GRID), "xgboost": grid(XGB_GRID)}
    if args.quick:
        grids = {k: v[:1] for k, v in grids.items()}

    preds, rows, chosen = [], [], {}
    for test_c in outer:
        train_cycles = [c for c in cycles if c != test_c]
        tr, te = data[data.cycle != test_c], data[data.cycle == test_c]
        Xtr, Xte = tr.drop(columns=NOT_FEATURES), te.drop(columns=NOT_FEATURES)
        for model in ["logreg", "xgboost"]:
            t0 = time.time()
            params, inner_auc = tune(model, data, train_cycles, dic, grids[model])
            if model == "logreg":
                A, B = logreg_matrix(Xtr, Xte, dic)
                prob = fit_logreg(A, tr.depression, params).predict_proba(B)[:, 1]
            else:
                fit_p = {k: v for k, v in params.items() if k != "n_estimators"}
                m = fit_xgb(Xtr, tr.depression, None, None, fit_p, n_estimators=params["n_estimators"])
                prob = m.predict_proba(Xte)[:, 1]
            chosen[f"{model}|{test_c}"] = {**params, "inner_auroc": round(inner_auc, 4)}
            preds.append(pd.DataFrame({"SEQN": te.SEQN, "cycle": test_c, "model": model,
                                       "depression": te.depression, "weight": te.WTMEC2YR, "prob": prob}))
            rows.append({"model": model, "test_cycle": test_c, **metrics(te.depression, prob, te.WTMEC2YR),
                         "minutes": round((time.time() - t0) / 60, 1)})
            print(f"{test_c} {model}: AUROC {rows[-1]['auroc']:.3f}  {params}  {rows[-1]['minutes']} min", flush=True)

    pred = pd.concat(preds, ignore_index=True)
    met = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv.gz", index=False)
    met.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "chosen_params.json").write_text(json.dumps(chosen, indent=1))
    summary = met.groupby("model")[["auroc", "auroc_weighted", "auprc", "brier"]].agg(["mean", "std"]).round(3)
    pooled = pred.groupby("model").apply(lambda g: roc_auc_score(g.depression, g.prob)).round(3)
    lines = ["leave-one-cycle-out, outcome PHQ-9 >= 10", "",
             "mean (sd) over test cycles, each cycle weighted equally:", summary.to_string(), "",
             "AUROC of all out-of-cycle predictions pooled:", pooled.to_string(), "",
             "per test cycle:", met.round(3).to_string(index=False)]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
