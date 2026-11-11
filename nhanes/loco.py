"""Leave-one-cycle-out (LOCO) modelling shared by the modeling notebooks.

Moved unchanged from modeling/step1_loco.py (design matrix, elastic net, XGBoost, inner-LOCO tuning,
metrics, run_loco) and modeling/step2_more_models.py (CatBoost, free_gpu), because later steps reuse them.

Outcome: depression = PHQ-9 >= 10. Each of the 7 cycles is the test set once; hyper-parameters are chosen
by an inner LOCO over the 6 training cycles (mean inner AUROC); preprocessing is fit on training rows only.
"""
import itertools
import json
import subprocess
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nhanes.paths import THREADS, WORK

FEAT = WORK / "features"
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


def run_loco(data, dic, out, quick=False, title="leave-one-cycle-out, outcome PHQ-9 >= 10",
             models=("logreg", "xgboost")):
    """LOCO for logreg and xgboost on `data` (NOT_FEATURES columns + features); `dic` maps feature -> type."""
    out.mkdir(parents=True, exist_ok=True)
    cycles = sorted(data.cycle.unique())
    outer = cycles[:1] if quick else cycles
    grids = {"logreg": grid(LOGREG_GRID), "xgboost": grid(XGB_GRID)}
    if quick:
        grids = {k: v[:1] for k, v in grids.items()}

    preds, rows, chosen = [], [], {}
    for test_c in outer:
        train_cycles = [c for c in cycles if c != test_c]
        tr, te = data[data.cycle != test_c], data[data.cycle == test_c]
        Xtr, Xte = tr.drop(columns=NOT_FEATURES), te.drop(columns=NOT_FEATURES)
        for model in models:
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
    pred.to_csv(out / "predictions.csv.gz", index=False)
    met.to_csv(out / "metrics.csv", index=False)
    (out / "chosen_params.json").write_text(json.dumps(chosen, indent=1))
    summary = met.groupby("model")[["auroc", "auroc_weighted", "auprc", "brier"]].agg(["mean", "std"]).round(3)
    pooled = pred.groupby("model").apply(lambda g: roc_auc_score(g.depression, g.prob)).round(3)
    lines = [title, "",
             "mean (sd) over test cycles, each cycle weighted equally:", summary.to_string(), "",
             "AUROC of all out-of-cycle predictions pooled:", pooled.to_string(), "",
             "per test cycle:", met.round(3).to_string(index=False)]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


CAT_GRID = [4, 6]
CAT_FIXED = dict(learning_rate=0.05, iterations=3000, od_type="Iter", od_wait=100, eval_metric="AUC",
                 auto_class_weights="Balanced", thread_count=THREADS, random_seed=SEED, verbose=False,
                 allow_writing_files=False)


def cat_frame(X, cat):
    X = X.copy()
    X[cat] = X[cat].astype("float").fillna(-99).astype(int).astype(str)  # missing = own level
    return X


def run_catboost(data, train_cycles, test_c, dic):
    feats = [c for c in data if c not in NOT_FEATURES]
    cat = [c for c in feats if dic.get(c) == "categorical"]
    scores, iters = {d: [] for d in CAT_GRID}, {d: [] for d in CAT_GRID}
    for val_c in train_cycles:
        tr, va = data[data.cycle.isin(train_cycles) & (data.cycle != val_c)], data[data.cycle == val_c]
        Xtr, Xva = cat_frame(tr[feats], cat), cat_frame(va[feats], cat)
        for depth in CAT_GRID:
            m = CatBoostClassifier(depth=depth, cat_features=cat, **CAT_FIXED)
            m.fit(Xtr, tr.depression, eval_set=(Xva, va.depression), use_best_model=True)
            scores[depth].append(roc_auc_score(va.depression, m.predict_proba(Xva)[:, 1]))
            iters[depth].append(m.get_best_iteration() + 1)
    depth = max(scores, key=lambda d: np.mean(scores[d]))
    n = int(np.median(iters[depth]))
    tr, te = data[data.cycle != test_c], data[data.cycle == test_c]
    fixed = {k: v for k, v in CAT_FIXED.items() if k not in ("iterations", "od_type", "od_wait")}
    m = CatBoostClassifier(depth=depth, iterations=n, cat_features=cat, **fixed)
    m.fit(cat_frame(tr[feats], cat), tr.depression)
    prob = m.predict_proba(cat_frame(te[feats], cat))[:, 1]
    return prob, {"depth": depth, "iterations": n, "inner_auroc": round(float(np.mean(scores[depth])), 4)}


def free_gpu():
    """Index of a GPU with no compute process on it, else None (never share a GPU)."""
    import torch
    if not torch.cuda.is_available():
        return None
    q = lambda args: subprocess.run(["nvidia-smi", *args, "--format=csv,noheader"],  # noqa: E731
                                    capture_output=True, text=True).stdout.split("\n")
    busy = {line.split(",")[0].strip() for line in q(["--query-compute-apps=gpu_uuid"]) if line.strip()}
    for line in q(["--query-gpu=index,uuid"]):
        if line.strip():
            idx, uuid = [s.strip() for s in line.split(",")]
            if uuid not in busy:
                return int(idx)
    return None
