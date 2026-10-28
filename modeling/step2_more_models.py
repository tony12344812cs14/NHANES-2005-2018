"""Step 2: more models on the same leave-one-cycle-out (LOCO) design and metrics as step 1.

  knn       k-nearest neighbours on the logistic-regression design matrix (one-hot + standardised);
            k chosen by the inner LOCO over the training cycles.
  catboost  all features; the categorical features go in as CatBoost categoricals (missing and -1
            are their own levels), numeric NaN handled natively; depth chosen by the inner LOCO,
            number of trees by early stopping.
  tabm      all features; TabM (Gorishniy et al. 2025) with the package defaults (k = 32, 3 x 512,
            dropout 0.1), numeric features quantile-normalised with a not-applicable indicator,
            categorical features as integer codes. Not grid-tuned (deep models are costly to tune and
            the defaults are the authors' recommendation); epochs by early stopping on a stratified 15%
            of the training rows. Runs on a GPU if one with no processes on it is free, else on CPU.

Usage: python modeling/step2_more_models.py [--models knn catboost tabm] [--quick]
Output: <NHANES_WORK>/modeling/step2/ predictions_<model>.csv.gz, metrics.csv, chosen_params.json,
        report.txt (also compares with step 1)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import RESULTS, THREADS  # noqa: E402
from step1_loco import FEAT, NOT_FEATURES, SEED, logreg_matrix, metrics  # noqa: E402

OUT = RESULTS / "step2"
KNN_GRID = [50, 150, 400]
CAT_GRID = [4, 6]
CAT_FIXED = dict(learning_rate=0.05, iterations=3000, od_type="Iter", od_wait=100, eval_metric="AUC",
                 auto_class_weights="Balanced", thread_count=THREADS, random_seed=SEED, verbose=False,
                 allow_writing_files=False)
TABM = dict(lr=2e-3, weight_decay=3e-4, batch=256, max_epochs=100, patience=16)


# ---------------- kNN ----------------
def _knn_auc(Xtr, ytr, Xva, yva, k):
    return roc_auc_score(yva, KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict_proba(Xva)[:, 1])


def run_knn(data, train_cycles, test_c, dic):
    jobs = []
    for val_c in train_cycles:
        tr, va = data[data.cycle.isin(train_cycles) & (data.cycle != val_c)], data[data.cycle == val_c]
        A, B = logreg_matrix(tr.drop(columns=NOT_FEATURES), va.drop(columns=NOT_FEATURES), dic)
        jobs += [(k, delayed(_knn_auc)(A, tr.depression, B, va.depression, k)) for k in KNN_GRID]
    scores = {k: [] for k in KNN_GRID}
    for (k, _), auc in zip(jobs, Parallel(n_jobs=THREADS)(j for _, j in jobs)):
        scores[k].append(auc)
    k = max(scores, key=lambda k: np.mean(scores[k]))
    tr, te = data[data.cycle != test_c], data[data.cycle == test_c]
    A, B = logreg_matrix(tr.drop(columns=NOT_FEATURES), te.drop(columns=NOT_FEATURES), dic)
    prob = KNeighborsClassifier(n_neighbors=k, n_jobs=THREADS).fit(A, tr.depression).predict_proba(B)[:, 1]
    return prob, {"n_neighbors": k, "inner_auroc": round(float(np.mean(scores[k])), 4)}


# ---------------- CatBoost ----------------
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


# ---------------- TabM ----------------
def free_gpu():
    """Index of a GPU with no compute process on it, else None (never share a GPU)."""
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


def tabm_arrays(train, test, dic):
    """Quantile-normal numeric features (+ not-applicable indicators) and integer-coded categoricals."""
    feats = [c for c in train if c not in NOT_FEATURES]
    cat = [c for c in feats if dic.get(c) == "categorical"]
    num = [c for c in feats if c not in cat]
    na = [c for c in num if (train[c] == -1).any()]

    def numeric(df):
        x = df[num].mask(df[num] == -1)
        return pd.concat([x, (df[na] == -1).astype(float).add_suffix("__na")], axis=1)

    ntr, nte = numeric(train), numeric(test)
    med = ntr.median()
    ntr, nte = ntr.fillna(med), nte.fillna(med)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=10**9, random_state=SEED)
    xtr, xte = qt.fit_transform(ntr).astype(np.float32), qt.transform(nte).astype(np.float32)
    ctr, cte, card = [], [], []
    for c in cat:
        levels = {v: i for i, v in enumerate(sorted(train[c].fillna(-99).unique()))}
        ctr.append(train[c].fillna(-99).map(levels).to_numpy())
        cte.append(test[c].fillna(-99).map(levels).fillna(len(levels)).to_numpy())  # unseen level
        card.append(len(levels) + 1)
    return xtr, xte, np.stack(ctr, 1).astype(np.int64), np.stack(cte, 1).astype(np.int64), card


def tabm_predict(model, x, c, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            logit = model(torch.as_tensor(x[i:i + 4096], device=device), torch.as_tensor(c[i:i + 4096], device=device))
            out.append(torch.sigmoid(logit.squeeze(-1)).mean(1).cpu().numpy())  # mean over the k heads
    return np.concatenate(out)


def run_tabm(data, train_cycles, test_c, dic, device):
    import tabm
    torch.manual_seed(SEED)
    tr, te = data[data.cycle != test_c], data[data.cycle == test_c]
    fit_idx, es_idx = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=tr.depression, random_state=SEED)
    fit, es = tr.iloc[fit_idx], tr.iloc[es_idx]
    xf, xe, cf, ce, card = tabm_arrays(fit, es, dic)
    _, xt, _, ct, _ = tabm_arrays(fit, te, dic)
    yf = torch.as_tensor(fit.depression.to_numpy(), dtype=torch.float32, device=device)
    model = tabm.TabM.make(n_num_features=xf.shape[1], cat_cardinalities=card, d_out=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TABM["lr"], weight_decay=TABM["weight_decay"])
    pos_w = torch.tensor((len(yf) - yf.sum().item()) / yf.sum().item(), device=device)
    xf_t, cf_t = torch.as_tensor(xf, device=device), torch.as_tensor(cf, device=device)
    best, best_state, bad, epoch = -1.0, None, 0, 0
    for epoch in range(1, TABM["max_epochs"] + 1):
        model.train()
        for b in torch.randperm(len(yf), device=device).split(TABM["batch"]):
            logit = model(xf_t[b], cf_t[b]).squeeze(-1)  # (batch, k): each head trained on its own loss
            y = yf[b].unsqueeze(1).expand_as(logit)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(es.depression, tabm_predict(model, xe, ce, device))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= TABM["patience"]:
                break
    model.load_state_dict(best_state)
    prob = tabm_predict(model, xt, ct, device)
    return prob, {"epochs_run": epoch, "early_stop_auroc": round(best, 4), "device": str(device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["knn", "catboost", "tabm"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.quick:
        OUT = RESULTS / "step2_quick"
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FEAT / "features_2005_2018.csv.gz")
    dic = pd.read_csv(FEAT / "feature_dictionary.csv").set_index("column").type.to_dict()
    cycles = sorted(data.cycle.unique())
    outer = cycles[:1] if args.quick else cycles
    if args.quick:
        KNN_GRID[:] = KNN_GRID[:1]
        CAT_GRID[:] = CAT_GRID[:1]
        TABM["max_epochs"] = 3
    device = torch.device("cpu")
    torch.set_num_threads(THREADS)

    met_path, par_path = OUT / "metrics.csv", OUT / "chosen_params.json"
    rows = pd.read_csv(met_path).to_dict("records") if met_path.exists() and not args.quick else []
    chosen = json.loads(par_path.read_text()) if par_path.exists() and not args.quick else {}
    for model in args.models:
        if model == "tabm":  # checked when TabM starts: it runs last, hours after launch
            gpu = free_gpu()
            device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
        rows = [r for r in rows if r["model"] != model]
        preds = []
        for test_c in outer:
            t0 = time.time()
            train_cycles = [c for c in cycles if c != test_c]
            if model == "knn":
                prob, params = run_knn(data, train_cycles, test_c, dic)
            elif model == "catboost":
                prob, params = run_catboost(data, train_cycles, test_c, dic)
            else:
                prob, params = run_tabm(data, train_cycles, test_c, dic, device)
            te = data[data.cycle == test_c]
            chosen[f"{model}|{test_c}"] = params
            preds.append(pd.DataFrame({"SEQN": te.SEQN, "cycle": test_c, "model": model,
                                       "depression": te.depression, "weight": te.WTMEC2YR, "prob": prob}))
            rows.append({"model": model, "test_cycle": test_c, **metrics(te.depression, prob, te.WTMEC2YR),
                         "minutes": round((time.time() - t0) / 60, 1)})
            print(f"{test_c} {model}: AUROC {rows[-1]['auroc']:.3f}  {params}  {rows[-1]['minutes']} min", flush=True)
        pd.concat(preds).to_csv(OUT / f"predictions_{model}.csv.gz", index=False)
        pd.DataFrame(rows).to_csv(met_path, index=False)
        par_path.write_text(json.dumps(chosen, indent=1))

    met = pd.DataFrame(rows)
    step1 = RESULTS / "step1" / "metrics.csv"
    if step1.exists():
        met = pd.concat([pd.read_csv(step1), met], ignore_index=True)
    order = ["logreg", "knn", "xgboost", "catboost", "tabm"]
    summ = met.groupby("model")[["auroc", "auroc_weighted", "auprc", "brier"]].agg(["mean", "std"]).round(3)
    summ = summ.reindex([m for m in order if m in summ.index])
    per = met.pivot(index="test_cycle", columns="model", values="auroc").round(3)
    per = per[[m for m in order if m in per.columns]]
    lines = ["leave-one-cycle-out, outcome PHQ-9 >= 10; step 1 (logreg, xgboost) + step 2", "",
             "mean (sd) over test cycles, each cycle weighted equally:", summ.to_string(), "",
             "AUROC per test cycle:", per.to_string(), "",
             f"TabM device: {device}"]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
