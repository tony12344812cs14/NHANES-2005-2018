"""Step 8b: TabPFN and TabICL on the paper1 bundle, 10-fold. Standalone: runs on any GPU machine.

Needs only: python >= 3.10, numpy, pandas, scikit-learn, torch, tabpfn==9.2.0, tabicl==2.2.0
(see modeling/requirements-foundation.txt). Input is the bundle from step8_export_bundle.py.

Variants (fixed before running; package defaults, no tuning, since these models are meant to be used
without tuning):
  tabpfn_clf   TabPFNClassifier (default checkpoint, TabPFN-3.5) on the 0/1 label; score = P(depressed)
  tabpfn_reg   TabPFNRegressor on the PHQ-9 total; score = predicted total
  tabicl_clf   TabICLClassifier (default checkpoint, TabICLv2) on the 0/1 label
  tabicl_reg   TabICLRegressor on the PHQ-9 total
For each test fold the model is fitted on the other 9 folds only. Categorical features are passed as
pandas 'category' columns; continuous features as floats (-1 = not applicable, NaN = missing).

TabPFN-3.5 weights need a (free, non-commercial) license: accept it once at https://ux.priorlabs.ai and
set TABPFN_TOKEN on a headless machine. TabICL weights download from Hugging Face on first use.

Usage: python step8_foundation_models.py BUNDLE_DIR OUT_DIR [--models tabpfn_clf,...] [--folds 0,1,...]
           [--device cuda:0] [--quick]
Each (model, fold) is written to OUT_DIR/<model>/fold<k>.csv (SEQN, fold, depression, phq9, score) as soon
as it finishes; finished ones are skipped on a rerun, so the script can be stopped and resumed.
--quick: fold 0 only, 2,000 training rows (checks the setup in a few minutes; written to OUT_DIR/quick/).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

MODELS = ["tabpfn_clf", "tabpfn_reg", "tabicl_clf", "tabicl_reg"]
SEED = 42


def load(bundle):
    data = pd.read_csv(bundle / "data.csv.gz")
    cols = json.loads((bundle / "columns.json").read_text())
    X = data[cols["categorical"] + cols["continuous"]].copy()
    for c in cols["categorical"]:
        X[c] = X[c].astype("Int64").astype("category")
    X[cols["continuous"]] = X[cols["continuous"]].astype(float)
    return data, X


def make(model, device):
    if model.startswith("tabpfn"):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        cls = TabPFNClassifier if model.endswith("clf") else TabPFNRegressor
        return cls(device=device, random_state=SEED)
    from tabicl import TabICLClassifier, TabICLRegressor
    cls = TabICLClassifier if model.endswith("clf") else TabICLRegressor
    return cls(device=device, random_state=SEED)


def score(m, model, X, chunk):
    """Predict in chunks of `chunk` rows (0 = all at once)."""
    parts = [X] if not chunk else [X.iloc[i:i + chunk] for i in range(0, len(X), chunk)]
    if model.endswith("clf"):
        return np.concatenate([m.predict_proba(p)[:, list(m.classes_).index(1)] for p in parts])
    return np.concatenate([np.asarray(m.predict(p), dtype=float) for p in parts])


def run_one(model, data, X, k, device, quick):
    tr, te = np.flatnonzero(data.fold != k), np.flatnonzero(data.fold == k)
    if quick:
        tr = np.random.default_rng(SEED).choice(tr, 2000, replace=False)
    y = data.depression if model.endswith("clf") else data.phq9.astype(float)
    m = make(model, device).fit(X.iloc[tr], y.iloc[tr])
    for chunk in (0, 1000, 250):
        try:
            return score(m, model, X.iloc[te], chunk), te
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  out of GPU memory predicting fold {k} with chunk={chunk or 'all'}; retrying smaller", flush=True)
    raise RuntimeError("out of GPU memory even with 250-row chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--folds", default=",".join(map(str, range(10))))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data, X = load(args.bundle)
    out = args.out / "quick" if args.quick else args.out
    folds = [0] if args.quick else [int(k) for k in args.folds.split(",")]
    gpu = torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else "cpu"
    print(f"{len(data)} people, {X.shape[1]} features; device {args.device} ({gpu}); "
          f"torch {torch.__version__}, python {platform.python_version()}", flush=True)
    for model in args.models.split(","):
        assert model in MODELS, model
        (out / model).mkdir(parents=True, exist_ok=True)
        for k in folds:
            path = out / model / f"fold{k}.csv"
            if path.exists():
                print(f"{model} fold {k}: done before, skipped", flush=True)
                continue
            t0 = time.time()
            s, te = run_one(model, data, X, k, args.device, args.quick)
            res = data.iloc[te][["SEQN", "fold", "depression", "phq9"]].assign(score=s)
            res.to_csv(path, index=False)
            print(f"{model} fold {k}: AUROC {roc_auc_score(res.depression, res.score):.3f}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
