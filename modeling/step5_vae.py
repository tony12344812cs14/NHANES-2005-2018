"""Step 5: denoising VAE representation of the 86 features of Dang et al. (2024), then logreg / XGBoost.

Question: does an unsupervised denoising VAE make the paper's features more predictive?
Same people, outcome, LOCO folds, models and grids as step 3; only the inputs change.

Per outer fold (test cycle never seen):
  1. Design matrix of the 86 features as for logistic regression in step 1 (one-hot categoricals,
     scaled numerics), fitted on the 6 training cycles.
  2. A VAE is trained on the training cycles without labels. Denoising: 15% swap noise (an entry is
     replaced by the same column of a random other row). Loss: MSE on numeric columns + BCE on one-hot
     columns + KL. Fixed architecture, no tuning: d -> 256 -> 128 -> 32 (latent) -> 128 -> 256 -> d;
     Adam 1e-3, batch 256, early stopping (patience 10) on reconstruction loss of 10% of training rows.
  3. Each person is encoded as the latent mean (32 numbers).
  4. logreg and xgboost tuned by inner LOCO over the training cycles, as in step 1, on
       latent   the 32 latent values only
       concat   the 86 original features + the 32 latent values
The VAE of an outer fold is trained once on all 6 training cycles, so during inner tuning it has seen
the (unlabelled) inner validation cycle; the outer test cycle is never seen.

Usage: python modeling/step5_vae.py PATH/TO/FullData_rm_dl_bt.csv [--quick]
Output: <NHANES_WORK>/modeling/step5_vae/ predictions.csv.gz, metrics.csv, chosen_params.json, report.txt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from nhanes.paths import RESULTS, WORK  # noqa: E402
from step1_loco import (LOGREG_GRID, NOT_FEATURES, SEED, XGB_GRID, fit_logreg, fit_xgb, grid,  # noqa: E402
                        logreg_matrix, metrics, tune)
from step2_more_models import free_gpu  # noqa: E402
from step3_paper1_features import paper1_table  # noqa: E402

OUT = RESULTS / "step5_vae"
VAE = dict(hidden=(256, 128), latent=32, swap=0.15, lr=1e-3, batch=256, max_epochs=200, patience=10)


class Vae(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        h1, h2 = VAE["hidden"]
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, h1), torch.nn.ReLU(), torch.nn.Linear(h1, h2), torch.nn.ReLU())
        self.mu, self.logvar = torch.nn.Linear(h2, VAE["latent"]), torch.nn.Linear(h2, VAE["latent"])
        self.dec = torch.nn.Sequential(torch.nn.Linear(VAE["latent"], h2), torch.nn.ReLU(), torch.nn.Linear(h2, h1),
                                       torch.nn.ReLU(), torch.nn.Linear(h1, d))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar


def swap_noise(x, p):
    rows = torch.randint(0, len(x), x.shape, device=x.device)
    mask = torch.rand(x.shape, device=x.device) < p
    return torch.where(mask, x.gather(0, rows), x)


def vae_loss(out, x, binary):
    recon, mu, logvar = out
    mse = ((recon[:, ~binary] - x[:, ~binary]) ** 2).sum(1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(recon[:, binary], x[:, binary], reduction="none").sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return (mse + bce).mean(), kl.mean()


def train_vae(Xtr, device, quick):
    """Train on Xtr (numpy), return (model, binary-column mask, epochs run)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    binary = torch.as_tensor(np.isin(Xtr, [0, 1]).all(0), device=device)
    x = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    perm = rng.permutation(len(x))
    n_es = len(x) // 10
    xe, xf = x[perm[:n_es]], x[perm[n_es:]]
    model = Vae(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=VAE["lr"])
    best, best_state, wait, epoch = np.inf, None, 0, 0
    for epoch in range(1, (3 if quick else VAE["max_epochs"]) + 1):
        model.train()
        for idx in torch.randperm(len(xf), device=device).split(VAE["batch"]):
            xb = xf[idx]
            recon, kl = vae_loss(model(swap_noise(xb, VAE["swap"])), xb, binary)
            opt.zero_grad()
            (recon + kl).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            r = vae_loss(model(xe), xe, binary)[0].item()
        if r < best:
            best, best_state, wait = r, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= VAE["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval(), epoch, best


def encode(model, X, device):
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.cat([model.mu(model.enc(b)) for b in x.split(4096)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper1_csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    feats, dic = paper1_table(args.paper1_csv)
    ours = pd.read_csv(WORK / "features" / "features_2005_2018.csv.gz", usecols=NOT_FEATURES)
    data = ours.merge(feats, on="SEQN", how="left", validate="1:1")
    feat_cols = list(dic)
    gpu = free_gpu()
    device = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
    print(f"{len(data)} people, {len(feat_cols)} features; VAE on {device}", flush=True)

    cycles = sorted(data.cycle.unique())
    grids = {"logreg": grid(LOGREG_GRID), "xgboost": grid(XGB_GRID)}
    if args.quick:
        grids = {k: v[:1] for k, v in grids.items()}
    zcols = [f"z{i:02d}" for i in range(VAE["latent"])]
    preds, rows, chosen = [], [], {}
    for test_c in cycles[:1] if args.quick else cycles:
        train_cycles = [c for c in cycles if c != test_c]
        is_tr = data.cycle != test_c
        t0 = time.time()
        A, B = logreg_matrix(data.loc[is_tr, feat_cols], data.loc[~is_tr, feat_cols], dic)
        vae, epochs, es_loss = train_vae(A, device, args.quick)
        Z = np.zeros((len(data), VAE["latent"]), dtype=np.float32)
        Z[is_tr.to_numpy()], Z[~is_tr.to_numpy()] = encode(vae, A, device), encode(vae, B, device)
        zdf = pd.DataFrame(Z, columns=zcols, index=data.index)
        print(f"{test_c} VAE: {A.shape[1]} inputs, {epochs} epochs, early-stop recon loss {es_loss:.2f}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        inputs = {"latent": (pd.concat([data[NOT_FEATURES], zdf], axis=1), {c: "continuous" for c in zcols}),
                  "concat": (pd.concat([data, zdf], axis=1), {**dic, **{c: "continuous" for c in zcols}})}
        for rep, (frame, d) in inputs.items():
            tr, te = frame[is_tr], frame[~is_tr]
            Xtr, Xte = tr.drop(columns=NOT_FEATURES), te.drop(columns=NOT_FEATURES)
            for model in ["logreg", "xgboost"]:
                t1 = time.time()
                params, inner_auc = tune(model, frame, train_cycles, d, grids[model])
                if model == "logreg":
                    P, Q = logreg_matrix(Xtr, Xte, d)
                    prob = fit_logreg(P, tr.depression, params).predict_proba(Q)[:, 1]
                else:
                    fit_p = {k: v for k, v in params.items() if k != "n_estimators"}
                    prob = fit_xgb(Xtr, tr.depression, None, None, fit_p,
                                   n_estimators=params["n_estimators"]).predict_proba(Xte)[:, 1]
                name = f"vae_{rep}+{model}"
                chosen[f"{name}|{test_c}"] = {**params, "inner_auroc": round(inner_auc, 4), "vae_epochs": epochs}
                preds.append(pd.DataFrame({"SEQN": te.SEQN, "cycle": test_c, "model": name, "depression": te.depression,
                                           "weight": te.WTMEC2YR, "prob": prob}))
                rows.append({"model": name, "test_cycle": test_c, **metrics(te.depression, prob, te.WTMEC2YR),
                             "minutes": round((time.time() - t1) / 60, 1)})
                print(f"{test_c} {name}: AUROC {rows[-1]['auroc']:.3f}  {params}  {rows[-1]['minutes']} min", flush=True)

    pred, met = pd.concat(preds, ignore_index=True), pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv.gz", index=False)
    met.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "chosen_params.json").write_text(json.dumps(chosen, indent=1))
    step3 = RESULTS / "step3_paper1" / "metrics.csv"
    if step3.exists():
        met = pd.concat([pd.read_csv(step3).assign(model=lambda m: "raw+" + m.model), met], ignore_index=True)
    summary = met.groupby("model", sort=False)[["auroc", "auroc_weighted", "auprc", "brier"]].agg(["mean", "std"]).round(3)
    lines = ["leave-one-cycle-out, outcome PHQ-9 >= 10; features of Dang et al. (2024) through a denoising VAE",
             "raw+ = step 3 (the 86 features as they are); vae_latent = 32 latent values; vae_concat = 86 + 32", "",
             "mean (sd) over test cycles, each cycle weighted equally:", summary.to_string(), "",
             "AUROC per test cycle:", met.pivot(index="test_cycle", columns="model", values="auroc").round(3).to_string(),
             "", f"VAE device: {device}"]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
