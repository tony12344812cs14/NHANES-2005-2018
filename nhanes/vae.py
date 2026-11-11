"""Denoising VAE of modeling/step5_vae.ipynb, moved unchanged because step 6 reuses it.

Denoising: 15% swap noise (an entry is replaced by the same column of a random other row). Loss: MSE on
numeric columns + BCE on one-hot columns + KL. Fixed architecture: d -> 256 -> 128 -> 32 (latent) ->
128 -> 256 -> d; Adam 1e-3, batch 256, early stopping (patience 10) on reconstruction loss of 10% of rows.
"""
import numpy as np
import torch

from nhanes.loco import SEED

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
