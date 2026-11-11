"""Result figures shared by the modeling notebooks."""
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#405B76", "#E9783D", "#5F9E6E", "#B55D60", "#8172B3", "#937860", "#91ACC5", "#C0A33E"]


def per_split_auroc(met, split="test_cycle", models=None, title="", ax=None, labels=None):
    """AUROC of each model on each test split (lines), with the mean over splits in the legend.
    met: one row per (model, split) with an `auroc` column."""
    models = models or list(dict.fromkeys(met.model))
    piv = met.pivot(index=split, columns="model", values="auroc")[models]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 0.9 * len(piv) + 3), 4.2))
    x = np.arange(len(piv))
    for i, m in enumerate(models):
        name = (labels or {}).get(m, m)
        ax.plot(x, piv[m], "o-", color=PALETTE[i % len(PALETTE)], label=f"{name} (mean {piv[m].mean():.3f})")
    ax.set_xticks(x, piv.index, rotation=30 if len(piv) > 7 else 0)
    ax.set_ylabel("AUROC (PHQ-9 ≥ 10)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def mean_auroc_bars(summary, title="", ax=None, ref=None, ref_label=None):
    """Horizontal bars of mean AUROC (± sd over splits). summary: index = label, columns mean, sd."""
    summary = summary.iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.45 * len(summary) + 1.5))
    sd = summary["sd"].fillna(0) if "sd" in summary else 0 * summary["mean"]
    ax.barh(summary.index, summary["mean"], xerr=sd, color="#91ACC5", capsize=3)
    for i, (v, e) in enumerate(zip(summary["mean"], sd)):
        ax.text(v + e + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    lo = float(np.nanmin(summary["mean"] - sd)) - 0.02
    if ref is not None:
        lo = min(lo, ref - 0.01)
    ax.set_xlim(max(0.5, lo), min(1.0, float((summary["mean"] + sd).max()) + 0.02))
    if ref is not None:
        ax.axvline(ref, color="#E9783D", ls="--", label=ref_label)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlabel("mean AUROC over test splits (± sd)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def summarise(met, by="model", order=None):
    """Mean and sd of AUROC per model, in `order`."""
    s = met.groupby(by).auroc.agg(mean="mean", sd="std")
    return s.reindex(order) if order else s


def show():
    plt.tight_layout()
    plt.show()

