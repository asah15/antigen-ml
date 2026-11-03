"""
evaluate.py — metrics and plots shared by both tracks.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, f1_score, accuracy_score,
                             balanced_accuracy_score, roc_curve,
                             precision_recall_curve)

# colorblind-safe palette
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    single = len(set(y_true.tolist())) < 2
    return {
        "roc_auc": float("nan") if single else roc_auc_score(y_true, y_prob),
        "pr_auc":  float("nan") if single else average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def best_threshold(y_true, y_prob):
    """Threshold maximising MCC on the given set (report-only; pick on val)."""
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_m = 0.5, -2
    for t in ts:
        m = matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)


def plot_roc_pr(results: dict, y_true, out_path):
    """results: {name: y_prob}. Saves a 2-panel ROC + PR figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (name, prob) in enumerate(results.items()):
        c = PALETTE[i % len(PALETTE)]
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax1.plot(fpr, tpr, color=c, lw=1.8,
                 label=f"{name} (AUC={roc_auc_score(y_true, prob):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ax2.plot(rec, prec, color=c, lw=1.8,
                 label=f"{name} (AP={average_precision_score(y_true, prob):.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    base = float(np.mean(y_true))
    ax2.plot([0, 1], [base, base], "--", color="#999", lw=1)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8, loc="lower left" if ax is ax2 else "lower right")
        ax.grid(alpha=0.25)
    fig.suptitle("Antigenicity model comparison (held-out test)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(summary_rows, out_path, metric="roc_auc"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["model"] for r in summary_rows]
    vals = [r[metric] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=[PALETTE[i % len(PALETTE)]
                                      for i in range(len(names))])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set(ylim=(0, 1.05), ylabel=metric.upper().replace("_", "-"),
           title=f"Held-out {metric.upper().replace('_', '-')} by model")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
