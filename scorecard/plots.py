"""Figures for the validation report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
MODEL = "#1f5f8b"
BENCH = "#c8702a"
BAD = "#b23a48"
GOOD = "#3a7ca5"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def woe_panels(binnings, variables, labels, path):
    n = len(variables)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, variables):
        tab = binnings[v].table
        x = np.arange(len(tab))
        colors = [BAD if w > 0 else GOOD for w in tab["woe"]]
        ax.bar(x, tab["woe"], color=colors, width=0.7)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(tab["bin"], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{labels.get(v, v)}  (IV = {binnings[v].iv:.3f})", fontsize=9.5, loc="left")
        ax.set_ylabel("WOE")
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.margins(y=0.25)
        for xi, (w, r) in enumerate(zip(tab["woe"], tab["bad_rate"])):
            ax.annotate(f"{r:.1%}", (xi, w), xytext=(0, 3 if w >= 0 else -10), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Weight of evidence by bin (labels: bad rate in the development sample)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roc_ks(y, risk_model, risk_bench, auc_model, auc_bench, ks_value, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for risk, color, label in [(risk_model, MODEL, f"Scorecard (AUC {auc_model:.3f})"), (risk_bench, BENCH, f"Lending Club sub-grade (AUC {auc_bench:.3f})")]:
        fpr, tpr, _ = roc_curve(y, risk)
        ax1.plot(fpr, tpr, color=color, linewidth=1.8, label=label)
    ax1.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.8)
    ax1.set_xlabel("False positive rate (goods)")
    ax1.set_ylabel("True positive rate (bads)")
    ax1.set_title("ROC curve, holdout sample", loc="left")
    ax1.legend(loc="lower right", fontsize=8.5)

    # Cumulative shares at each distinct score, so that tied scores are not split.
    yy = np.asarray(y)
    fpr, tpr, _ = roc_curve(yy, risk_model)
    n_bad, n_good = yy.sum(), len(yy) - yy.sum()
    frac = (tpr * n_bad + fpr * n_good) / len(yy)
    k = int(np.argmax(tpr - fpr))
    ax2.plot(frac, tpr, color=BAD, linewidth=1.8, label="Bads")
    ax2.plot(frac, fpr, color=GOOD, linewidth=1.8, label="Goods")
    ax2.vlines(frac[k], fpr[k], tpr[k], color=INK, linewidth=1)
    ax2.annotate(f"KS = {ks_value:.3f}", (frac[k], (tpr[k] + fpr[k]) / 2), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax2.set_xlabel("Share of loans, ordered from lowest score")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title("KS curve, holdout sample", loc="left")
    ax2.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration(hl_table: pd.DataFrame, jeffreys_bounds, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    x = hl_table["mean_predicted"].to_numpy()
    yv = hl_table["observed_bad_rate"].to_numpy()
    lo, hi = jeffreys_bounds
    lim = max(x.max(), np.max(hi)) * 1.1
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle="--", linewidth=0.8, label="Perfect calibration")
    ax.errorbar(x, yv, yerr=[yv - lo, hi - yv], fmt="o", color=MODEL, ecolor=MODEL, elinewidth=1, capsize=2.5, markersize=5, label="Observed bad rate (95% Jeffreys interval)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean predicted bad rate")
    ax.set_ylabel("Observed bad rate")
    ax.set_title("Calibration by decile of prediction, holdout", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def score_distribution(score, y, cuts, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    score = np.asarray(score)
    y = np.asarray(y)
    bins = np.arange(np.floor(score.min() / 5) * 5, np.ceil(score.max() / 5) * 5 + 5, 5)
    ax.set_axisbelow(True)
    for mask, color, label in [(y == 0, GOOD, f"Goods (n = {(y == 0).sum():,})"), (y == 1, BAD, f"Bads (n = {(y == 1).sum():,})")]:
        ax.hist(score[mask], bins=bins, density=True, histtype="stepfilled", alpha=0.18, color=color)
        ax.hist(score[mask], bins=bins, density=True, histtype="step", linewidth=1.6, color=color, label=label)
    for c in cuts:
        ax.axvline(c, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution, holdout sample (dotted lines: grade boundaries)", loc="left")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def label_budget(summary: pd.DataFrame, ref_logit: float, ref_scorecard: float, path):
    """Mean holdout AUC by budget; error bars are the SD over pools."""
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    styles = {
        "Random + logistic": (MUTED, "o", "-"),
        "SJUS + logistic": (MODEL, "o", "-"),
        "Random + MP": (MUTED, "s", "--"),
        "SJUS + MP": (BENCH, "s", "--"),
    }
    for method, (color, marker, ls) in styles.items():
        sub = summary[summary["method"] == method].sort_values("budget")
        if sub.empty:
            continue
        x = sub["budget"].to_numpy() * 100 * (0.97 if "logistic" in method else 1.03)
        ax.errorbar(x, sub["auc_mean"], yerr=sub["auc_sd"], color=color, marker=marker, linestyle=ls, capsize=3, linewidth=1.5, markersize=5, label=method)
    ax.axhline(ref_scorecard, color=INK, linewidth=0.9, linestyle=":", label=f"WOE scorecard, all labels ({ref_scorecard:.3f})")
    ax.axhline(ref_logit, color=MUTED, linewidth=0.9, linestyle="-.", label=f"Logistic, all labels ({ref_logit:.3f})")
    ax.axhline(0.5, color=MUTED, linewidth=0.6)
    ax.set_xscale("log")
    budgets = sorted(summary["budget"].unique())
    ax.set_xticks([b * 100 for b in budgets])
    ax.set_xticklabels([f"{b:.0%}" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Labelling budget (share of the 2,000-loan pool, log scale)")
    ax.set_ylabel("Holdout AUC (mean ± SD over 20 pools)")
    ax.set_title("Predicting bad loans with a limited number of labels", loc="left")
    ax.set_ylim(0.44, max(ref_logit, ref_scorecard) + 0.03)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
