"""Default prediction with a limited labelling budget (ADGSSL components).

Setting: application data are available for a pool of loans, but outcomes
can be obtained for only a share of them (for example, when a label requires
manual review, or when performance has not yet been observed). The question is
how much discrimination is achieved with b% of the pool labelled, and whether
the choice of which loans to label (random or SJUS) and of the learner
(logistic regression on the labelled loans, or measure propagation over a
graph of all loans) matter.

Design
- 20 pools of 2,000 loans drawn at random from the development sample of
  run_scorecard.py (same split and seed). The pools overlap, because each
  takes 2,000 of 6,899 loans.
- Budgets b = 2%, 5%, 10%, 20% and 40% of the pool (40 to 800 loans).
- Selection: random, or SJUS with Euclidean distances on the label-free
  feature matrix of the pool (ADGSSL also offers cosine distances; they are
  not used here).
- Learners, with hyperparameters fixed in advance (no labelled validation set
  is assumed to exist):
  - L2-regularised logistic regression (C = 1) on the labelled loans;
  - measure propagation (K = 5, mu = 0.1, nu = 0.01) on a symmetric
    K-nearest-neighbour graph of the pool and the holdout loans, run until q
    changes by less than 1e-6;
  - as a sensitivity check, measure propagation exactly as in the ADGSSL
    code: directed graph and the original stopping rule (tol = 0.1).
- Features: numeric variables (log1p for amounts), a missing indicator and
  median imputation for employment length, and a dummy for the loan term.
  The imputation median and the dummy categories come from the development
  sample. The standardisation is computed on the pool and holdout loans
  together, as in a transductive setting where all unlabelled loans are
  available; no outcome is used.
- Evaluation: AUC on the holdout sample of run_scorecard.py (2,958 loans),
  the same for every method, budget and pool. If the labelled set contains
  no bad loan, the method cannot rank and its AUC is recorded as 0.5.
- References: logistic regression on all 6,899 development loans with the
  same features, and the WOE scorecard of run_scorecard.py.

Usage (from the repository root, after run_scorecard.py):
    python run_label_budget.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from run_scorecard import HOLDOUT_SHARE, SEED, md_table
from scorecard import data, plots
from scorecard import label_budget as lb
from scorecard import metrics as M

N_POOLS = 20
POOL_SIZE = 2000
BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.40]
MP_MAIN = dict(mu=0.1, nu=0.01, stop="q_change", eps=1e-6, max_iter=1000)
MP_ADGSSL = dict(mu=0.1, nu=0.01, stop="adgssl", tol=0.1, max_iter=1000)
K_NEIGHBOURS = 5
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MAIN = ["Random + logistic", "SJUS + logistic", "Random + MP", "SJUS + MP"]
SENSITIVITY = ["Random + MP (ADGSSL code)", "SJUS + MP (ADGSSL code)"]


def logistic_risk(X_lab, y_lab, X_eval):
    if y_lab.min() == y_lab.max():
        return None
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X_lab, y_lab)
    return clf.predict_proba(X_eval)[:, 1]


def mp_risk(W, n_nodes, labelled_nodes, labelled_y, eval_nodes, params):
    if labelled_y.min() == labelled_y.max():
        return None, np.nan
    labels = np.full(n_nodes, -1)
    labels[labelled_nodes] = labelled_y
    q, classes, iters = lb.measure_propagation(W, labels, **params)
    return q[eval_nodes, int(np.flatnonzero(classes == 1)[0])], iters


def auc_or_half(y, risk) -> float:
    return 0.5 if risk is None else M.auc(y, risk)


def neighbour_bad_share(X: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    """Mean share of bads among the k nearest neighbours, for bads and for goods."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    share = y[idx].mean(axis=1)
    return float(share[y == 1].mean()), float(share[y == 0].mean())


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = data.load()
    Xd, Xh, yd, yh = train_test_split(X, y, test_size=HOLDOUT_SHARE, stratify=y, random_state=SEED)
    Xd, Xh = Xd.reset_index(drop=True), Xh.reset_index(drop=True)
    F = lb.feature_matrix(pd.concat([Xd, Xh], ignore_index=True), data.NUMERIC, data.CATEGORICAL, reference=Xd)
    Fd, Fh = F.iloc[: len(Xd)].to_numpy(), F.iloc[len(Xd):].to_numpy()

    # references (all development labels) and outcome similarity of neighbours
    sc_all = StandardScaler().fit(Fd)
    ref_logit = M.auc(yh, LogisticRegression(C=1.0, max_iter=2000).fit(sc_all.transform(Fd), yd).predict_proba(sc_all.transform(Fh))[:, 1])
    disc = pd.read_csv(TABLES / "discrimination.csv") if (TABLES / "discrimination.csv").exists() else None
    ref_scorecard = float(disc.loc[disc["sample"] == "Holdout", "AUC"].iloc[0]) if disc is not None else np.nan
    nb_bad, nb_good = neighbour_bad_share(sc_all.transform(Fd), yd, K_NEIGHBOURS)

    rows = []
    t0 = time.time()
    for rep in range(N_POOLS):
        rng = np.random.default_rng(1000 + rep)
        pool = rng.choice(len(Xd), POOL_SIZE, replace=False)
        scaler = StandardScaler().fit(np.vstack([Fd[pool], Fh]))  # label-free
        Xp, Xe = scaler.transform(Fd[pool]), scaler.transform(Fh)
        yp = yd[pool]

        phi, psi, clusters = lb.sjus_scores(Xp)
        nodes = np.vstack([Xp, Xe])  # pool first, then holdout
        graphs = {"main": lb.knn_graph(nodes, K_NEIGHBOURS, symmetric=True), "adgssl": lb.knn_graph(nodes, K_NEIGHBOURS, symmetric=False)}
        n_nodes = len(nodes)
        eval_nodes = np.arange(POOL_SIZE, n_nodes)

        for b in BUDGETS:
            budget = math.ceil(b * POOL_SIZE)
            selections = {
                "Random": np.sort(rng.choice(POOL_SIZE, budget, replace=False)),
                "SJUS": lb.sjus_select(phi, psi, clusters, budget),
            }
            for strategy, s in selections.items():
                common = {"pool": rep, "budget": b, "labelled": budget, "bads_labelled": int(yp[s].sum())}
                lr = logistic_risk(Xp[s], yp[s], Xe)
                rows.append({**common, "method": f"{strategy} + logistic", "auc": auc_or_half(yh, lr), "mp_iterations": np.nan})
                mp, it = mp_risk(graphs["main"], n_nodes, s, yp[s], eval_nodes, MP_MAIN)
                rows.append({**common, "method": f"{strategy} + MP", "auc": auc_or_half(yh, mp), "mp_iterations": it})
                mp, it = mp_risk(graphs["adgssl"], n_nodes, s, yp[s], eval_nodes, MP_ADGSSL)
                rows.append({**common, "method": f"{strategy} + MP (ADGSSL code)", "auc": auc_or_half(yh, mp), "mp_iterations": it})
        print(f"pool {rep + 1}/{N_POOLS} done ({time.time() - t0:.0f} s)")

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "label_budget_runs.csv", index=False)

    summary = (
        runs.groupby(["budget", "method"])
        .agg(
            labelled=("labelled", "first"),
            bads_labelled_mean=("bads_labelled", "mean"),
            pools_without_bad=("bads_labelled", lambda s: int((s == 0).sum())),
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            mp_iterations_median=("mp_iterations", "median"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], MAIN + SENSITIVITY, ordered=True)
    summary = summary.sort_values(["budget", "method"]).reset_index(drop=True)
    summary["method"] = summary["method"].astype(str)
    summary.to_csv(TABLES / "label_budget_summary.csv", index=False)

    wide = runs.pivot_table(index=["pool", "budget"], columns="method", values="auc").reset_index()
    comparisons = [
        ("SJUS - Random, logistic", "SJUS + logistic", "Random + logistic"),
        ("SJUS - Random, MP", "SJUS + MP", "Random + MP"),
        ("MP - logistic, random selection", "Random + MP", "Random + logistic"),
        ("MP - logistic, SJUS selection", "SJUS + MP", "SJUS + logistic"),
        ("SJUS - Random, MP (ADGSSL code)", "SJUS + MP (ADGSSL code)", "Random + MP (ADGSSL code)"),
    ]
    diffs = []
    for name, a, b_ in comparisons:
        for b in BUDGETS:
            d = (wide.loc[wide["budget"] == b, a] - wide.loc[wide["budget"] == b, b_]).to_numpy()
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            diffs.append(
                {
                    "comparison": name,
                    "budget": b,
                    "mean_difference": d.mean(),
                    "ci_low": d.mean() - half,
                    "ci_high": d.mean() + half,
                    "pools_positive": f"{int((d > 0).sum())} of {len(d)}",
                }
            )
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(TABLES / "label_budget_differences.csv", index=False)

    plots.label_budget(summary[summary["method"].isin(MAIN)], ref_logit, ref_scorecard, FIGURES / "label_budget.png")

    fmt_budget = lambda v: f"{v:.0%}"  # noqa: E731
    main_tab = summary[summary["method"].isin(MAIN)].drop(columns=["mp_iterations_median"])
    sens_tab = summary[summary["method"].isin(SENSITIVITY)]
    iters_main = runs.loc[runs["method"].isin(["Random + MP", "SJUS + MP"]), "mp_iterations"]
    iters_sjus_adgssl = runs.loc[runs["method"] == "SJUS + MP (ADGSSL code)", "mp_iterations"].dropna()
    lines = [
        "# Label-budget experiment\n",
        "Generated by `run_label_budget.py`; the design is described in the docstring of that script. "
        f"{N_POOLS} pools of {POOL_SIZE:,} development loans; every AUC is computed on the same holdout sample "
        f"({len(yh):,} loans, {int(yh.sum())} bad).\n",
        f"References on the holdout sample: logistic regression on all {len(yd):,} development loans with the same features, AUC {ref_logit:.3f}; "
        f"WOE scorecard of `run_scorecard.py`, AUC {ref_scorecard:.3f}.\n",
        "## Main results\n",
        "Measure propagation (MP) here uses the symmetric nearest-neighbour graph and runs until the class distributions stop changing "
        f"(median {int(iters_main.median())} iterations).\n",
        md_table(
            main_tab.assign(budget=main_tab["budget"].map(fmt_budget)),
            {"bads_labelled_mean": ".1f", "auc_mean": ".3f", "auc_sd": ".3f", "auc_min": ".3f", "auc_max": ".3f"},
        ),
        "",
        "`bads_labelled_mean`: average number of bad loans in the labelled set. `pools_without_bad`: pools in which the labelled set "
        "contained no bad loan; the AUC of those runs is 0.5.\n",
        "![Label budget](figures/label_budget.png)\n",
        "## Paired differences\n",
        "Differences in holdout AUC between two methods on the same pool and budget, with a 95% t-interval over the 20 pools. "
        "The pools overlap and share one holdout sample, so the differences are not independent and the intervals are likely too narrow; "
        "they indicate the size and consistency of a difference rather than a formal test.\n",
        md_table(diffs.assign(budget=diffs["budget"].map(fmt_budget)), {"mean_difference": "+.3f", "ci_low": "+.3f", "ci_high": "+.3f"}),
        "",
        "## Sensitivity: measure propagation as in the ADGSSL code\n",
        "Directed nearest-neighbour graph and the original stopping rule (stop when the relative decrease of the convergence measure is at most 0.1). "
        f"With SJUS selection this rule stopped after {int(iters_sjus_adgssl.median())} iterations in the median run "
        f"({int((iters_sjus_adgssl <= 2).sum())} of {len(iters_sjus_adgssl)} runs stopped after at most 2 iterations).\n",
        md_table(sens_tab.assign(budget=sens_tab["budget"].map(fmt_budget)), {"bads_labelled_mean": ".1f", "auc_mean": ".3f", "auc_sd": ".3f", "auc_min": ".3f", "auc_max": ".3f", "mp_iterations_median": ".0f"}),
        "",
        "## Outcome similarity of neighbouring loans\n",
        f"On the development sample, among the {K_NEIGHBOURS} nearest neighbours of each loan (standardised features), the share of bad loans is "
        f"{nb_bad:.1%} for bad loans and {nb_good:.1%} for good loans (overall bad rate {yd.mean():.2%}).\n",
    ]
    (REPORTS / "label_budget.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string())
    print(diffs.to_string())
    print(f"reference logistic {ref_logit:.3f}, scorecard {ref_scorecard:.3f}, neighbour bad share {nb_bad:.4f}/{nb_good:.4f}")


if __name__ == "__main__":
    main()
