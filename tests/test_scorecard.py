"""Checks of the building blocks on synthetic data.

Run with ``python -m pytest tests`` or ``python tests/test_scorecard.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorecard import binning, label_budget, metrics, model  # noqa: E402


def _synthetic(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 6, size=n).astype(float)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    eta = -3 + 0.8 * x1 + 0.3 * x2 + np.select([cat == "b", cat == "c"], [0.4, 0.9], 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y


def test_numeric_binning_is_monotonic_and_respects_min_share():
    df, y = _synthetic()
    b = binning.bin_numeric(df["x1"], y, "x1", min_share=0.05, max_bins=6)
    rates = b.table["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) > 0) or np.all(np.diff(rates) < 0)
    assert b.table["share"].min() >= 0.05 - 1e-12
    assert len(b.table) <= 6
    assert b.table["n"].sum() == len(df)


def test_woe_sign_follows_bad_rate():
    df, y = _synthetic()
    b = binning.bin_categorical(df["cat"], y, "cat")
    overall = y.mean()
    for _, r in b.table.iterrows():
        assert np.sign(r["woe"]) == np.sign(r["bad_rate"] - overall) or abs(r["bad_rate"] - overall) < 0.002


def test_missing_values_get_their_own_bin():
    df, y = _synthetic()
    x = df["x1"].copy()
    x.iloc[:300] = np.nan
    b = binning.bin_numeric(x, y, "x1")
    assert b.table["bin"].iloc[-1] == binning.MISSING_LABEL
    assert b.table["n"].iloc[-1] == 300
    assert np.all(b.assign(x)[:300] == len(b.table) - 1)


def test_logit_matches_unpenalised_sklearn():
    df, y = _synthetic()
    X = df[["x1", "x2"]].to_numpy()
    fit = model.fit_logit(X, y, ["x1", "x2"])
    ref = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-10).fit(X, y)
    assert np.allclose(fit.coef[1:], ref.coef_.ravel(), atol=1e-4)
    assert np.isclose(fit.coef[0], ref.intercept_[0], atol=1e-4)


def test_score_scaling():
    s = model.ScoreScale(base_score=600, base_odds=50, pdo=20)
    assert np.isclose(s.score_to_pd(600), 1 / 51)
    assert np.isclose(s.score_to_pd(620), 1 / 101)


def test_ks_and_auc_on_perfect_and_random_rankings():
    y = np.array([0] * 90 + [1] * 10)
    perfect = np.arange(100)
    assert metrics.auc(y, perfect) == 1.0
    assert np.isclose(metrics.ks(y, perfect), 1.0)
    rng = np.random.default_rng(1)
    noise = rng.random(100_000)
    yy = (rng.random(100_000) < 0.1).astype(int)
    assert abs(metrics.auc(yy, noise) - 0.5) < 0.01


def test_psi_is_zero_for_identical_distributions():
    counts = np.array([10, 20, 30, 40])
    assert metrics.psi_from_counts(counts, counts * 3) == 0.0
    assert metrics.psi_from_counts(counts, counts[::-1]) > 0.25


def test_hosmer_lemeshow_degrees_of_freedom():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.2, size=5000)
    y = (rng.random(5000) < p).astype(int)
    stat, p_in, _ = metrics.hosmer_lemeshow(y, p, groups=10)
    stat2, p_out, _ = metrics.hosmer_lemeshow(y, p, groups=10, df=10)
    assert stat == stat2 and p_out > p_in


def test_rounded_score_stays_close_to_exact_score():
    df, y = _synthetic()
    bins = binning.bin_all(df, y, ["x1", "x2"], ["cat"])
    woe = pd.DataFrame({k: b.transform(df[k]) for k, b in bins.items()})
    variables = ["x1", "x2", "cat"]
    fit = model.fit_logit(woe[variables].to_numpy(), y, variables)
    card = model.Scorecard(bins, variables, fit, model.ScoreScale())
    card.center_rounding(df)
    assert abs(card.rounding_shift(df)) <= 0.5 + 1e-9


def test_jeffreys_interval_contains_observed_rate():
    lo, hi, p = metrics.jeffreys(12, 400, 0.03)
    assert lo < 12 / 400 < hi
    assert 0 < p < 1


def test_measure_propagation_returns_distributions_and_keeps_clear_labels():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, size=(100, 2))
    b = rng.normal(3, 0.3, size=(100, 2))
    X = np.vstack([a, b])
    labels = np.full(200, -1)
    labels[[0, 1, 2]] = 0
    labels[[100, 101, 102]] = 1
    W = label_budget.knn_graph(X, 5)
    assert (W != W.T).nnz == 0  # symmetric by default
    assert np.all(np.diff(label_budget.knn_graph(X, 5, symmetric=False).indptr) == 5)
    q, classes, _ = label_budget.measure_propagation(W, labels)
    assert np.allclose(q.sum(axis=1), 1)
    pred = classes[q.argmax(axis=1)]
    assert (pred[:100] == 0).mean() > 0.95 and (pred[100:] == 1).mean() > 0.95


def test_sjus_selects_exactly_the_budget():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(150, 3)), rng.normal(4, 1, size=(150, 3))])
    phi, psi, clusters = label_budget.sjus_scores(X)
    for budget in (10, 30, 90):
        sel = label_budget.sjus_select(phi, psi, clusters, budget)
        assert len(sel) == budget == len(np.unique(sel))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
