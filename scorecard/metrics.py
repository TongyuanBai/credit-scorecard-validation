"""Discrimination, calibration and stability measures used in the validation report.

Conventions: ``y`` is 1 for a bad loan and 0 for a good loan; ``risk`` is any
quantity that increases with the risk of a bad outcome (a predicted
probability, or minus a score); ``pd_`` is a predicted probability of the bad
outcome. The binomial and Jeffreys tests are those used in PD backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------------------------------------------------
# discrimination
# ----------------------------------------------------------------------
def auc(y, risk) -> float:
    return float(roc_auc_score(y, risk))


def gini(y, risk) -> float:
    return 2 * auc(y, risk) - 1


def ks(y, risk) -> float:
    """Kolmogorov-Smirnov statistic: largest gap between the cumulative
    distributions of bads and goods over the risk ordering."""
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def bootstrap_ci(stat, y, *risks, n_boot: int = 1000, level: float = 0.95, seed: int = 0):
    """Percentile interval from a bootstrap stratified by y.

    ``stat`` receives (y, *risks) and returns a float. Several risk vectors can
    be passed so that, e.g., the difference of two AUCs is resampled jointly.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    risks = [np.asarray(r) for r in risks]
    idx_bad, idx_good = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(idx_bad, len(idx_bad)), rng.choice(idx_good, len(idx_good))])
        values[b] = stat(y[idx], *[r[idx] for r in risks])
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


# ----------------------------------------------------------------------
# calibration and backtesting
# ----------------------------------------------------------------------
def brier(y, pd_) -> float:
    return float(np.mean((np.asarray(pd_) - np.asarray(y)) ** 2))


def hosmer_lemeshow(y, pd_, groups: int = 10, df: int | None = None) -> tuple[float, float, pd.DataFrame]:
    """Hosmer-Lemeshow statistic over groups of equal size ordered by the
    predicted probability.

    Returns the statistic, its chi-square p-value and the table of observed
    and expected bads. ``df`` defaults to ``groups - 2``, the reference
    distribution on the sample used to fit the model; on an independent
    validation sample ``groups`` degrees of freedom are used (Hosmer,
    Lemeshow and Sturdivant, Applied Logistic Regression, 3rd ed., Section 5.5).
    """
    frame = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    frame["group"] = pd.qcut(frame["pd"].rank(method="first"), groups, labels=False)
    tab = frame.groupby("group").agg(n=("y", "size"), observed=("y", "sum"), expected=("pd", "sum"), mean_predicted=("pd", "mean"))
    tab["observed_bad_rate"] = tab["observed"] / tab["n"]
    e, o, n = tab["expected"], tab["observed"], tab["n"]
    hl = float((((o - e) ** 2) / (e * (1 - e / n))).sum())
    df = groups - 2 if df is None else df
    return hl, float(stats.chi2.sf(hl, df)), tab.reset_index()


def binomial_test_upper(defaults: int, n: int, pd_: float) -> float:
    """One-sided p-value of H0: true bad rate <= pd_ (prediction not too low)."""
    return float(stats.binom.sf(defaults - 1, n, pd_))


def jeffreys(defaults: int, n: int, pd_: float, level: float = 0.95) -> tuple[float, float, float]:
    """Jeffreys interval for the bad rate and the p-value of the Jeffreys test.

    The posterior of the bad rate under the Jeffreys prior is
    Beta(D + 1/2, N - D + 1/2), with D bads among N loans. The p-value is its
    distribution function at the predicted rate; a small value indicates that
    the prediction is too low, and one minus it tests the other direction.
    """
    a, b = defaults + 0.5, n - defaults + 0.5
    lo, hi = stats.beta.ppf([(1 - level) / 2, (1 + level) / 2], a, b)
    return float(lo), float(hi), float(stats.beta.cdf(pd_, a, b))


def grade_backtest(y, pd_, grade, grade_order) -> pd.DataFrame:
    """Observed bad rate against the mean predicted probability in each grade.

    Both tails are reported: ``p_too_low`` tests whether the prediction is
    too low (the usual direction in PD backtesting), ``p_too_high`` whether
    it is too high (conservative).
    """
    frame = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_), "grade": np.asarray(grade)})
    rows = []
    for g in grade_order:
        sub = frame[frame["grade"] == g]
        n, d, mpd = len(sub), int(sub["y"].sum()), float(sub["pd"].mean())
        lo, hi, p_jeff = jeffreys(d, n, mpd)
        rows.append(
            {
                "grade": g,
                "n": n,
                "bads": d,
                "observed_bad_rate": d / n,
                "mean_predicted": mpd,
                "jeffreys_low": lo,
                "jeffreys_high": hi,
                "p_too_low": p_jeff,
                "p_too_high": 1 - p_jeff,
                "p_binomial_too_low": binomial_test_upper(d, n, mpd),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# stability
# ----------------------------------------------------------------------
def psi_from_counts(expected_counts, actual_counts, eps: float = 1e-6) -> float:
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    e = np.clip(e / e.sum(), eps, None)
    a = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(expected_counts, actual_counts, labels, eps: float = 1e-6) -> pd.DataFrame:
    """Contribution of each bin to the PSI; empty bins are clipped as in psi_from_counts."""
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    ep, ap = e / e.sum(), a / a.sum()
    ec, ac = np.clip(ep, eps, None), np.clip(ap, eps, None)
    return pd.DataFrame({"bin": labels, "expected_share": ep, "actual_share": ap, "psi": (ac - ec) * np.log(ac / ec)})


def stability_label(psi: float) -> str:
    """Rule of thumb commonly used in credit model monitoring."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate shift"
    return "significant shift"
