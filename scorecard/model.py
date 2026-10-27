"""Logistic regression on WOE variables, variable selection and score scaling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .binning import VariableBinning


# ----------------------------------------------------------------------
# maximum likelihood logistic regression (Newton-Raphson)
# ----------------------------------------------------------------------
@dataclass
class LogitFit:
    names: list[str]
    coef: np.ndarray  # intercept first
    se: np.ndarray
    loglik: float
    n_obs: int

    @property
    def z(self) -> np.ndarray:
        return self.coef / self.se

    @property
    def p_value(self) -> np.ndarray:
        return 2 * stats.norm.sf(np.abs(self.z))

    @property
    def aic(self) -> float:
        return 2 * len(self.coef) - 2 * self.loglik

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "variable": ["(Intercept)"] + self.names,
                "coef": self.coef,
                "std_err": self.se,
                "z": self.z,
                "p_value": self.p_value,
            }
        )

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        return self.coef[0] + X @ self.coef[1:]


def fit_logit(X: np.ndarray, y: np.ndarray, names: list[str], max_iter: int = 50, tol: float = 1e-10) -> LogitFit:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    converged = False
    for _ in range(max_iter):
        eta = X1 @ beta
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        grad = X1.T @ (y - p)
        hess = X1.T @ (X1 * w[:, None])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    if not converged:
        warnings.warn(f"Newton-Raphson did not converge in {max_iter} iterations", RuntimeWarning)
    eta = X1 @ beta
    p = 1 / (1 + np.exp(-eta))
    w = p * (1 - p)
    cov = np.linalg.inv(X1.T @ (X1 * w[:, None]))
    loglik = float(np.sum(y * eta - np.logaddexp(0, eta)))
    return LogitFit(list(names), beta, np.sqrt(np.diag(cov)), loglik, len(y))


def vif(X: np.ndarray, names: list[str]) -> pd.Series:
    """Variance inflation factor of each column (regression on the others with intercept)."""
    out = {}
    for j, name in enumerate(names):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ coef
        r2 = 1 - resid.var() / X[:, j].var()
        out[name] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out, name="vif")


# ----------------------------------------------------------------------
# variable selection
# ----------------------------------------------------------------------
@dataclass
class SelectionLog:
    steps: list[dict]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def select_variables(
    woe: pd.DataFrame,
    y: np.ndarray,
    iv: pd.Series,
    min_iv: float = 0.02,
    max_corr: float = 0.6,
    max_p: float = 0.05,
) -> tuple[list[str], SelectionLog]:
    """IV filter, correlation filter and backward elimination.

    Backward elimination removes, one at a time, a variable whose coefficient
    has the wrong sign (non-positive, given WOE = ln(%bad/%good)) or whose
    Wald p-value exceeds ``max_p``. Wrong signs are removed first, then the
    largest p-value.

    A variable dropped by the correlation filter is offered once more if the
    variable that blocked it is later removed and no remaining variable is
    correlated with it above ``max_corr``; it then goes through backward
    elimination again with the others.
    """
    log = []
    kept = [v for v in iv.sort_values(ascending=False).index if iv[v] >= min_iv]
    for v in iv.index:
        if iv[v] < min_iv:
            log.append({"step": "IV filter", "variable": v, "reason": f"IV = {iv[v]:.4f} < {min_iv}"})

    corr = woe[kept].corr().abs()
    selected: list[str] = []
    blocked: list[str] = []
    for v in kept:  # kept is ordered by decreasing IV
        clash = [u for u in selected if corr.loc[v, u] > max_corr]
        if clash:
            blocked.append(v)
            log.append(
                {
                    "step": "Correlation filter",
                    "variable": v,
                    "reason": f"correlation {corr.loc[v, clash[0]]:.2f} with {clash[0]} (higher IV)",
                }
            )
        else:
            selected.append(v)

    def eliminate(current: list[str]) -> list[str]:
        current = list(current)
        while current:
            fit = fit_logit(woe[current].to_numpy(), y, current)
            table = fit.summary().iloc[1:].set_index("variable")
            wrong = table[table["coef"] <= 0]
            if len(wrong):
                v = wrong["p_value"].idxmax()
                reason = f"coefficient {table.loc[v, 'coef']:.3f} has the wrong sign"
            elif table["p_value"].max() > max_p:
                v = table["p_value"].idxmax()
                reason = f"p = {table.loc[v, 'p_value']:.3f} > {max_p}"
            else:
                break
            log.append({"step": "Backward elimination", "variable": v, "reason": reason})
            current.remove(v)
        return current

    selected = eliminate(selected)
    retried: set[str] = set()
    while True:
        candidates = [v for v in blocked if v not in retried and all(corr.loc[v, u] <= max_corr for u in selected)]
        if not candidates:
            break
        v = candidates[0]
        retried.add(v)
        log.append({"step": "Re-entry", "variable": v, "reason": "no remaining variable has a correlation above the limit"})
        selected = eliminate(selected + [v])
    order = {v: i for i, v in enumerate(kept)}
    return sorted(selected, key=order.get), SelectionLog(log)


# ----------------------------------------------------------------------
# score scaling
# ----------------------------------------------------------------------
@dataclass
class ScoreScale:
    """Score = offset + factor * ln(odds of good).

    ``base_score`` points correspond to good:bad odds of ``base_odds``, and the
    odds double every ``pdo`` points.
    """

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_to_pd(self, score: np.ndarray) -> np.ndarray:
        """Probability of the bad outcome for a score (named pd by convention)."""
        log_odds_good = (np.asarray(score, dtype=float) - self.offset) / self.factor
        return 1 / (1 + np.exp(log_odds_good))


@dataclass
class Scorecard:
    binnings: dict[str, VariableBinning]
    variables: list[str]
    fit: LogitFit
    scale: ScoreScale
    base_adjustment: float = 0.0

    def points_table(self) -> pd.DataFrame:
        base = int(round(self.base_points + self.base_adjustment))
        rows = [{"variable": "(Base points)", "bin": "", "woe": np.nan, "points": base}]
        coef = dict(zip(self.fit.names, self.fit.coef[1:]))
        for v in self.variables:
            tab = self.binnings[v].table
            for _, r in tab.iterrows():
                pts = -self.scale.factor * coef[v] * r["woe"]
                rows.append({"variable": v, "bin": r["bin"], "woe": r["woe"], "points": int(round(pts))})
        return pd.DataFrame(rows)

    @property
    def base_points(self) -> float:
        return self.scale.offset - self.scale.factor * self.fit.coef[0]

    def center_rounding(self, df: pd.DataFrame) -> float:
        """Shift the base points by the mean rounding error of the attribute
        points on ``df``. Because the base points are an integer as well, the
        mean score can still move by up to half a point; see rounding_shift."""
        self.base_adjustment = 0.0
        coef = dict(zip(self.fit.names, self.fit.coef[1:]))
        error = np.zeros(len(df))
        for v in self.variables:
            exact = -self.scale.factor * coef[v] * self.binnings[v].transform(df[v])
            error += np.round(exact) - exact
        self.base_adjustment = -float(error.mean())
        return self.base_adjustment

    def exact_score(self, df: pd.DataFrame) -> np.ndarray:
        """Score before rounding: offset - factor * linear predictor."""
        woe = np.column_stack([self.binnings[v].transform(df[v]) for v in self.variables])
        return self.scale.offset - self.scale.factor * self.fit.linear_predictor(woe)

    def rounding_shift(self, df: pd.DataFrame) -> float:
        """Mean of (integer score - exact score) on ``df``."""
        return float(np.mean(self.score(df) - self.exact_score(df)))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Integer score from the rounded points of the scorecard."""
        table = self.points_table()
        total = np.full(len(df), table.iloc[0]["points"], dtype=float)
        for v in self.variables:
            pts = table.loc[table["variable"] == v, "points"].to_numpy()
            total += pts[self.binnings[v].assign(df[v])]
        return total

    def predict_bad_rate(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted probability of the bad outcome implied by the integer score."""
        return self.scale.score_to_pd(self.score(df))
