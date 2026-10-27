"""Coarse classing, weight of evidence (WOE) and information value (IV).

Numeric variables are binned in three steps:

1. Fine classing at up to ``n_fine`` quantiles, taken at observed values (or
   at every distinct value when the variable takes few values).
2. Adjacent fine classes are merged until the bad rate is monotonic in the
   variable. Both directions are tried and the one with the larger IV is kept.
3. Classes holding less than ``min_share`` of the observations are merged into
   the neighbour with the closer bad rate. Then the adjacent pair with the
   smallest chi-square statistic (2 x 2 table of bads and goods) is merged
   until at most ``max_bins`` remain and every adjacent pair has a chi-square
   of at least ``min_chi2`` (3.84 corresponds to the 5% level).

Categorical variables are ordered by bad rate and then go through step 3.
Missing values, if present, form a separate class that is never merged.

WOE is defined as ln(%bad / %good), so a higher WOE means a higher risk and
the logistic regression coefficients of WOE-transformed variables are
expected to be positive. A count of 0.5 is added to every class before the
distributions are computed, which keeps the WOE finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SMOOTHING = 0.5
MISSING_LABEL = "Missing"


@dataclass
class VariableBinning:
    """Binning of one variable, estimated on the development sample."""

    name: str
    kind: str  # "numeric" or "categorical"
    edges: list | None  # numeric: interior cut points, bins are right-closed
    groups: list | None  # categorical: list of lists of category labels
    has_missing: bool
    table: pd.DataFrame
    iv: float

    # ------------------------------------------------------------------
    def assign(self, x: pd.Series) -> np.ndarray:
        """Return the bin index of each observation (missing = last bin)."""
        x = pd.Series(x)
        missing = x.isna().to_numpy()
        idx = np.zeros(len(x), dtype=int)
        n_regular = len(self.table) - int(self.has_missing)
        if self.kind == "numeric":
            values = x.to_numpy(dtype=float)
            idx[~missing] = np.searchsorted(np.asarray(self.edges), values[~missing], side="left")
        else:
            lookup = {cat: g for g, cats in enumerate(self.groups) for cat in cats}
            # Categories not seen in development go to the largest group.
            fallback = int(self.table["n"].iloc[:n_regular].to_numpy().argmax())
            idx[~missing] = [lookup.get(v, fallback) for v in x[~missing].astype(str)]
        if missing.any():
            if not self.has_missing:
                raise ValueError(f"{self.name}: missing values were not present in development")
            idx[missing] = n_regular
        return idx

    def transform(self, x: pd.Series) -> np.ndarray:
        """Replace each value by the WOE of its bin."""
        return self.table["woe"].to_numpy()[self.assign(x)]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _woe_iv(n_bad: np.ndarray, n_good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = len(n_bad)
    dist_bad = (n_bad + SMOOTHING) / (n_bad.sum() + SMOOTHING * k)
    dist_good = (n_good + SMOOTHING) / (n_good.sum() + SMOOTHING * k)
    woe = np.log(dist_bad / dist_good)
    iv_contrib = (dist_bad - dist_good) * woe
    return woe, iv_contrib


def _merge(counts: list[list[float]], members: list[list], i: int) -> None:
    """Merge class i and class i + 1 in place."""
    counts[i] = [counts[i][0] + counts[i + 1][0], counts[i][1] + counts[i + 1][1]]
    members[i] = members[i] + members[i + 1]
    del counts[i + 1]
    del members[i + 1]


def _rate(c: list[float]) -> float:
    return c[1] / c[0] if c[0] > 0 else 0.0


def _chi2(a: list[float], b: list[float]) -> float:
    """Pearson chi-square of the 2 x 2 table (bad, good) x (class a, class b)."""
    obs = np.array([[a[1], a[0] - a[1]], [b[1], b[0] - b[1]]], dtype=float)
    expected = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def _make_monotonic(counts, members, increasing: bool) -> None:
    changed = True
    while changed and len(counts) > 1:
        changed = False
        for i in range(len(counts) - 1):
            r0, r1 = _rate(counts[i]), _rate(counts[i + 1])
            if (increasing and r0 >= r1) or (not increasing and r0 <= r1):
                _merge(counts, members, i)
                changed = True
                break


def _enforce_size_and_count(counts, members, min_count: float, max_bins: int, min_chi2: float = 0.0) -> None:
    # Small classes are merged into the neighbour with the closer bad rate.
    while len(counts) > 1:
        sizes = [c[0] for c in counts]
        i = int(np.argmin(sizes))
        if sizes[i] >= min_count:
            break
        if i == 0:
            j = 0
        elif i == len(counts) - 1:
            j = i - 1
        else:
            left = abs(_rate(counts[i]) - _rate(counts[i - 1]))
            right = abs(_rate(counts[i]) - _rate(counts[i + 1]))
            j = i - 1 if left <= right else i
        _merge(counts, members, j)
    # The most similar adjacent pair is merged until at most max_bins remain
    # and every adjacent pair differs with a chi-square of at least min_chi2.
    while len(counts) > 1:
        stats = [_chi2(counts[i], counts[i + 1]) for i in range(len(counts) - 1)]
        if len(counts) <= max_bins and min(stats) >= min_chi2:
            break
        _merge(counts, members, int(np.argmin(stats)))


def _table(labels, n, n_bad) -> tuple[pd.DataFrame, float]:
    n = np.asarray(n, dtype=float)
    n_bad = np.asarray(n_bad, dtype=float)
    n_good = n - n_bad
    woe, iv_contrib = _woe_iv(n_bad, n_good)
    table = pd.DataFrame(
        {
            "bin": labels,
            "n": n.astype(int),
            "share": n / n.sum(),
            "n_bad": n_bad.astype(int),
            "bad_rate": np.divide(n_bad, n, out=np.zeros_like(n), where=n > 0),
            "woe": woe,
            "iv": iv_contrib,
        }
    )
    return table, float(iv_contrib.sum())


def _fmt(v: float) -> str:
    return f"{v:,.6g}"


def _interval_labels(edges: list[float]) -> list[str]:
    """Labels of right-closed bins: '<= e1', '(e1, e2]', ..., '> ek'."""
    if not edges:
        return ["All values"]
    labels = [f"<= {_fmt(edges[0])}"]
    labels += [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(f"> {_fmt(edges[-1])}")
    return labels


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def bin_numeric(
    x: pd.Series,
    y: np.ndarray,
    name: str,
    n_fine: int = 20,
    min_share: float = 0.05,
    max_bins: int = 6,
    min_chi2: float = 3.84,
) -> VariableBinning:
    x = pd.Series(x).reset_index(drop=True)
    y = np.asarray(y)
    missing = x.isna().to_numpy()
    xv, yv = x[~missing].to_numpy(dtype=float), y[~missing]

    distinct = np.unique(xv)
    if len(distinct) <= n_fine:
        fine_edges = distinct[:-1]
    else:
        # method='lower' keeps every cut point at an observed value
        q = np.quantile(xv, np.linspace(0, 1, n_fine + 1)[1:-1], method="lower")
        fine_edges = np.unique(q)
        fine_edges = fine_edges[fine_edges < distinct.max()]
    fine_idx = np.searchsorted(fine_edges, xv, side="left")
    k = len(fine_edges) + 1
    n = np.bincount(fine_idx, minlength=k).astype(float)
    bad = np.bincount(fine_idx, weights=yv, minlength=k)
    keep = n > 0
    base_counts = [[n[i], bad[i]] for i in range(k) if keep[i]]
    base_members = [[i] for i in range(k) if keep[i]]
    min_count = min_share * len(x)

    best = None
    for increasing in (True, False):
        counts = [c[:] for c in base_counts]
        members = [m[:] for m in base_members]
        _make_monotonic(counts, members, increasing)
        _enforce_size_and_count(counts, members, min_count, max_bins, min_chi2)
        nb = np.array([c[1] for c in counts])
        ng = np.array([c[0] - c[1] for c in counts])
        iv = float(_woe_iv(nb, ng)[1].sum())
        if best is None or iv > best[0]:
            best = (iv, counts, members)
    _, counts, members = best

    # Interior cut points: the upper fine edge of every merged class but the last.
    edges = [float(fine_edges[m[-1]]) for m in members[:-1]]
    labels = _interval_labels(edges)
    n_cls = [c[0] for c in counts]
    bad_cls = [c[1] for c in counts]
    if missing.any():
        labels.append(MISSING_LABEL)
        n_cls.append(float(missing.sum()))
        bad_cls.append(float(y[missing].sum()))
    table, iv = _table(labels, n_cls, bad_cls)
    return VariableBinning(name, "numeric", edges, None, bool(missing.any()), table, iv)


def bin_categorical(
    x: pd.Series,
    y: np.ndarray,
    name: str,
    min_share: float = 0.05,
    max_bins: int = 6,
    min_chi2: float = 3.84,
) -> VariableBinning:
    x = pd.Series(x).reset_index(drop=True)
    y = np.asarray(y)
    missing = x.isna().to_numpy()
    frame = pd.DataFrame({"x": x[~missing].astype(str), "y": y[~missing]})
    stats = frame.groupby("x")["y"].agg(["size", "sum"])
    stats["rate"] = stats["sum"] / stats["size"]
    stats = stats.sort_values("rate")
    counts = [[float(r["size"]), float(r["sum"])] for _, r in stats.iterrows()]
    members = [[cat] for cat in stats.index]
    _enforce_size_and_count(counts, members, min_share * len(x), max_bins, min_chi2)

    labels = [", ".join(m) for m in members]
    n_cls = [c[0] for c in counts]
    bad_cls = [c[1] for c in counts]
    if missing.any():
        labels.append(MISSING_LABEL)
        n_cls.append(float(missing.sum()))
        bad_cls.append(float(y[missing].sum()))
    table, iv = _table(labels, n_cls, bad_cls)
    return VariableBinning(name, "categorical", None, members, bool(missing.any()), table, iv)


def bin_all(
    df: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    **kwargs,
) -> dict[str, VariableBinning]:
    out = {}
    for col in numeric:
        out[col] = bin_numeric(df[col], y, col, **kwargs)
    cat_kwargs = {k: v for k, v in kwargs.items() if k != "n_fine"}
    for col in categorical:
        out[col] = bin_categorical(df[col], y, col, **cat_kwargs)
    return out


def bin_counts(binning: VariableBinning, x: pd.Series) -> np.ndarray:
    """Number of observations of x in each bin of a fitted binning."""
    return np.bincount(binning.assign(x), minlength=len(binning.table))
