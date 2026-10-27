"""Develop the application scorecard and write the validation report.

Usage (from the repository root):
    python run_scorecard.py

Outputs:
    reports/validation_report.md   validation report
    reports/tables/*.csv           tables used in the report
    reports/figures/*.png          figures used in the report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from scorecard import binning, data, model, plots
from scorecard.label_budget import feature_matrix
from scorecard import metrics as M

SEED = 2026
HOLDOUT_SHARE = 0.30
N_BOOT = 1000
N_GRADES = 5
HL_GROUPS = 10
CHI2_GRID = [0.0, 1.0, 2.71, 3.84]  # 3.84 = adjacent classes differ at the 5% level
CV_REPEATS, CV_FOLDS = 3, 5
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = floatfmt.get(c)
            if callable(f):
                cells.append(f(v))
            elif f and isinstance(v, (float, np.floating)):
                cells.append(format(v, f))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def integer_cuts(score: np.ndarray, n_groups: int) -> np.ndarray:
    """Quantile cut points of an integer score, rounded up to integers.
    A score belongs to the group above a cut when it is >= the cut."""
    return np.unique(np.ceil(np.quantile(score, np.linspace(0, 1, n_groups + 1)[1:-1])))


def pipeline_auc(Xa, ya, Xb, yb, min_chi2: float) -> float:
    """Binning, selection and fit on (Xa, ya); AUC of the linear predictor on (Xb, yb)."""
    bins = binning.bin_all(Xa, ya, data.NUMERIC, data.CATEGORICAL, min_chi2=min_chi2)
    iv = pd.Series({k: b.iv for k, b in bins.items()})
    woe = pd.DataFrame({k: b.transform(Xa[k]) for k, b in bins.items()})
    selected, _ = model.select_variables(woe, ya, iv)
    fit = model.fit_logit(woe[selected].to_numpy(), ya, selected)
    woe_b = np.column_stack([bins[v].transform(Xb[v]) for v in selected])
    return M.auc(yb, fit.linear_predictor(woe_b))


def choose_min_chi2(Xd, yd) -> tuple[float, pd.DataFrame]:
    """Repeated stratified k-fold CV on the development sample. The largest
    threshold (the most parsimonious binning) whose mean AUC is within one
    standard error of the best is kept (one-standard-error rule)."""
    rows = []
    for chi in CHI2_GRID:
        aucs = []
        for rep in range(CV_REPEATS):
            for tr, te in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=rep).split(Xd, yd):
                aucs.append(pipeline_auc(Xd.iloc[tr].reset_index(drop=True), yd[tr], Xd.iloc[te].reset_index(drop=True), yd[te], chi))
        aucs = np.array(aucs)
        rows.append({"min_chi2": chi, "cv_auc_mean": aucs.mean(), "cv_auc_sd": aucs.std(ddof=1), "cv_auc_se": aucs.std(ddof=1) / np.sqrt(len(aucs))})
    cv = pd.DataFrame(rows)
    best = cv.loc[cv["cv_auc_mean"].idxmax()]
    eligible = cv[cv["cv_auc_mean"] >= best["cv_auc_mean"] - best["cv_auc_se"]]
    return float(eligible["min_chi2"].max()), cv


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # data and split
    # ------------------------------------------------------------------
    X, y = data.load()
    Xd, Xh, yd, yh = train_test_split(X, y, test_size=HOLDOUT_SHARE, stratify=y, random_state=SEED)
    Xd, Xh = Xd.reset_index(drop=True), Xh.reset_index(drop=True)

    # ------------------------------------------------------------------
    # binning, IV and variable selection (development sample only)
    # ------------------------------------------------------------------
    min_chi2, cv_tab = choose_min_chi2(Xd, yd)
    cv_tab.to_csv(TABLES / "binning_cv.csv", index=False)
    bins = binning.bin_all(Xd, yd, data.NUMERIC, data.CATEGORICAL, min_chi2=min_chi2)
    iv = pd.Series({k: b.iv for k, b in bins.items()}).sort_values(ascending=False)
    woe_d = pd.DataFrame({k: b.transform(Xd[k]) for k, b in bins.items()})
    selected, sel_log = model.select_variables(woe_d, yd, iv)

    iv_table = pd.DataFrame(
        {
            "variable": iv.index,
            "description": [data.DESCRIPTIONS.get(v, v) for v in iv.index],
            "bins": [len(bins[v].table) for v in iv.index],
            "iv": iv.values,
            "selected": ["yes" if v in selected else "" for v in iv.index],
        }
    )
    iv_table.to_csv(TABLES / "information_value.csv", index=False)
    sel_log.to_frame().to_csv(TABLES / "selection_log.csv", index=False)
    pd.concat({v: bins[v].table for v in iv.index}, names=["variable", "row"]).reset_index(level="row", drop=True).to_csv(TABLES / "bins_all_variables.csv")

    # ------------------------------------------------------------------
    # final model and scorecard
    # ------------------------------------------------------------------
    fit = model.fit_logit(woe_d[selected].to_numpy(), yd, selected)
    vifs = model.vif(woe_d[selected].to_numpy(), selected)
    coef = fit.summary()
    coef["iv"] = coef["variable"].map(iv)
    coef["vif"] = coef["variable"].map(vifs)
    coef.to_csv(TABLES / "coefficients.csv", index=False)

    card = model.Scorecard(bins, selected, fit, model.ScoreScale())
    card.center_rounding(Xd)
    points = card.points_table()
    points.to_csv(TABLES / "scorecard_points.csv", index=False)
    rounding_shift = card.rounding_shift(Xd)
    exact_pred_d = card.scale.score_to_pd(card.exact_score(Xd))

    score_d, score_h = card.score(Xd), card.score(Xh)
    pd_d, pd_h = card.predict_bad_rate(Xd), card.predict_bad_rate(Xh)
    bench_h = data.sub_grade_rank(Xh[data.BENCHMARK])

    # ------------------------------------------------------------------
    # discrimination
    # ------------------------------------------------------------------
    disc_rows = []
    for name, yy, ss in [("Development", yd, score_d), ("Holdout", yh, score_h)]:
        a = M.auc(yy, -ss)
        k = M.ks(yy, -ss)
        a_ci = M.bootstrap_ci(M.auc, yy, -ss, n_boot=N_BOOT, seed=SEED)
        k_ci = M.bootstrap_ci(M.ks, yy, -ss, n_boot=N_BOOT, seed=SEED)
        disc_rows.append(
            {
                "sample": name,
                "loans": len(yy),
                "bads": int(yy.sum()),
                "AUC": a,
                "AUC 95% CI": f"{a_ci[0]:.3f}-{a_ci[1]:.3f}",
                "Gini": 2 * a - 1,
                "KS": k,
                "KS 95% CI": f"{k_ci[0]:.3f}-{k_ci[1]:.3f}",
            }
        )
    disc = pd.DataFrame(disc_rows)
    disc.to_csv(TABLES / "discrimination.csv", index=False)

    auc_h, auc_b = M.auc(yh, -score_h), M.auc(yh, bench_h)
    diff_ci = M.bootstrap_ci(lambda yy, a, b: M.auc(yy, a) - M.auc(yy, b), yh, -score_h, bench_h, n_boot=N_BOOT, seed=SEED)
    ks_h = M.ks(yh, -score_h)

    # challenger: L2 logistic regression on all candidate variables (standardised, no binning)
    F = feature_matrix(pd.concat([Xd, Xh], ignore_index=True), data.NUMERIC, data.CATEGORICAL, reference=Xd)
    Fd, Fh = F.iloc[: len(Xd)].to_numpy(), F.iloc[len(Xd):].to_numpy()
    scaler = StandardScaler().fit(Fd)
    challenger = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(Fd), yd)
    chal_h = challenger.predict_proba(scaler.transform(Fh))[:, 1]
    auc_c = M.auc(yh, chal_h)
    chal_ci = M.bootstrap_ci(lambda yy, a, b: M.auc(yy, a) - M.auc(yy, b), yh, -score_h, chal_h, n_boot=N_BOOT, seed=SEED)

    # ------------------------------------------------------------------
    # calibration on the holdout sample
    # ------------------------------------------------------------------
    hl_stat, hl_p, hl_tab = M.hosmer_lemeshow(yh, pd_h, groups=HL_GROUPS, df=HL_GROUPS)
    jb = [M.jeffreys(int(o), int(n), 0.0)[:2] for o, n in zip(hl_tab["observed"], hl_tab["n"])]
    hl_tab["jeffreys_low"] = [b[0] for b in jb]
    hl_tab["jeffreys_high"] = [b[1] for b in jb]
    hl_tab["group"] = hl_tab["group"] + 1
    hl_tab.to_csv(TABLES / "calibration_deciles.csv", index=False)
    citl_p = stats.binomtest(int(yh.sum()), len(yh), float(pd_h.mean())).pvalue
    brier_h = M.brier(yh, pd_h)
    brier_ref = M.brier(yh, np.full(len(yh), yd.mean()))
    over = hl_tab[hl_tab["mean_predicted"] > hl_tab["jeffreys_high"]]
    under = hl_tab[hl_tab["mean_predicted"] < hl_tab["jeffreys_low"]]

    # ------------------------------------------------------------------
    # rating grades
    # ------------------------------------------------------------------
    cuts = integer_cuts(score_d, N_GRADES)
    grade_names = [chr(ord("A") + i) for i in range(len(cuts) + 1)]  # A = lowest risk

    def to_grade(score):
        # number of cuts <= score; the highest scores get grade A
        return np.array(grade_names[::-1])[np.searchsorted(cuts, score, side="right")]

    grade_d, grade_h = to_grade(score_d), to_grade(score_h)
    bounds = [-np.inf] + list(cuts) + [np.inf]
    score_ranges = {}
    for i, g in enumerate(grade_names[::-1]):
        lo_s, hi_s = bounds[i], bounds[i + 1]
        lo_txt = "" if np.isinf(lo_s) else f">= {lo_s:.0f}"
        hi_txt = "" if np.isinf(hi_s) else f"< {hi_s:.0f}"
        score_ranges[g] = " and ".join(t for t in (lo_txt, hi_txt) if t)
    bt = M.grade_backtest(yh, pd_h, grade_h, grade_names)
    bt.insert(1, "score_range", bt["grade"].map(score_ranges))
    bt_dev = M.grade_backtest(yd, pd_d, grade_d, grade_names)
    bt.to_csv(TABLES / "grade_backtest_holdout.csv", index=False)
    bt_dev.to_csv(TABLES / "grade_backtest_development.csv", index=False)
    too_low = bt[bt["p_too_low"] < 0.05]["grade"].tolist()
    too_high = bt[bt["p_too_high"] < 0.05]["grade"].tolist()

    # ------------------------------------------------------------------
    # stability
    # ------------------------------------------------------------------
    s_cuts = integer_cuts(score_d, 10)
    s_idx_d = np.searchsorted(s_cuts, score_d, side="right")
    s_idx_h = np.searchsorted(s_cuts, score_h, side="right")
    k = len(s_cuts) + 1
    s_bounds = [-np.inf] + list(s_cuts) + [np.inf]
    s_labels = [f"[{s_bounds[i]:.0f}, {s_bounds[i + 1]:.0f})" for i in range(k)]
    counts_d, counts_h = np.bincount(s_idx_d, minlength=k), np.bincount(s_idx_h, minlength=k)
    psi_tab = M.psi_table(counts_d, counts_h, s_labels)
    psi_score = M.psi_from_counts(counts_d, counts_h)
    psi_tab.to_csv(TABLES / "psi_score.csv", index=False)
    csi = pd.DataFrame(
        {
            "variable": selected,
            "csi": [M.psi_from_counts(binning.bin_counts(bins[v], Xd[v]), binning.bin_counts(bins[v], Xh[v])) for v in selected],
        }
    )
    csi["assessment"] = csi["csi"].map(M.stability_label)
    csi.to_csv(TABLES / "csi.csv", index=False)

    # ------------------------------------------------------------------
    # figures
    # ------------------------------------------------------------------
    plots.woe_panels(bins, selected, {v: v for v in selected}, FIGURES / "woe_bins.png")
    plots.roc_ks(yh, -score_h, bench_h, auc_h, auc_b, ks_h, FIGURES / "roc_ks.png")
    plots.calibration(hl_tab, (hl_tab["jeffreys_low"].to_numpy(), hl_tab["jeffreys_high"].to_numpy()), FIGURES / "calibration.png")
    plots.score_distribution(score_h, yh, cuts, FIGURES / "score_distribution.png")

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    removed = sel_log.to_frame()
    coef_md = coef.copy()
    coef_md["p_value"] = coef_md["p_value"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    rate = lambda v: pct(v, 2)  # noqa: E731

    lines = []
    w = lines.append
    w("# Validation report: Lending Club application scorecard\n")
    w(f"Generated by `run_scorecard.py`. All results refer to one stratified random split (seed {SEED}). The bootstrap intervals describe "
      "the sampling variation of the holdout estimates for the fitted model; they do not include the variation that re-fitting on a different split would add.\n")

    w("## 1. Data and outcome\n")
    w(f"- Source: Lending Club loans issued in the first quarter of 2016, as distributed in the R package `modeldata` (`lending_club`). "
      f"{len(y):,} loans, {int(y.sum()):,} bad ({pct(y.mean(), 2)}).")
    w("- Outcome: a loan is **bad** if it was charged off, in default or 21-120 days late when the data were extracted, and good otherwise. "
      "The extraction date is not given, so the outcome has no fixed performance window and is not a 12-month default flag. "
      "The model therefore estimates the probability of this bad flag, referred to below as the predicted bad rate.")
    w(f"- Split: stratified random split into a development sample ({len(yd):,} loans, {int(yd.sum())} bad) and a holdout sample "
      f"({len(yh):,} loans, {int(yh.sum())} bad). Binning, variable selection and estimation use the development sample only.")
    w("- Variables kept out of the candidate set:")
    for k_, v_ in data.EXCLUDED.items():
        w(f"  - `{k_}`: {v_}.")
    w("- The extract has no application dates, so no out-of-time sample can be formed. The holdout comes from the same quarter as the development sample.\n")

    w("## 2. Variable screening and selection\n")
    w(f"{len(iv)} candidate variables were binned with monotonic coarse classes: at least 5% of the development sample per class, at most 6 classes, "
      f"and adjacent classes merged while the chi-square statistic comparing their bad rates is below {min_chi2:.2f}. "
      "Selection then used an information value of at least 0.02, a maximum absolute correlation of 0.6 between WOE-transformed variables "
      "(the variable with the higher IV is kept), and backward elimination of variables with a wrong-signed coefficient or a Wald p-value above 0.05. "
      "A variable dropped by the correlation filter was offered again if the variable that blocked it was later eliminated.\n")
    best_cv = cv_tab.loc[cv_tab["cv_auc_mean"].idxmax()]
    all_within = bool((cv_tab["cv_auc_mean"] >= best_cv["cv_auc_mean"] - best_cv["cv_auc_se"]).all())
    w(f"Candidate merging thresholds were compared by {CV_REPEATS} repetitions of {CV_FOLDS}-fold cross-validation of the whole procedure (binning, selection and fit) "
      "on the development sample, with the rule of keeping the largest threshold whose mean AUC is within one standard error of the best:\n")
    w(md_table(cv_tab, {"min_chi2": ".2f", "cv_auc_mean": ".4f", "cv_auc_sd": ".4f", "cv_auc_se": ".4f"}))
    w("")
    if all_within:
        w(f"The cross-validation did not distinguish between the thresholds: all mean AUCs lie within one standard error of the best, so the rule selected the "
          f"most parsimonious value in the grid ({min_chi2:.2f}). The standard errors treat the {CV_REPEATS * CV_FOLDS} folds as independent and are therefore too small.\n")
    else:
        w(f"Selected threshold: {min_chi2:.2f}. The standard errors treat the {CV_REPEATS * CV_FOLDS} folds as independent and are therefore too small.\n")
    w(md_table(iv_table[["variable", "description", "bins", "iv", "selected"]], {"iv": ".4f"}))
    w("")
    w("Steps after the IV filter:\n")
    w(md_table(removed[removed["step"] != "IV filter"][["step", "variable", "reason"]]))
    w("")
    w("![WOE by bin](figures/woe_bins.png)\n")

    w("## 3. Model and scorecard\n")
    w(f"Logistic regression of the bad flag on the WOE of the {len(selected)} selected variables. With WOE = ln(%bad/%good), all coefficients are expected to be positive.\n")
    w(md_table(coef_md[["variable", "coef", "std_err", "z", "p_value", "iv", "vif"]], {"coef": ".3f", "std_err": ".3f", "z": ".2f", "iv": lambda v: "" if pd.isna(v) else f"{v:.3f}", "vif": lambda v: "" if pd.isna(v) else f"{v:.2f}"}))
    w("")
    sc = card.scale
    w(f"Scaling: {sc.base_score:.0f} points at good:bad odds of {sc.base_odds:.0f}:1, {sc.pdo:.0f} points to double the odds. "
      "Attribute points are rounded to integers, and the base points are shifted by the mean rounding error of the attribute points before they are rounded themselves. "
      f"After rounding, the mean development score differs from the unrounded score by {rounding_shift:+.2f} points, and the mean predicted bad rate on the development sample "
      f"is {pct(pd_d.mean(), 2)} (unrounded {pct(exact_pred_d.mean(), 2)}; observed {pct(yd.mean(), 2)}). The points table is in `tables/scorecard_points.csv`.\n")

    w("## 4. Discrimination\n")
    w(md_table(disc, {"AUC": ".3f", "Gini": ".3f", "KS": ".3f"}))
    w("")
    w(f"Intervals: percentile bootstrap, {N_BOOT} resamples stratified by outcome.\n")
    w(f"The holdout AUC ({auc_h:.3f}) is higher than both the development AUC ({disc['AUC'].iloc[0]:.3f}) and the cross-validated AUC of the whole procedure "
      f"on the development sample ({float(cv_tab.loc[cv_tab['min_chi2'] == min_chi2, 'cv_auc_mean'].iloc[0]):.3f}, Section 2). This split therefore appears favourable to the model, "
      "and the cross-validated value is the more cautious estimate of discrimination on new loans of the same kind.\n")
    w(f"Benchmark on the holdout sample: Lending Club's sub-grade (A1 to G5, used as an ordinal ranking) has an AUC of {auc_b:.3f}. "
      f"The difference scorecard minus sub-grade is {auc_h - auc_b:+.3f} (95% bootstrap interval {diff_ci[0]:+.3f} to {diff_ci[1]:+.3f}). "
      "The sub-grade was assigned by the lender using information that is not in this extract, so it is a reference point rather than a like-for-like competitor.\n")
    w(f"Challenger: L2-regularised logistic regression (C = 1) on all {Fd.shape[1]} standardised candidate features without binning "
      f"(amounts log-transformed; employment length imputed with the median plus a missing indicator). Holdout AUC {auc_c:.3f}; "
      f"scorecard minus challenger {auc_h - auc_c:+.3f} (95% bootstrap interval {chal_ci[0]:+.3f} to {chal_ci[1]:+.3f}). "
      f"The scorecard uses {len(selected)} variables with {min(len(bins[v].table) for v in selected)} to {max(len(bins[v].table) for v in selected)} classes each.\n")
    w("![ROC and KS](figures/roc_ks.png)\n")
    w("![Score distribution](figures/score_distribution.png)\n")

    w("## 5. Calibration on the holdout sample\n")
    w("The tests below are the binomial and Jeffreys tests used in PD backtesting, applied to the bad flag defined in Section 1. Because the holdout is a random "
      "sample from the same quarter and the outcome has no fixed horizon, they check calibration on comparable loans; they are not a backtest of a 12-month PD.\n")
    w(f"- Overall: mean predicted bad rate {rate(pd_h.mean())} against an observed bad rate of {rate(yh.mean())} (two-sided exact binomial test, p = {citl_p:.3f}).")
    w(f"- Hosmer-Lemeshow statistic over {HL_GROUPS} groups of the predicted bad rate: {hl_stat:.2f}, p = {hl_p:.3f} "
      f"({HL_GROUPS} degrees of freedom, the reference distribution for an independent validation sample).")
    w(f"- Brier score {brier_h:.4f}; a constant prediction equal to the development bad rate gives {brier_ref:.4f}.")
    if len(over) or len(under):
        parts = []
        for _, r in over.iterrows():
            parts.append(f"decile {int(r['group'])}: predicted {rate(r['mean_predicted'])}, observed {rate(r['observed_bad_rate'])} (interval {rate(r['jeffreys_low'])} to {rate(r['jeffreys_high'])})")
        for _, r in under.iterrows():
            parts.append(f"decile {int(r['group'])}: predicted {rate(r['mean_predicted'])}, observed {rate(r['observed_bad_rate'])} (interval {rate(r['jeffreys_low'])} to {rate(r['jeffreys_high'])})")
        w("- Deciles whose 95% Jeffreys interval for the observed bad rate excludes the mean prediction: " + "; ".join(parts) + ".")
    top = hl_tab.iloc[-1]
    w(f"- Highest decile: predicted {rate(top['mean_predicted'])}, observed {rate(top['observed_bad_rate'])}.\n")
    w("![Calibration](figures/calibration.png)\n")
    w(f"Rating grades: {len(grade_names)} grades cut at the quintiles of the development score (A = lowest risk). For each grade the observed bad rate "
      "on the holdout sample is compared with the mean prediction for the same loans. `p_too_low` and `p_too_high` are the two one-sided Jeffreys tests; "
      "`p_binomial_too_low` is the one-sided exact binomial test.\n")
    w(md_table(bt, {"observed_bad_rate": rate, "mean_predicted": rate, "jeffreys_low": rate, "jeffreys_high": rate, "p_too_low": ".3f", "p_too_high": ".3f", "p_binomial_too_low": ".3f"}))
    w("")
    summary_grades = []
    summary_grades.append(f"prediction too low at the 5% level: {', '.join(too_low) if too_low else 'none'}")
    summary_grades.append(f"prediction too high at the 5% level: {', '.join(too_high) if too_high else 'none'}")
    obs = bt["observed_bad_rate"].to_numpy()
    reversals = [f"{bt['grade'].iloc[i]} ({rate(obs[i])}) and {bt['grade'].iloc[i + 1]} ({rate(obs[i + 1])})" for i in range(len(obs) - 1) if obs[i + 1] <= obs[i]]
    order_txt = f"Observed bad rates go from {rate(obs[0])} in grade A to {rate(obs[-1])} in grade {bt['grade'].iloc[-1]}"
    order_txt += f"; they are not in increasing order for grades {'; '.join(reversals)}." if reversals else ", increasing from grade to grade."
    w(f"Grades with {summary_grades[0]}; {summary_grades[1]}. {order_txt}\n")
    dev_low = bt_dev[bt_dev["p_too_low"] < 0.05]["grade"].tolist()
    dev_high = bt_dev[bt_dev["p_too_high"] < 0.05]["grade"].tolist()
    repeated = [g for g in too_low if g in dev_low] + [g for g in too_high if g in dev_high]
    if repeated:
        detail = "; ".join(
            f"grade {g}: observed {rate(bt_dev.loc[bt_dev['grade'] == g, 'observed_bad_rate'].iloc[0])}, predicted {rate(bt_dev.loc[bt_dev['grade'] == g, 'mean_predicted'].iloc[0])}"
            for g in repeated
        )
        w(f"The same deviations appear in the development sample ({detail}; table in `tables/grade_backtest_development.csv`). Ten one-sided tests would produce "
          "about 0.5 rejections at the 5% level by chance, but a deviation repeated in both samples points to a lack of fit of the additive model in these score ranges. "
          "A common remedy is to assign each grade the bad rate observed for it in the development sample (a master scale) and to test those grade-level rates on new data.\n")
    else:
        w(f"With {2 * len(bt)} one-sided tests, about {0.05 * 2 * len(bt):.1f} rejections at the 5% level are expected by chance.\n")

    w("## 6. Stability\n")
    w(f"Population stability index of the score, development against holdout, over {k} bins cut at the deciles of the development score (fewer than 10 because of tied scores): "
      f"{psi_score:.4f} ({M.stability_label(psi_score)}). "
      "Characteristic stability index (CSI) of each model variable over its bins:\n")
    w(md_table(csi, {"csi": ".4f"}))
    w("")
    w("Thresholds: below 0.10 stable, 0.10-0.25 moderate shift, above 0.25 significant shift. Both samples are random draws from the same quarter, so these values "
      "measure sampling variation only and say nothing about stability over time. In production, the same calculation would be run on each new application cohort.\n")

    w("## 7. Limitations\n")
    w("- One quarter of originations and a single random split; no out-of-time test.")
    w("- The outcome was recorded at an unspecified date after origination, so loans of different ages are pooled and some loans labelled good may still go bad.")
    w("- Only funded applications are observed; no reject inference is attempted.")
    w(f"- {int(y.sum())} bads in total; grade-level results rest on few bads.")
    w("- The extract contains a subset of the variables available at application, which also limits the comparison with the lender's sub-grade.")
    w("- Excluding `addr_state` does not remove the possibility that other variables act as proxies for protected characteristics; no fair-lending analysis was carried out.\n")

    (REPORTS / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "selected": selected,
        "auc_holdout": auc_h,
        "ks_holdout": ks_h,
        "auc_subgrade": auc_b,
        "auc_challenger": auc_c,
        "psi": psi_score,
        "hl_p": hl_p,
        "rounding_shift": rounding_shift,
        "min_chi2": min_chi2,
        "grades_too_low": too_low,
        "grades_too_high": too_high,
    }
    print(pd.Series(summary).to_string())
    print(f"Report written to {REPORTS / 'validation_report.md'}")


if __name__ == "__main__":
    main()
