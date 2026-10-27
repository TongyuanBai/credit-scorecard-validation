# Credit scorecard development and validation on Lending Club loans

An application scorecard for consumer loans, built with weight-of-evidence (WOE) binning and logistic regression, and documented with a validation report on discrimination, calibration, rating grades and population stability. A second experiment tests how well bad loans can be predicted when only a small share of loans is labelled, using the sample-selection and label-propagation components of [ADGSSL](https://github.com/TongyuanBai/ADGSSL).

The data are 9,857 Lending Club loans issued in the first quarter of 2016, distributed with the R package [`modeldata`](https://modeldata.tidymodels.org/reference/lending_club.html). A loan is bad (5.25% of loans) if it was charged off, in default or 21-120 days late when the data were extracted. The extraction date is not given, so the outcome has no fixed performance window, and the model predicts this bad flag rather than a 12-month probability of default.

## Results on the holdout sample

70/30 stratified split; 2,958 holdout loans, 155 bad. Binning, variable selection and estimation use the development sample only.

| Measure | Value |
|---|---|
| AUC (95% bootstrap interval) | 0.690 (0.645-0.730) |
| AUC from cross-validation on the development sample | 0.642 |
| Gini | 0.380 |
| KS | 0.295 |
| AUC of Lending Club's own sub-grade (benchmark) | 0.736 |
| AUC of a logistic regression on all 19 standardised features (challenger) | 0.707 |
| Mean predicted / observed bad rate | 5.21% / 5.24% |
| Hosmer-Lemeshow test, 10 groups, 10 degrees of freedom | p = 0.536 |
| Observed bad rate by grade, A (lowest risk) to E | 1.37%, 2.39%, 6.42%, 6.19%, 10.75% (C above D) |
| Grades rejected by a one-sided Jeffreys test at 5% | A (prediction too high), C (prediction too low); same pattern in the development sample |
| Population stability index, development vs holdout | 0.0015 (both samples from the same quarter) |

![ROC and KS curves](reports/figures/roc_ks.png)

The holdout AUC is higher than the development AUC (0.673) and the cross-validated AUC (0.642), so this split appears favourable; 0.64 is the more cautious estimate for new loans of the same kind. The scorecard is 0.017 below the challenger (95% bootstrap interval -0.050 to +0.014) with 5 variables of 2 to 3 classes each. The over-prediction in grade A and the under-prediction in grade C appear in both samples, which points to a lack of fit in those score ranges rather than chance. The full report, with variable screening, coefficients, the calibration plot, the grade table and stability indices, is in [`reports/validation_report.md`](reports/validation_report.md).

## Method

1. **Candidate variables.** 18 application variables. Excluded: Lending Club's sub-grade and interest rate, which are outputs of the lender's own risk assessment (the sub-grade is kept as a benchmark); income verification status, which the lender decided and which would give fewer points to applicants whose income was checked; and the borrower's state, to reduce the risk of geographic proxies for protected characteristics (no fair-lending analysis was carried out).
2. **Coarse classing.** Numeric variables are split at up to 20 quantiles (cut points at observed values). Adjacent classes are merged until the bad rate is monotonic, until each class holds at least 5% of the sample and at most 6 classes remain, and while two neighbouring classes do not differ significantly. Repeated 5-fold cross-validation of the whole procedure on the development sample did not distinguish between the thresholds tried, so the most parsimonious one was taken (chi-square of 3.84, the 5% level). Missing values form their own class. WOE = ln(%bad / %good).
3. **Variable selection.** Information value of at least 0.02; absolute correlation between WOE variables of at most 0.6 (the variable with the higher IV is kept); backward elimination of variables with a wrong-signed coefficient or a Wald p-value above 0.05. A variable dropped for correlation is offered again if the variable that blocked it is eliminated. Five variables remain, with variance inflation factors between 1.00 and 1.09.
4. **Scorecard.** Logistic regression on the WOE variables, scaled to 600 points at good:bad odds of 50:1 with 20 points to double the odds. Points are rounded to integers; after rounding, the mean development score moves by 0.06 points.
5. **Validation.** AUC, Gini and KS with stratified bootstrap intervals; comparison with the lender's sub-grade and with an unbinned logistic regression; Hosmer-Lemeshow test, Brier score and calibration by decile; Jeffreys tests in both directions and a one-sided binomial test by grade; PSI of the score and CSI of each variable.

| Variable | Description | IV | Points |
|---|---|---|---|
| `open_il_12m` | Instalment accounts opened in the past 12 months | 0.178 | -15 to 8 |
| `inq_last_6mths` | Credit inquiries in the past 6 months | 0.107 | -13 to 7 |
| `all_util` | Balance to credit limit on all trades | 0.092 | -5 to 9 |
| `inq_fi` | Personal finance inquiries | 0.074 | -9 to 2 |
| `funded_amnt` | Loan amount | 0.023 | -11 to 2 |

Points table: [`reports/tables/scorecard_points.csv`](reports/tables/scorecard_points.csv).

## Predicting bad loans with few labels

ADGSSL was developed for food adulteration detection with limited labels. It has two components. SJUS chooses which samples to label within a budget, using the minimum spanning tree of the data. Measure propagation (MP) spreads the labels over a nearest-neighbour graph. Both are re-implemented in [`scorecard/label_budget.py`](scorecard/label_budget.py) following the ADGSSL code, with two changes in the main analysis. First, the nearest-neighbour graph is symmetrised: the update of the ADGSSL code minimises the MP objective only on a symmetric graph. Second, propagation runs until the class distributions stop changing. MP exactly as in the ADGSSL code is reported as a sensitivity check.

Setting: 20 random pools of 2,000 development loans, of which 2% to 40% can be labelled. The labelled loans are chosen at random or by SJUS. Bad loans are predicted with logistic regression on the labelled loans or with MP, and all AUCs are computed on the same holdout sample.

![Label budget experiment](reports/figures/label_budget.png)

- With 40 to 200 labelled loans (2% to 10% of the pool, on average between 1.45 and 11.15 bad loans), the mean holdout AUC of the four methods was between 0.51 and 0.60, and individual runs ranged from 0.39 to 0.68. Logistic regression with all 6,899 development labels reached 0.707.
- With logistic regression, no difference between SJUS and random selection was detected: mean differences ranged from -0.002 to +0.036, every 95% interval includes zero, and the intervals are wide (up to +0.076 at the 5% budget).
- MP had a lower mean AUC than logistic regression at every budget. With SJUS selection, MP was lower in 15 to 17 of the 20 pools at each budget; with random selection, in 13 of 20 at the three smallest budgets and in 18 and 20 of 20 at the two largest.
- With MP, the mean difference between SJUS and random selection was positive at every budget (+0.001 to +0.033), and SJUS gave the higher AUC in 14 of 20 pools at the 20% and 40% budgets. The evidence for an advantage of SJUS with MP is therefore weak.
- SJUS selected fewer bad loans than random selection at every budget, for example 8.25 against 11.15 on average at the 10% budget.
- With MP exactly as in the ADGSSL code, propagation stopped after 2 iterations in the median run at most budgets, and the mean AUC was between 0.50 and 0.54.

The intervals are t-intervals over the 20 pools. The pools overlap and share one holdout sample, so the intervals are likely too narrow and are not formal tests. Propagation methods assume that neighbouring observations tend to share the same outcome. Among the 5 nearest neighbours of a development loan, the share of bad loans is 6.7% for bad loans and 4.8% for good loans (overall 5.25%), so this assumption holds only weakly here. The graph also weights all 19 features equally, whereas logistic regression estimates a weight for each. Detailed tables: [`reports/label_budget.md`](reports/label_budget.md).

## Repository structure

```text
├── run_scorecard.py          # scorecard development and validation report
├── run_label_budget.py       # label-budget experiment
├── scorecard/
│   ├── data.py               # data loading, candidate and excluded variables
│   ├── binning.py            # coarse classing, WOE and IV
│   ├── model.py              # logistic regression, variable selection, score scaling
│   ├── metrics.py            # AUC, KS, bootstrap, Hosmer-Lemeshow, Jeffreys test, PSI
│   ├── label_budget.py       # SJUS selection and measure propagation
│   └── plots.py              # figures
├── scripts/download_data.py  # downloads the data and verifies the checksum
├── tests/test_scorecard.py   # checks on synthetic data
├── data/README.md            # data source and checksum
└── reports/                  # generated report, tables and figures
```

## Usage

```bash
pip install -r requirements.txt
python scripts/download_data.py   # writes data/lending_club.csv
python run_scorecard.py           # about 30 seconds
python run_label_budget.py        # about 2 minutes; run after run_scorecard.py
python tests/test_scorecard.py    # or, with pytest installed: python -m pytest tests
```

The reported results were produced with the versions in `requirements-exact.txt` (Python 3.13). The label-budget results use spectral clustering, k-means and iterative solvers, and can change slightly with other library versions.

## Limitations

The extract covers one quarter of originations and has no application dates, so the model is validated on a random holdout sample rather than an out-of-time sample, and the stability indices measure sampling variation only. The outcome was recorded at an unspecified date, so loans of different ages are pooled, and the calibration tests check agreement on comparable loans rather than a 12-month PD. Only funded loans are observed and no reject inference is attempted.

## License

MIT License (see `LICENSE`). The data are not redistributed here; see `data/README.md`.
