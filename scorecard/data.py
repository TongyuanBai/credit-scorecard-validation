"""Loading the Lending Club extract and defining the candidate variables."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("SCORECARD_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
DATA_FILE = "lending_club.csv"

# Variables that are outputs of the lender's own risk assessment, or that are
# hard to justify as a reason for decline, are kept out of the scorecard.
# The sub-grade is used as a benchmark.
BENCHMARK = "sub_grade"
EXCLUDED = {
    "sub_grade": "the lender's own risk grade; used as the benchmark",
    "int_rate": "priced from the lender's risk grade",
    "verification_status": (
        "whether income was verified was decided by the lender, so the variable may partly reflect the lender's own "
        "assessment; verified income has the higher bad rate, which would give fewer points to applicants whose income was checked"
    ),
    "addr_state": (
        "geographic variable; excluded to reduce the risk of proxies for protected characteristics "
        "(no fair-lending analysis of the remaining variables was carried out)"
    ),
}

NUMERIC = [
    "funded_amnt",
    "annual_inc",
    "emp_length",
    "delinq_2yrs",
    "inq_last_6mths",
    "revol_util",
    "acc_now_delinq",
    "open_il_6m",
    "open_il_12m",
    "open_il_24m",
    "total_bal_il",
    "all_util",
    "inq_fi",
    "inq_last_12m",
    "delinq_amnt",
    "num_il_tl",
    "total_il_high_credit_limit",
]
CATEGORICAL = ["term"]

DESCRIPTIONS = {
    "funded_amnt": "Loan amount funded",
    "term": "Loan term (36 or 60 months)",
    "annual_inc": "Self-reported annual income",
    "emp_length": "Employment length in years (10 = 10 or more; missing if not reported)",
    "delinq_2yrs": "Delinquencies (30+ days) in the past 2 years",
    "inq_last_6mths": "Credit inquiries in the past 6 months",
    "revol_util": "Revolving line utilisation (%)",
    "acc_now_delinq": "Accounts on which the borrower is now delinquent",
    "open_il_6m": "Currently active instalment trades",
    "open_il_12m": "Instalment accounts opened in the past 12 months",
    "open_il_24m": "Instalment accounts opened in the past 24 months",
    "total_bal_il": "Total current balance of instalment accounts",
    "all_util": "Balance to credit limit on all trades (%)",
    "inq_fi": "Personal finance inquiries",
    "inq_last_12m": "Credit inquiries in the past 12 months",
    "delinq_amnt": "Past-due amount owed on delinquent accounts",
    "num_il_tl": "Number of instalment accounts",
    "total_il_high_credit_limit": "Total high credit / credit limit of instalment accounts",
}


def load(path: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the predictors and the target (1 = bad, 0 = good)."""
    path = Path(path) if path else DATA_DIR / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first (see data/README.md).")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("rownames", "Unnamed: 0") if c in df.columns])
    y = (df.pop("Class") == "bad").astype(int).to_numpy()
    df["emp_length"] = employment_years(df["emp_length"])
    return df, y


def employment_years(emp_length: pd.Series) -> pd.Series:
    """emp_lt_1 -> 0, emp_1 ... emp_9 -> 1 ... 9, emp_ge_10 -> 10, emp_unk -> missing.

    Treating employment length as an ordered variable lets the binning keep
    neighbouring years together instead of grouping them by bad rate alone.
    """
    s = emp_length.astype(str)
    years = s.str.extract(r"emp_(\d+)$")[0].astype(float)
    years[s == "emp_lt_1"] = 0.0
    years[s == "emp_ge_10"] = 10.0
    return years


def sub_grade_rank(sub_grade: pd.Series) -> np.ndarray:
    """A1 -> 1, A2 -> 2, ..., G5 -> 35 (higher = riskier)."""
    s = sub_grade.astype(str)
    return ((s.str[0].map(ord) - ord("A")) * 5 + s.str[1].astype(int)).to_numpy()
