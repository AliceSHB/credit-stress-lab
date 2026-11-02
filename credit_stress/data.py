"""Data layer: loan-level credit file + macro series.

Two independent sources are used, and they are deliberately kept separate:

* ``load_loans``  - UCI "default of credit card clients" (30,000 accounts,
  Taiwan, 2005). Used to estimate a loan-level probability of default.
* ``load_macro``  - four FRED series (US). Used to estimate the satellite
  model that links default rates to the macro cycle.

Joining a loan file to a macro series is portfolio specific; the point of the
split is that each link of the chain is fitted on real data rather than
assumed.
"""
from __future__ import annotations

import pathlib

import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

RENAME = {
    "default payment next month": "default",
    "PAY_0": "PAY_1",
}


def load_loans(path: str | None = None) -> pd.DataFrame:
    """Loan-level file with a clean schema and derived features."""
    path = path or DATA / "default of credit card clients.xls"
    df = pd.read_excel(path, header=1)
    df = df.rename(columns=RENAME)
    df = df.drop(columns=["ID"])
    df.columns = [c.strip().upper() for c in df.columns]

    # utilisation: balance relative to the credit limit, and payment behaviour
    df["UTILISATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"].clip(lower=1)
    df["PAY_RATIO"] = df["PAY_AMT1"] / df["BILL_AMT1"].clip(lower=1)
    df["DELINQ_MAX"] = df[[f"PAY_{i}" for i in range(1, 7)]].max(axis=1)
    df["DELINQ_MEAN"] = df[[f"PAY_{i}" for i in range(1, 7)]].mean(axis=1)
    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["PAY_AMT_TREND"] = df["PAY_AMT1"] - df["PAY_AMT6"]
    df["LOG_LIMIT"] = (df["LIMIT_BAL"] + 1).pipe(__import__("numpy").log)
    df["EAD"] = df["BILL_AMT1"].clip(lower=0)
    return df


FRED_SERIES = {
    "UNRATE": "unemployment_rate",
    "GDPC1": "real_gdp",
    "CSUSHPINSA": "house_price_index",
    "DRCCLACBS": "card_delinquency_rate",
}


def _read_fred(name: str) -> pd.Series:
    df = pd.read_csv(DATA / f"fred_{name}.csv")
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def load_macro(path: str | None = None) -> pd.DataFrame:
    """Quarterly macro frame: level, growth and the satellite target."""
    series = {alias: _read_fred(code) for code, alias in FRED_SERIES.items()}
    frame = pd.DataFrame(series).sort_index()
    q = frame.resample("QE").last()
    q["gdp_growth"] = q["real_gdp"].pct_change(4) * 100.0
    q["hpi_growth"] = q["house_price_index"].pct_change(4) * 100.0
    q = q.dropna(subset=["gdp_growth", "hpi_growth"])
    return q[["unemployment_rate", "gdp_growth", "hpi_growth", "card_delinquency_rate"]]


FEATURES = [
    "LIMIT_BAL", "LOG_LIMIT", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_1", "PAY_2", "PAY_3", "DELINQ_MAX", "DELINQ_MEAN",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_TREND",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT_TREND",
    "UTILISATION", "PAY_RATIO",
]
TARGET = "DEFAULT"
