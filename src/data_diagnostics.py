"""Data diagnostics for the Bitcoin volatility thesis.

This script downloads daily BTC-USD prices, constructs percentage log returns,
and computes preliminary diagnostics used in the data section. In particular,
it reports both excess kurtosis and Pearson kurtosis to avoid ambiguity:

- pandas.Series.kurt() returns excess kurtosis, with normal benchmark 0;
- Pearson kurtosis is excess kurtosis + 3, with normal benchmark 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


TICKER = "BTC-USD"
START_DATE = "2016-01-01"
END_DATE = "2026-06-01"

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
DATA_CLEAN = ROOT / "data_clean"
OUTPUT_TABLES = ROOT / "output" / "tables"


def ensure_directories() -> None:
    """Create the output directories used by the script."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)


def download_btc_prices() -> pd.Series:
    """Download BTC-USD closing prices from Yahoo Finance."""
    btc = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    btc.to_csv(DATA_RAW / "btc_yfinance.csv")

    # Recent yfinance versions can return MultiIndex columns.
    if isinstance(btc.columns, pd.MultiIndex):
        close = btc["Close"].iloc[:, 0]
    else:
        close = btc["Close"]

    return close.dropna().rename("price")


def construct_log_returns(price: pd.Series) -> pd.DataFrame:
    """Construct daily log returns in percentage points."""
    log_return = (100 * np.log(price / price.shift(1))).dropna().rename("log_return")
    data = pd.concat([price, log_return], axis=1).dropna()
    data.to_csv(DATA_CLEAN / "btc_daily_returns.csv")
    return data


def compute_descriptive_statistics(returns: pd.Series) -> pd.DataFrame:
    """Compute descriptive statistics with explicit kurtosis definitions."""
    excess_kurtosis = returns.kurt()  # pandas default: Fisher/excess kurtosis
    pearson_kurtosis = excess_kurtosis + 3

    stats = pd.DataFrame(
        {
            "Statistic": [
                "Observations",
                "Mean",
                "Median",
                "Standard deviation",
                "Minimum",
                "Maximum",
                "Skewness",
                "Excess kurtosis",
                "Pearson kurtosis",
            ],
            "Value": [
                returns.count(),
                returns.mean(),
                returns.median(),
                returns.std(),
                returns.min(),
                returns.max(),
                returns.skew(),
                excess_kurtosis,
                pearson_kurtosis,
            ],
        }
    )

    stats.to_csv(OUTPUT_TABLES / "descriptive_statistics.csv", index=False)
    stats.to_latex(
        OUTPUT_TABLES / "descriptive_statistics.tex",
        index=False,
        float_format="%.4f",
    )
    return stats


def run_stationarity_test(returns: pd.Series) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test."""
    adf_stat, p_value, used_lags, nobs, critical_values, _ = adfuller(
        returns, autolag="AIC"
    )

    results = pd.DataFrame(
        {
            "Statistic": [
                "ADF statistic",
                "p-value",
                "Used lags",
                "Number of observations",
                "1% critical value",
                "5% critical value",
                "10% critical value",
            ],
            "Value": [
                adf_stat,
                p_value,
                used_lags,
                nobs,
                critical_values["1%"],
                critical_values["5%"],
                critical_values["10%"],
            ],
        }
    )

    results.to_csv(OUTPUT_TABLES / "adf_test.csv", index=False)
    return results


def run_ljung_box_tests(returns: pd.Series) -> pd.DataFrame:
    """Run Ljung-Box tests on returns and squared returns."""
    lags = [10, 20, 30]
    lb_returns = acorr_ljungbox(returns, lags=lags, return_df=True)
    lb_squared_returns = acorr_ljungbox(returns**2, lags=lags, return_df=True)

    results = pd.concat(
        {
            "log_return": lb_returns,
            "squared_log_return": lb_squared_returns,
        },
        axis=1,
    )

    results.to_csv(OUTPUT_TABLES / "ljung_box_tests.csv")
    return results


def run_arch_lm_test(returns: pd.Series, nlags: int = 10) -> pd.DataFrame:
    """Run the ARCH-LM test on demeaned returns."""
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns - returns.mean(), nlags=nlags)

    results = pd.DataFrame(
        {
            "Statistic": [
                "Number of lags",
                "ARCH-LM statistic",
                "ARCH-LM p-value",
                "F-statistic",
                "F-test p-value",
            ],
            "Value": [nlags, lm_stat, lm_pvalue, f_stat, f_pvalue],
        }
    )

    results.to_csv(OUTPUT_TABLES / "arch_lm_test.csv", index=False)
    return results


def main() -> None:
    ensure_directories()
    price = download_btc_prices()
    data = construct_log_returns(price)
    returns = data["log_return"]

    descriptive_stats = compute_descriptive_statistics(returns)
    adf_results = run_stationarity_test(returns)
    ljung_box_results = run_ljung_box_tests(returns)
    arch_lm_results = run_arch_lm_test(returns)

    print("\nDescriptive statistics")
    print(descriptive_stats.to_string(index=False))

    print("\nADF test")
    print(adf_results.to_string(index=False))

    print("\nLjung-Box tests")
    print(ljung_box_results)

    print("\nARCH-LM test")
    print(arch_lm_results.to_string(index=False))


if __name__ == "__main__":
    main()
