"""
Unit tests for the statistical-validation primitives.

These test the *pure math* (no network/data downloads) so CI runs fast and
deterministically. We verify each statistic against a case with a known answer.
"""
import numpy as np
import pandas as pd
import pytest

from src.research.statistical_validation import (
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    compute_factors,
    newey_west_tstat,
)


# --------------------------------------------------------------------------- #
#  Newey-West t-stat
# --------------------------------------------------------------------------- #
def test_newey_west_zero_mean_not_significant():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, 500))
    mean, t, p = newey_west_tstat(series)
    assert abs(mean) < 0.2
    assert p > 0.05  # a zero-mean noise series should NOT be significant


def test_newey_west_strong_mean_significant():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0.5, 1.0, 500))  # clear positive mean
    mean, t, p = newey_west_tstat(series)
    assert mean > 0.3
    assert t > 2.0
    assert p < 0.01


def test_newey_west_handles_short_series():
    mean, t, p = newey_west_tstat(pd.Series([1.0, 2.0]))
    assert p == 1.0  # too short -> not significant by construction


# --------------------------------------------------------------------------- #
#  Benjamini-Hochberg FDR
# --------------------------------------------------------------------------- #
def test_bh_all_large_pvalues_none_reject():
    pvals = [0.9, 0.8, 0.7, 0.95]
    reject, adj = benjamini_hochberg(pvals)
    assert not reject.any()
    assert np.all(adj >= np.array(pvals) - 1e-9)  # adjusted >= raw


def test_bh_one_tiny_pvalue_rejected():
    pvals = [1e-6, 0.6, 0.7, 0.8, 0.9]
    reject, adj = benjamini_hochberg(pvals)
    assert reject[0]            # the tiny p-value survives correction
    assert reject.sum() == 1


def test_bh_empty_input():
    reject, adj = benjamini_hochberg([])
    assert reject.size == 0 and adj.size == 0


# --------------------------------------------------------------------------- #
#  Bootstrap Sharpe CI
# --------------------------------------------------------------------------- #
def test_bootstrap_sharpe_point_and_ci_ordering():
    rng = np.random.default_rng(7)
    # daily returns with a clearly positive drift (mean >> standard error)
    rets = pd.Series(rng.normal(0.003, 0.01, 750))
    point, lo, hi = bootstrap_sharpe_ci(rets, n_boot=1000)
    assert lo <= point <= hi
    assert point > 0


def test_bootstrap_sharpe_too_few_points():
    point, lo, hi = bootstrap_sharpe_ci(pd.Series([0.01, 0.02]))
    assert (point, lo, hi) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
#  Factor computation (no look-ahead, correct shapes)
# --------------------------------------------------------------------------- #
def _synthetic_ohlcv(n=200, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)
    return df


def test_compute_factors_columns_and_no_lookahead():
    df = _synthetic_ohlcv()
    factors = compute_factors(df)
    for col in ["mom_20", "mom_60", "rsi_14", "vol_ratio", "zscore_20"]:
        assert col in factors.columns
    # momentum at t only depends on close[t] and close[t-20]; the LAST row must
    # be finite (uses no future data)
    assert np.isfinite(factors["mom_20"].iloc[-1])
    assert len(factors) == len(df)


def test_rsi_bounded_0_100():
    df = _synthetic_ohlcv()
    rsi = compute_factors(df)["rsi_14"].dropna()
    assert rsi.min() >= 0 and rsi.max() <= 100
