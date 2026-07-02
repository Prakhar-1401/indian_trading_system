"""
Unit tests for portfolio-construction math.

We construct the optimizer directly from a known mean/covariance (bypassing the
network `load()`) so the optimizers are tested deterministically.
"""
import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimizer import (
    PortfolioOptimizer,
    diversification_ratio,
    portfolio_stats,
    risk_contributions,
)


def _make_optimizer(seed=0, n_assets=5, n_days=500):
    """Build an optimizer with synthetic-but-realistic returns/cov."""
    rng = np.random.default_rng(seed)
    syms = [f"A{i}" for i in range(n_assets)]
    rets = pd.DataFrame(
        rng.normal(0.0006, 0.012, (n_days, n_assets)), columns=syms
    )
    opt = PortfolioOptimizer()
    opt.symbols = syms
    opt.returns = rets
    opt.mu = rets.mean() * 252
    opt.cov = rets.cov() * 252
    return opt


# --------------------------------------------------------------------------- #
#  Risk-contribution / stats primitives
# --------------------------------------------------------------------------- #
def test_risk_contributions_sum_to_one():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.5, 0.5])
    rc = risk_contributions(w, cov)
    assert pytest.approx(rc.sum(), abs=1e-9) == 1.0


def test_portfolio_stats_basic():
    mu = np.array([0.10, 0.10])
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    w = np.array([0.5, 0.5])
    ret, vol, sharpe = portfolio_stats(w, mu, cov, rf=0.0)
    assert pytest.approx(ret, abs=1e-9) == 0.10
    # diversification reduces vol below 0.2
    assert vol < 0.2


def test_diversification_ratio_ge_one():
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    w = np.array([0.5, 0.5])
    assert diversification_ratio(w, cov) >= 1.0


# --------------------------------------------------------------------------- #
#  Each optimizer: weights valid (sum to 1, non-negative for long-only methods)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", [
    "max_sharpe", "min_variance", "risk_parity", "hrp", "black_litterman",
])
def test_optimizer_weights_valid(method):
    opt = _make_optimizer()
    res = opt.optimize(method)
    w = res.weights
    assert pytest.approx(w.sum(), abs=1e-6) == 1.0
    assert (w >= -1e-6).all()                  # long-only
    assert len(w) == len(opt.symbols)
    assert np.isfinite(res.sharpe)


def test_risk_parity_equalizes_risk():
    opt = _make_optimizer(seed=2, n_assets=4)
    res = opt.optimize("risk_parity")
    rc = res.risk_contributions.values
    # equal-risk: contributions should be close to 1/n
    assert np.std(rc) < 0.05


def test_min_variance_has_lowest_vol():
    opt = _make_optimizer(seed=5)
    mv = opt.optimize("min_variance").volatility
    ms = opt.optimize("max_sharpe").volatility
    # min-variance vol should be <= max-sharpe vol (within tolerance)
    assert mv <= ms + 1e-6


def test_unknown_method_raises():
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        opt.optimize("nonsense")
