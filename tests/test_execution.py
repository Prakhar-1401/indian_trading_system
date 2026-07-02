"""
Unit tests for execution-quality math.

The square-root impact model, implementation shortfall and Almgren-Chriss
trajectory are tested with the network `_liquidity` call monkeypatched to a
fixed liquidity snapshot, so results are deterministic.
"""
import numpy as np
import pytest

from src.execution.market_impact import ExecutionAnalyzer


@pytest.fixture
def analyzer(monkeypatch):
    ex = ExecutionAnalyzer(commission_bps=3.0)
    monkeypatch.setattr(ex, "_liquidity", lambda symbol, lookback=60: {
        "price": 1000.0,
        "adv_shares": 1_000_000.0,
        "daily_vol": 0.02,
        "half_spread_bps": 5.0,
    })
    return ex


# --------------------------------------------------------------------------- #
#  Square-root impact model
# --------------------------------------------------------------------------- #
def test_impact_scales_with_sqrt_participation(analyzer):
    small = analyzer.estimate_impact("X", order_shares=10_000)    # 1% ADV
    large = analyzer.estimate_impact("X", order_shares=40_000)    # 4% ADV
    # 4x the size -> 2x the impact (square-root law)
    ratio = large.impact_bps / small.impact_bps
    assert pytest.approx(ratio, rel=0.02) == 2.0


def test_impact_components_add_up(analyzer):
    est = analyzer.estimate_impact("X", order_shares=10_000)
    total = est.impact_bps + est.spread_cost_bps + analyzer.commission_bps
    assert pytest.approx(est.total_cost_bps, abs=1e-9) == total
    assert est.total_cost_value > 0


def test_zero_order_zero_impact(analyzer):
    est = analyzer.estimate_impact("X", order_shares=0)
    assert est.impact_bps == 0.0


# --------------------------------------------------------------------------- #
#  Implementation shortfall
# --------------------------------------------------------------------------- #
def test_implementation_shortfall_buy_costs_positive(analyzer):
    isf = analyzer.implementation_shortfall(
        decision_price=100.0, avg_exec_price=100.5, final_price=101.0,
        shares=1000, side="buy",
    )
    # paid more than decision -> positive execution cost
    assert isf.execution_cost_bps > 0
    assert isf.total_shortfall_bps > 0


def test_implementation_shortfall_opportunity_cost(analyzer):
    # only half filled, price ran away -> opportunity cost on the unfilled half
    isf = analyzer.implementation_shortfall(
        decision_price=100.0, avg_exec_price=100.0, final_price=102.0,
        shares=1000, side="buy", filled_shares=500,
    )
    assert isf.opportunity_cost_bps > 0


# --------------------------------------------------------------------------- #
#  Almgren-Chriss
# --------------------------------------------------------------------------- #
def test_almgren_trajectory_conserves_shares(analyzer):
    res = analyzer.almgren_chriss("X", total_shares=100_000, n_slices=10)
    assert len(res.trajectory) == 10
    # all shares get traded
    assert pytest.approx(res.trajectory["shares_traded"].sum(), rel=1e-6) == 100_000
    # remaining is monotonically decreasing
    rem = res.trajectory["shares_remaining"].values
    assert np.all(np.diff(rem) <= 1e-6)


def test_almgren_higher_risk_aversion_front_loads(analyzer):
    slow = analyzer.almgren_chriss("X", 100_000, n_slices=10, risk_aversion=1e-9)
    fast = analyzer.almgren_chriss("X", 100_000, n_slices=10, risk_aversion=1e-3)
    # higher risk aversion trades more in the first slice (reduces exposure faster)
    assert fast.trajectory["shares_traded"].iloc[0] >= slow.trajectory["shares_traded"].iloc[0]
