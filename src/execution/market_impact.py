"""
Execution Quality — modelling what it actually COSTS to trade.

A backtest that assumes a flat 0.1% slippage is fiction. Real costs scale with
how much you trade relative to liquidity, and a large order moves the price
against you. This module brings three institutional concepts:

1. Square-root market impact model
       impact (bps) = c * sigma * sqrt(Q / ADV)
   - sigma = daily volatility, Q = order size (shares), ADV = avg daily volume,
     c = a calibration constant (~1). This is the empirically observed law used
     across the industry (Almgren et al. 2005, BARRA, Kissell). Cost grows with
     the SQUARE ROOT of participation, not linearly.

2. Implementation Shortfall (Perold, 1988)
   - The gap between the "paper" (decision-price) return and the realised return
     after commissions, spread, and market impact. This is the single number a
     buy-side trading desk is measured on.

3. Almgren-Chriss optimal execution
   - Splits a parent order into child slices over time to minimise a combination
     of market impact (trade fast = expensive) and timing risk (trade slow =
     exposed to volatility). Produces the optimal trading trajectory for a given
     risk aversion lambda.

REFERENCES
----------
- Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions".
- Almgren, Thum, Hauptmann, Li (2005), square-root impact calibration.
- Perold (1988), "The Implementation Shortfall: Paper vs Reality".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataManager
from src.utils.helpers import setup_logging

logger = setup_logging()

TRADING_DAYS = 252


@dataclass
class ImpactEstimate:
    symbol: str
    order_shares: float
    order_value: float
    adv_shares: float
    participation: float          # Q / ADV
    daily_vol: float
    spread_cost_bps: float
    impact_bps: float
    total_cost_bps: float
    total_cost_value: float


@dataclass
class ImplementationShortfall:
    decision_price: float
    avg_exec_price: float
    final_price: float
    side: str
    shares: float
    execution_cost_bps: float     # exec vs decision
    opportunity_cost_bps: float   # unfilled drift (final vs decision)
    commission_bps: float
    total_shortfall_bps: float
    total_shortfall_value: float


@dataclass
class AlmgrenChrissResult:
    n_slices: int
    trajectory: pd.DataFrame      # remaining + traded per period
    expected_impact_cost: float
    expected_timing_risk: float
    expected_total_cost: float


class ExecutionAnalyzer:
    """
    USAGE:
        ex = ExecutionAnalyzer()
        est = ex.estimate_impact("RELIANCE", order_value=5_000_000)
        ex.print_impact(est)
    """

    def __init__(self, data_manager: Optional[DataManager] = None,
                 impact_constant: float = 1.0, commission_bps: float = 3.0):
        self.dm = data_manager or DataManager()
        self.c = impact_constant
        self.commission_bps = commission_bps   # ~brokerage+STT+exchange for NSE delivery

    # ----- liquidity snapshot ------------------------------------------- #
    def _liquidity(self, symbol: str, lookback: int = 60) -> dict:
        df = self.dm.get_stock_data(symbol, period="6mo")
        if df is None or df.empty:
            raise RuntimeError(f"No data for {symbol}")
        df = df.tail(lookback)
        close = df["close"].astype(float)
        last = float(close.iloc[-1])
        adv_shares = float(df["volume"].astype(float).mean()) if "volume" in df else 0.0
        daily_vol = float(close.pct_change().std())
        # proxy half-spread from high-low range (Indian large caps ~3-8 bps)
        hl = (df["high"].astype(float) - df["low"].astype(float)) / close
        half_spread_bps = float(hl.mean()) * 10000 * 0.1   # ~10% of daily range
        half_spread_bps = float(np.clip(half_spread_bps, 1.0, 25.0))
        return {"price": last, "adv_shares": adv_shares,
                "daily_vol": daily_vol, "half_spread_bps": half_spread_bps}

    # ----- 1. square-root market impact --------------------------------- #
    def estimate_impact(
        self, symbol: str, order_value: float = 0.0, order_shares: float = 0.0
    ) -> ImpactEstimate:
        liq = self._liquidity(symbol)
        price = liq["price"]
        if order_shares <= 0:
            order_shares = order_value / price if price > 0 else 0.0
        order_value = order_shares * price

        adv = liq["adv_shares"]
        participation = order_shares / adv if adv > 0 else 0.0
        sigma = liq["daily_vol"]

        # square-root law: impact in bps
        impact_bps = self.c * sigma * np.sqrt(max(participation, 0)) * 10000
        spread_bps = liq["half_spread_bps"]
        total_bps = impact_bps + spread_bps + self.commission_bps
        total_value = order_value * total_bps / 10000

        return ImpactEstimate(
            symbol=symbol, order_shares=order_shares, order_value=order_value,
            adv_shares=adv, participation=participation, daily_vol=sigma,
            spread_cost_bps=spread_bps, impact_bps=impact_bps,
            total_cost_bps=total_bps, total_cost_value=total_value,
        )

    # ----- 2. implementation shortfall ---------------------------------- #
    def implementation_shortfall(
        self, decision_price: float, avg_exec_price: float, final_price: float,
        shares: float, side: str = "buy", filled_shares: Optional[float] = None,
    ) -> ImplementationShortfall:
        side = side.lower()
        sign = 1 if side == "buy" else -1
        filled = shares if filled_shares is None else filled_shares
        unfilled = max(shares - filled, 0)

        # execution cost: paid vs decision on the filled portion
        exec_cost_bps = sign * (avg_exec_price - decision_price) / decision_price * 10000
        exec_cost_bps *= filled / shares if shares else 0
        # opportunity cost: drift on the part we failed to fill
        opp_cost_bps = sign * (final_price - decision_price) / decision_price * 10000
        opp_cost_bps *= unfilled / shares if shares else 0
        commission_bps = self.commission_bps

        total_bps = exec_cost_bps + opp_cost_bps + commission_bps
        total_value = shares * decision_price * total_bps / 10000

        return ImplementationShortfall(
            decision_price=decision_price, avg_exec_price=avg_exec_price,
            final_price=final_price, side=side, shares=shares,
            execution_cost_bps=exec_cost_bps, opportunity_cost_bps=opp_cost_bps,
            commission_bps=commission_bps, total_shortfall_bps=total_bps,
            total_shortfall_value=total_value,
        )

    # ----- 3. Almgren-Chriss optimal execution -------------------------- #
    def almgren_chriss(
        self, symbol: str, total_shares: float, n_slices: int = 10,
        horizon_days: float = 1.0, risk_aversion: float = 1e-6,
        eta: Optional[float] = None, gamma: Optional[float] = None,
    ) -> AlmgrenChrissResult:
        """
        Closed-form Almgren-Chriss trajectory for liquidating `total_shares`.

        eta   = temporary impact coefficient (cost of trading fast)
        gamma = permanent impact coefficient
        risk_aversion (lambda) trades off cost vs variance.
        """
        liq = self._liquidity(symbol)
        price = liq["price"]
        sigma = liq["daily_vol"] * price          # absolute daily vol in price units
        adv = liq["adv_shares"]

        tau = horizon_days / n_slices             # time per slice (days)
        # calibrate impact coefficients from liquidity if not supplied
        if eta is None:
            eta = (liq["half_spread_bps"] / 10000 * price) / max(adv * 0.01, 1)
        if gamma is None:
            gamma = eta * 0.1

        # Almgren-Chriss kappa: characteristic time of optimal trajectory
        eta_tilde = eta - 0.5 * gamma * tau
        eta_tilde = max(eta_tilde, 1e-12)
        kappa_sq = risk_aversion * sigma ** 2 / eta_tilde
        kappa = np.sqrt(max(kappa_sq, 1e-18))

        T = horizon_days
        times = np.linspace(0, T, n_slices + 1)
        if kappa * T < 1e-6:
            # linear (TWAP) limit when risk aversion -> 0
            remaining = total_shares * (1 - times / T)
        else:
            remaining = total_shares * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)
        traded = -np.diff(remaining)

        traj = pd.DataFrame({
            "period": np.arange(1, n_slices + 1),
            "shares_traded": traded,
            "shares_remaining": remaining[1:],
            "pct_of_total": traded / total_shares * 100,
        })

        # expected costs
        impact_cost = float(eta / tau * np.sum(traded ** 2))
        timing_risk = float(np.sqrt(sigma ** 2 * tau * np.sum(remaining[1:] ** 2)))
        total_cost = impact_cost + risk_aversion * timing_risk

        return AlmgrenChrissResult(
            n_slices=n_slices, trajectory=traj,
            expected_impact_cost=impact_cost,
            expected_timing_risk=timing_risk,
            expected_total_cost=total_cost,
        )

    # ----- reporting ----------------------------------------------------- #
    def print_impact(self, est: ImpactEstimate) -> None:
        line = "=" * 64
        print("\n" + line)
        print(f"  MARKET IMPACT ESTIMATE — {est.symbol}  (square-root model)")
        print(line)
        print(f"  Order size:        {est.order_shares:>14,.0f} shares")
        print(f"  Order value:       Rs {est.order_value:>14,.0f}")
        print(f"  ADV:               {est.adv_shares:>14,.0f} shares")
        print(f"  Participation:     {est.participation*100:>13.3f}%  (Q/ADV)")
        print(f"  Daily volatility:  {est.daily_vol*100:>13.2f}%")
        print("  " + "-" * 60)
        print(f"  Spread cost:       {est.spread_cost_bps:>13.2f} bps")
        print(f"  Market impact:     {est.impact_bps:>13.2f} bps")
        print(f"  Commission:        {self.commission_bps:>13.2f} bps")
        print(f"  TOTAL COST:        {est.total_cost_bps:>13.2f} bps  "
              f"(Rs {est.total_cost_value:,.0f})")
        print(line + "\n")

    def print_almgren(self, res: AlmgrenChrissResult) -> None:
        line = "=" * 64
        print("\n" + line)
        print("  ALMGREN-CHRISS OPTIMAL EXECUTION TRAJECTORY")
        print(line)
        print(f"  Slices: {res.n_slices}")
        print(f"  {'Period':>6} {'Trade':>14} {'Remaining':>14} {'% Total':>9}")
        print("  " + "-" * 50)
        for _, r in res.trajectory.iterrows():
            print(f"  {int(r['period']):>6} {r['shares_traded']:>14,.0f} "
                  f"{r['shares_remaining']:>14,.0f} {r['pct_of_total']:>8.2f}%")
        print("  " + "-" * 50)
        print(f"  Expected impact cost: {res.expected_impact_cost:>14,.2f}")
        print(f"  Expected timing risk: {res.expected_timing_risk:>14,.2f}")
        print(line + "\n")


if __name__ == "__main__":  # pragma: no cover
    ex = ExecutionAnalyzer()
    ex.print_impact(ex.estimate_impact("RELIANCE", order_value=10_000_000))
    ex.print_almgren(ex.almgren_chriss("RELIANCE", total_shares=50_000, n_slices=10))
