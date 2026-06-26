"""
Monte Carlo Simulation — Stress-test strategies with 10,000 scenarios.

WHAT IS MONTE CARLO?
=====================
Instead of backtesting on ONE historical path, we generate THOUSANDS
of possible market scenarios to see:
- Worst case scenario (95th percentile drawdown)
- Best case scenario
- Probability of hitting your target return
- Probability of ruin (blowing up)

WHY QUANT FIRMS USE THIS:
==========================
A single backtest can fool you (overfitting to one path).
Monte Carlo answers: "If the market does something DIFFERENT from
history, will I survive?"

METHODS:
========
1. BOOTSTRAP: Resample actual daily returns randomly (preserves distribution)
2. PARAMETRIC: Generate returns from fitted distribution (Normal, t-dist)
3. BLOCK BOOTSTRAP: Resample in blocks (preserves autocorrelation)

WE USE: Block bootstrap + parametric for comprehensive stress testing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class MonteCarloResult:
    """Results of Monte Carlo simulation."""
    num_simulations: int
    days_simulated: int
    initial_capital: float
    # Return statistics
    mean_return: float
    median_return: float
    std_return: float
    best_case: float  # 95th percentile
    worst_case: float  # 5th percentile
    # Risk metrics
    prob_profit: float  # Probability of positive return
    prob_target: float  # Probability of hitting target
    prob_ruin: float  # Probability of >30% drawdown
    max_drawdown_median: float
    max_drawdown_95: float  # 95th percentile worst drawdown
    # Distribution
    percentiles: Dict[int, float]  # {5: x, 25: x, 50: x, 75: x, 95: x}
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (expected loss beyond VaR)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for strategy stress testing.
    
    USAGE:
        sim = MonteCarloSimulator()
        result = sim.simulate_portfolio(
            symbols=["RELIANCE", "TCS", "SBIN"],
            weights=[0.4, 0.3, 0.3],
            days=252,
            num_sims=10000
        )
        sim.print_report(result)
    """

    def __init__(self):
        self.dm = DataManager()

    def _get_returns(self, symbol: str, period: str = "5y") -> pd.Series:
        """Get historical daily returns for a symbol."""
        df = self.dm.get_stock_data(symbol, period=period)
        if df.empty:
            return pd.Series(dtype=float)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df['close'].pct_change().dropna()

    def simulate_single_stock(self, symbol: str, days: int = 252,
                              num_sims: int = 10000, capital: float = 1000000,
                              method: str = "block_bootstrap") -> MonteCarloResult:
        """Run Monte Carlo simulation for a single stock."""
        returns = self._get_returns(symbol)
        if len(returns) < 60:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        return self._run_simulation(returns, days, num_sims, capital)

    def simulate_portfolio(self, symbols: List[str], weights: List[float],
                          days: int = 252, num_sims: int = 10000,
                          capital: float = 1000000) -> MonteCarloResult:
        """
        Run Monte Carlo for a portfolio of stocks.
        
        Uses correlated returns (preserves inter-stock relationships).
        """
        # Get all returns
        all_returns = pd.DataFrame()
        for symbol in symbols:
            r = self._get_returns(symbol)
            if not r.empty:
                all_returns[symbol] = r

        if all_returns.empty:
            return None

        # Drop NaN rows (align dates)
        all_returns = all_returns.dropna()

        # Calculate portfolio returns
        weights_arr = np.array(weights[:len(all_returns.columns)])
        weights_arr = weights_arr / weights_arr.sum()  # Normalize
        portfolio_returns = all_returns.values @ weights_arr

        return self._run_simulation(
            pd.Series(portfolio_returns), days, num_sims, capital
        )

    def _run_simulation(self, returns: pd.Series, days: int,
                       num_sims: int, capital: float,
                       target_return: float = 0.15) -> MonteCarloResult:
        """
        Core simulation engine using block bootstrap.
        
        Block bootstrap: sample blocks of 5-20 days to preserve
        momentum/mean-reversion patterns.
        """
        returns_arr = returns.values
        n = len(returns_arr)

        # Simulation results
        final_values = np.zeros(num_sims)
        max_drawdowns = np.zeros(num_sims)

        block_size = min(10, n // 10)  # 10-day blocks

        for sim in range(num_sims):
            # Generate one path using block bootstrap
            path = np.zeros(days)
            idx = 0
            while idx < days:
                # Random starting point for block
                start = np.random.randint(0, n - block_size)
                block = returns_arr[start:start + block_size]
                end = min(idx + block_size, days)
                path[idx:end] = block[:end - idx]
                idx = end

            # Calculate equity curve
            equity = capital * np.cumprod(1 + path)
            final_values[sim] = equity[-1]

            # Calculate max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_drawdowns[sim] = drawdown.min()  # Most negative

        # Calculate statistics
        final_returns = (final_values / capital - 1) * 100  # In %

        percentiles = {
            5: np.percentile(final_returns, 5),
            10: np.percentile(final_returns, 10),
            25: np.percentile(final_returns, 25),
            50: np.percentile(final_returns, 50),
            75: np.percentile(final_returns, 75),
            90: np.percentile(final_returns, 90),
            95: np.percentile(final_returns, 95),
        }

        # VaR and CVaR
        var_95 = np.percentile(final_returns, 5)  # 5th percentile loss
        cvar_95 = final_returns[final_returns <= var_95].mean()

        return MonteCarloResult(
            num_simulations=num_sims,
            days_simulated=days,
            initial_capital=capital,
            mean_return=round(np.mean(final_returns), 2),
            median_return=round(np.median(final_returns), 2),
            std_return=round(np.std(final_returns), 2),
            best_case=round(percentiles[95], 2),
            worst_case=round(percentiles[5], 2),
            prob_profit=round((final_returns > 0).mean() * 100, 1),
            prob_target=round((final_returns > target_return * 100).mean() * 100, 1),
            prob_ruin=round((max_drawdowns < -0.30).mean() * 100, 1),
            max_drawdown_median=round(np.median(max_drawdowns) * 100, 1),
            max_drawdown_95=round(np.percentile(max_drawdowns, 5) * 100, 1),
            percentiles={k: round(v, 2) for k, v in percentiles.items()},
            var_95=round(var_95, 2),
            cvar_95=round(cvar_95, 2) if not np.isnan(cvar_95) else var_95,
        )

    def print_report(self, result: MonteCarloResult, title: str = "PORTFOLIO"):
        """Print Monte Carlo simulation report."""
        if result is None:
            print("\n  ❌ Could not run simulation (insufficient data)")
            return

        print("\n" + "=" * 70)
        print(f"  🎲 MONTE CARLO SIMULATION — {title}")
        print("=" * 70)
        print(f"  Simulations: {result.num_simulations:,}")
        print(f"  Time Horizon: {result.days_simulated} trading days (~{result.days_simulated//252}y {(result.days_simulated%252)//21}m)")
        print(f"  Initial Capital: ₹{result.initial_capital:,.0f}")

        # Return Distribution
        print(f"\n  📊 RETURN DISTRIBUTION")
        print(f"  " + "-" * 40)
        print(f"    Mean Return:    {result.mean_return:+.2f}%")
        print(f"    Median Return:  {result.median_return:+.2f}%")
        print(f"    Std Deviation:  {result.std_return:.2f}%")
        print(f"    Best Case (95): {result.best_case:+.2f}%")
        print(f"    Worst Case (5): {result.worst_case:+.2f}%")

        # Probability Analysis
        print(f"\n  🎯 PROBABILITY ANALYSIS")
        print(f"  " + "-" * 40)
        print(f"    P(Profit > 0%):   {result.prob_profit:.1f}%")
        print(f"    P(Return > 15%):  {result.prob_target:.1f}%")
        print(f"    P(Ruin > -30%):   {result.prob_ruin:.1f}%")

        # Risk Metrics
        print(f"\n  ⚠️ RISK METRICS")
        print(f"  " + "-" * 40)
        print(f"    VaR (95%):        {result.var_95:+.2f}% (you won't lose more than this 95% of time)")
        print(f"    CVaR (95%):       {result.cvar_95:+.2f}% (if things go bad, expect this loss)")
        print(f"    Max DD (median):  {result.max_drawdown_median:.1f}%")
        print(f"    Max DD (worst):   {result.max_drawdown_95:.1f}%")

        # Visual distribution
        print(f"\n  📈 RETURN PERCENTILES")
        print(f"  " + "-" * 40)
        bar_width = 30
        for pct, val in sorted(result.percentiles.items()):
            # Normalize to bar
            normalized = min(max((val + 50) / 100, 0), 1)  # -50% to +50% range
            bar_len = int(normalized * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            label = f"P{pct:>2}"
            print(f"    {label} [{bar}] {val:+.1f}%")

        # Final values
        print(f"\n  💰 EXPECTED PORTFOLIO VALUE (₹{result.initial_capital/100000:.0f}L invested)")
        for pct in [5, 25, 50, 75, 95]:
            final_val = result.initial_capital * (1 + result.percentiles[pct] / 100)
            print(f"    P{pct:>2}: ₹{final_val:>12,.0f} ({result.percentiles[pct]:+.1f}%)")

        print("=" * 70)
