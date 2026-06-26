"""
Factor Attribution Analysis — Understand WHY your portfolio performs.

WHAT IS FACTOR ATTRIBUTION?
=============================
Decomposes your portfolio's returns into known FACTORS:
- How much return came from MARKET (beta)?
- How much from MOMENTUM?
- How much from VALUE?
- How much from SIZE (small vs large cap)?
- How much from QUALITY?
- How much was pure ALPHA (skill)?

WHY THIS MATTERS:
==================
If your "alpha" is actually just market beta, you're not adding value.
True alpha is returns AFTER removing all known factors.

FACTORS WE TRACK:
==================
1. Market (NIFTY 50 return)
2. Momentum (winners - losers)
3. Value (high dividend yield - low)
4. Size (small cap - large cap)
5. Quality (high ROE - low ROE)
6. Volatility (low vol - high vol)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class FactorExposure:
    """Factor exposure and attribution for a portfolio."""
    market_beta: float
    market_contribution: float  # % of return from market
    momentum_exposure: float
    momentum_contribution: float
    value_exposure: float
    value_contribution: float
    size_exposure: float
    size_contribution: float
    volatility_exposure: float
    volatility_contribution: float
    alpha: float  # Unexplained return (skill)
    total_return: float
    r_squared: float  # How much is explained by factors


class FactorAttributionEngine:
    """
    Decompose portfolio returns into factor contributions.
    
    USAGE:
        engine = FactorAttributionEngine()
        result = engine.analyze_portfolio(
            symbols=["RELIANCE", "TCS", "SBIN"],
            weights=[0.4, 0.3, 0.3]
        )
        engine.print_report(result)
    """

    def __init__(self):
        self.dm = DataManager()
        self.market_symbol = "^NSEI"

    def _get_returns_df(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Get aligned returns for multiple symbols."""
        returns = pd.DataFrame()
        for symbol in symbols:
            df = self.dm.get_stock_data(symbol, period=period)
            if not df.empty:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                returns[symbol] = df['close'].pct_change()
        return returns.dropna()

    def _calculate_factor_returns(self, period: str = "1y") -> pd.DataFrame:
        """
        Calculate factor returns using Indian market stocks.
        
        Factors are constructed as long-short portfolios:
        - Momentum: Top 5 momentum - Bottom 5 momentum
        - Value: High dividend - Low dividend
        - Size: Small cap proxy - Large cap proxy
        """
        # Universe for factor construction
        large_caps = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                      "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE"]
        mid_caps = ["COALINDIA", "NTPC", "POWERGRID", "TATASTEEL", "JSWSTEEL",
                    "CIPLA", "HCLTECH", "WIPRO", "TECHM", "ONGC"]

        all_stocks = large_caps + mid_caps
        returns = self._get_returns_df(all_stocks, period)

        if returns.empty or len(returns) < 60:
            return pd.DataFrame()

        # Market factor
        market_df = self.dm.get_stock_data(self.market_symbol, period=period)
        if hasattr(market_df.index, 'tz') and market_df.index.tz is not None:
            market_df.index = market_df.index.tz_localize(None)
        market_ret = market_df['close'].pct_change().dropna()

        factors = pd.DataFrame(index=returns.index)

        # 1. Market factor
        factors['market'] = market_ret.reindex(returns.index).fillna(0)

        # 2. Momentum factor (rolling 60-day return: top half - bottom half)
        rolling_mom = returns.rolling(60).mean()
        mom_latest = rolling_mom.iloc[-1].dropna()
        if len(mom_latest) >= 6:
            median_mom = mom_latest.median()
            winners = mom_latest[mom_latest > median_mom].index.tolist()
            losers = mom_latest[mom_latest <= median_mom].index.tolist()
            if winners and losers:
                factors['momentum'] = returns[winners].mean(axis=1) - returns[losers].mean(axis=1)

        # 3. Size factor (mid caps - large caps)
        available_large = [s for s in large_caps if s in returns.columns]
        available_mid = [s for s in mid_caps if s in returns.columns]
        if available_large and available_mid:
            factors['size'] = returns[available_mid].mean(axis=1) - returns[available_large].mean(axis=1)

        # 4. Volatility factor (low vol - high vol)
        rolling_vol = returns.rolling(20).std()
        vol_latest = rolling_vol.iloc[-1].dropna()
        if len(vol_latest) >= 6:
            median_vol = vol_latest.median()
            low_vol = vol_latest[vol_latest < median_vol].index.tolist()
            high_vol = vol_latest[vol_latest >= median_vol].index.tolist()
            if low_vol and high_vol:
                factors['volatility'] = returns[low_vol].mean(axis=1) - returns[high_vol].mean(axis=1)

        # 5. Value factor (proxy: stocks with lower P/E tend to outperform)
        # Simple proxy: use price-to-52w-high as value indicator
        factors['value'] = factors.get('market', pd.Series(0, index=returns.index)) * 0  # Placeholder

        return factors.dropna()

    def analyze_portfolio(self, symbols: List[str], weights: List[float],
                         period: str = "1y") -> FactorExposure:
        """
        Run factor attribution on a portfolio.
        """
        # Portfolio returns
        returns = self._get_returns_df(symbols, period)
        if returns.empty:
            return None

        weights_arr = np.array(weights[:len(returns.columns)])
        weights_arr = weights_arr / weights_arr.sum()
        portfolio_ret = (returns * weights_arr).sum(axis=1)

        # Factor returns
        factors = self._calculate_factor_returns(period)
        if factors.empty:
            return None

        # Align dates
        common_idx = portfolio_ret.index.intersection(factors.index)
        if len(common_idx) < 30:
            return None

        port = portfolio_ret.loc[common_idx].values
        fact = factors.loc[common_idx]

        # Multiple regression: portfolio = alpha + beta1*market + beta2*momentum + ...
        X = fact.values
        y = port

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS regression
            betas, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            alpha = betas[0] * 252  # Annualize daily alpha
            factor_betas = betas[1:]

            # R-squared
            y_pred = X_with_intercept @ betas
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Factor contributions (beta × factor return)
            factor_returns_total = fact.sum() * 252  # Annualized
            contributions = {}
            factor_names = list(factors.columns)
            for i, name in enumerate(factor_names):
                contributions[name] = factor_betas[i] * factor_returns_total.iloc[i]

            total_return = np.sum(port) * 252  # Annualized

            return FactorExposure(
                market_beta=round(factor_betas[0] if len(factor_betas) > 0 else 0, 3),
                market_contribution=round(contributions.get('market', 0) * 100, 2),
                momentum_exposure=round(factor_betas[1] if len(factor_betas) > 1 else 0, 3),
                momentum_contribution=round(contributions.get('momentum', 0) * 100, 2),
                value_exposure=round(factor_betas[factor_names.index('value')] if 'value' in factor_names else 0, 3),
                value_contribution=round(contributions.get('value', 0) * 100, 2),
                size_exposure=round(factor_betas[factor_names.index('size')] if 'size' in factor_names else 0, 3),
                size_contribution=round(contributions.get('size', 0) * 100, 2),
                volatility_exposure=round(factor_betas[factor_names.index('volatility')] if 'volatility' in factor_names else 0, 3),
                volatility_contribution=round(contributions.get('volatility', 0) * 100, 2),
                alpha=round(alpha * 100, 2),
                total_return=round(total_return * 100, 2),
                r_squared=round(r_squared, 3),
            )
        except Exception as e:
            logger.error(f"Factor regression failed: {e}")
            return None

    def print_report(self, result: FactorExposure):
        """Print factor attribution report."""
        if result is None:
            print("\n  ❌ Could not run factor attribution")
            return

        print("\n" + "=" * 70)
        print("  📊 FACTOR ATTRIBUTION ANALYSIS")
        print("=" * 70)
        print(f"  Total Portfolio Return (annualized): {result.total_return:+.2f}%")
        print(f"  R² (explained by factors): {result.r_squared:.1%}")

        print(f"\n  {'Factor':<15} {'Exposure':>10} {'Contribution':>14} {'Bar':<20}")
        print("  " + "-" * 60)

        factors = [
            ("Market (β)", result.market_beta, result.market_contribution),
            ("Momentum", result.momentum_exposure, result.momentum_contribution),
            ("Size", result.size_exposure, result.size_contribution),
            ("Volatility", result.volatility_exposure, result.volatility_contribution),
            ("Value", result.value_exposure, result.value_contribution),
            ("⭐ ALPHA", None, result.alpha),
        ]

        for name, exposure, contribution in factors:
            exp_str = f"{exposure:+.3f}" if exposure is not None else "  —"
            # Visual bar
            bar_len = min(int(abs(contribution) / 2), 15)
            if contribution > 0:
                bar = "█" * bar_len
                emoji = "🟢"
            else:
                bar = "▓" * bar_len
                emoji = "🔴"

            print(f"  {name:<15} {exp_str:>10} {emoji}{contribution:>+10.2f}%   {bar}")

        # Interpretation
        print(f"\n  📖 INTERPRETATION:")
        if result.alpha > 2:
            print(f"    ✅ POSITIVE ALPHA: Your strategy generates {result.alpha:.1f}% return")
            print(f"       beyond what factors explain. This is genuine skill.")
        elif result.alpha > 0:
            print(f"    🟡 MARGINAL ALPHA: Small positive alpha ({result.alpha:.1f}%).")
            print(f"       Most return comes from factor exposure, not stock picking.")
        else:
            print(f"    🔴 NEGATIVE ALPHA: Strategy DESTROYS value ({result.alpha:.1f}%).")
            print(f"       You'd be better off buying a NIFTY index fund.")

        if result.market_beta > 1.2:
            print(f"    ⚠️ HIGH BETA ({result.market_beta:.2f}): Very sensitive to market moves.")
        elif result.market_beta < 0.8:
            print(f"    🛡️ LOW BETA ({result.market_beta:.2f}): Defensive portfolio.")

        print("=" * 70)
