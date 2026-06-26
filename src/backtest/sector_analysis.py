"""
Sector Heatmap & Correlation Matrix — See market structure at a glance.

WHAT THIS SHOWS:
=================
1. SECTOR HEATMAP: Which sectors are hot/cold today/this week/this month
2. CORRELATION MATRIX: Which stocks move together (diversification check)

WHY THIS MATTERS:
==================
- If all your stocks are 90% correlated, you DON'T have diversification
- Sector rotation: money flows from one sector to another in cycles
- Low correlation pairs → best for pairs trading
- High correlation = concentrated risk

SECTORS TRACKED:
================
- IT: TCS, INFY, WIPRO, HCLTECH, TECHM
- Banking: HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK
- Energy: RELIANCE, ONGC, NTPC, POWERGRID, COALINDIA
- Pharma: SUNPHARMA, DRREDDY, CIPLA
- Auto: MARUTI, TATAMOTORS, BAJAJ-AUTO, HEROMOTOCO
- Metals: TATASTEEL, JSWSTEEL, HINDALCO
- FMCG: ITC, NESTLEIND, HINDUNILVR
- Capital Goods: LT, BHARTIARTL
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger

from src.data.fetcher import DataManager


# Sector definitions
SECTORS = {
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "Energy": ["RELIANCE", "NTPC", "POWERGRID", "COALINDIA", "ONGC"],
    "Pharma": ["SUNPHARMA", "CIPLA"],
    "Auto": ["MARUTI", "BAJAJ-AUTO"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "FMCG": ["ITC", "NESTLEIND"],
    "Infra": ["LT", "ULTRACEMCO", "GRASIM"],
    "Finance": ["BAJFINANCE", "TITAN"],
    "Telecom": ["BHARTIARTL"],
}


class SectorAnalyzer:
    """
    Sector heatmap and correlation analysis.
    
    USAGE:
        analyzer = SectorAnalyzer()
        analyzer.print_sector_heatmap()
        analyzer.print_correlation_matrix()
    """

    def __init__(self):
        self.dm = DataManager()

    def get_sector_performance(self, period: str = "1mo") -> pd.DataFrame:
        """Calculate sector performance over given period."""
        results = []

        for sector, stocks in SECTORS.items():
            returns = []
            for symbol in stocks:
                df = self.dm.get_stock_data(symbol, period=period)
                if not df.empty and len(df) >= 2:
                    ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    returns.append(ret)

            if returns:
                results.append({
                    'sector': sector,
                    'return': round(np.mean(returns), 2),
                    'best_stock_return': round(max(returns), 2),
                    'worst_stock_return': round(min(returns), 2),
                    'num_stocks': len(returns),
                })

        return pd.DataFrame(results).sort_values('return', ascending=False)

    def get_stock_returns(self, period: str = "3mo") -> pd.DataFrame:
        """Get daily returns for all tracked stocks."""
        all_stocks = []
        for stocks in SECTORS.values():
            all_stocks.extend(stocks)
        all_stocks = list(set(all_stocks))

        returns = pd.DataFrame()
        for symbol in all_stocks:
            df = self.dm.get_stock_data(symbol, period=period)
            if not df.empty:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                returns[symbol] = df['close'].pct_change()

        return returns.dropna()

    def get_correlation_matrix(self, period: str = "3mo") -> pd.DataFrame:
        """Calculate correlation matrix between all stocks."""
        returns = self.get_stock_returns(period)
        if returns.empty:
            return pd.DataFrame()
        return returns.corr()

    def get_sector_correlation(self, period: str = "3mo") -> pd.DataFrame:
        """Calculate correlation between SECTORS (averaged)."""
        returns = self.get_stock_returns(period)
        if returns.empty:
            return pd.DataFrame()

        sector_returns = pd.DataFrame()
        for sector, stocks in SECTORS.items():
            available = [s for s in stocks if s in returns.columns]
            if available:
                sector_returns[sector] = returns[available].mean(axis=1)

        return sector_returns.corr()

    def print_sector_heatmap(self):
        """Print sector performance heatmap for multiple timeframes."""
        print("\n" + "=" * 70)
        print("  🗺️ SECTOR HEATMAP — Performance by Timeframe")
        print("=" * 70)

        timeframes = [("1W", "5d"), ("1M", "1mo"), ("3M", "3mo"), ("6M", "6mo")]

        # Header
        print(f"\n  {'Sector':<12}", end="")
        for label, _ in timeframes:
            print(f" {label:>8}", end="")
        print(f"  {'Trend':<15}")
        print("  " + "-" * 60)

        # Collect all data
        sector_data = {}
        for label, period in timeframes:
            perf = self.get_sector_performance(period)
            for _, row in perf.iterrows():
                if row['sector'] not in sector_data:
                    sector_data[row['sector']] = {}
                sector_data[row['sector']][label] = row['return']

        # Print rows sorted by 1M performance
        sorted_sectors = sorted(
            sector_data.items(),
            key=lambda x: x[1].get('1M', 0),
            reverse=True
        )

        for sector, data in sorted_sectors:
            print(f"  {sector:<12}", end="")
            for label, _ in timeframes:
                val = data.get(label, 0)
                if val > 3:
                    color = "🟢"
                elif val > 0:
                    color = "🟡"
                elif val > -3:
                    color = "🟠"
                else:
                    color = "🔴"
                print(f" {color}{val:>+5.1f}%", end="")

            # Trend arrow
            returns_list = [data.get(l, 0) for l, _ in timeframes]
            if len(returns_list) >= 2:
                if returns_list[0] > returns_list[-1] and returns_list[0] > 0:
                    trend = "📈 Accelerating"
                elif returns_list[0] < returns_list[-1] and returns_list[-1] > 0:
                    trend = "📉 Decelerating"
                elif all(r < 0 for r in returns_list):
                    trend = "⬇️ Downtrend"
                elif all(r > 0 for r in returns_list):
                    trend = "⬆️ Uptrend"
                else:
                    trend = "↔️ Mixed"
            else:
                trend = "—"
            print(f"  {trend}")

        # Sector rotation insight
        print(f"\n  💡 SECTOR ROTATION INSIGHT:")
        if sorted_sectors:
            hot = sorted_sectors[0][0]
            cold = sorted_sectors[-1][0]
            print(f"    🔥 Hottest: {hot} ({sorted_sectors[0][1].get('1M', 0):+.1f}% this month)")
            print(f"    ❄️ Coldest: {cold} ({sorted_sectors[-1][1].get('1M', 0):+.1f}% this month)")

        print("=" * 70)

    def print_correlation_matrix(self, top_n: int = 12):
        """Print correlation matrix for top stocks."""
        print("\n" + "=" * 70)
        print("  📊 CORRELATION MATRIX (3-Month)")
        print("=" * 70)

        # Sector-level correlation
        sector_corr = self.get_sector_correlation()
        if sector_corr.empty:
            print("  ❌ Insufficient data")
            return

        print("\n  SECTOR CORRELATIONS:")
        print(f"  {'':12}", end="")
        for col in sector_corr.columns[:8]:
            print(f" {col[:6]:>7}", end="")
        print()
        print("  " + "-" * 70)

        for idx in sector_corr.index[:8]:
            print(f"  {idx:<12}", end="")
            for col in sector_corr.columns[:8]:
                val = sector_corr.loc[idx, col]
                if idx == col:
                    print(f"    {'—':>4}", end="")
                else:
                    if val > 0.7:
                        color = "🔴"  # High correlation = risk
                    elif val > 0.4:
                        color = "🟡"
                    else:
                        color = "🟢"  # Low correlation = good diversification
                    print(f" {color}{val:>4.2f}", end="")
            print()

        # Find best diversification pairs
        print(f"\n  🟢 BEST DIVERSIFICATION PAIRS (lowest correlation):")
        pairs = []
        for i, s1 in enumerate(sector_corr.columns):
            for j, s2 in enumerate(sector_corr.columns):
                if i < j:
                    pairs.append((s1, s2, sector_corr.loc[s1, s2]))
        pairs.sort(key=lambda x: x[2])

        for s1, s2, corr in pairs[:5]:
            print(f"    {s1} ↔ {s2}: {corr:.3f}")

        # Highest correlation (concentrated risk)
        print(f"\n  🔴 HIGHEST CORRELATION (concentrated risk):")
        for s1, s2, corr in pairs[-3:]:
            print(f"    {s1} ↔ {s2}: {corr:.3f}")

        print("\n  KEY: 🟢 < 0.4 (good diversification) | 🟡 0.4-0.7 | 🔴 > 0.7 (concentrated)")
        print("=" * 70)
