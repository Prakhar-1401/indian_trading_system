"""
Market Regime Detection — Know WHEN to trade aggressively vs defensively.

WHAT IS A REGIME?
=================
Markets cycle through distinct "regimes" or moods:
- BULL: Trending up, low volatility → Go aggressive, ride trends
- BEAR: Trending down, high volatility → Go defensive, reduce exposure
- SIDEWAYS: No clear direction → Trade mean reversion, pairs
- VOLATILE: High uncertainty → Reduce size, widen stops, buy puts

WHY THIS MATTERS (Jane Street level thinking):
==============================================
Most retail traders use the SAME strategy in all conditions.
Quant firms ADAPT:
- In BULL regime: Momentum strategies dominate (buy breakouts)
- In BEAR regime: Mean reversion works better (buy dips)
- In SIDEWAYS: Pairs trading, options premium selling
- In VOLATILE: Cash is king, or use options for protection

HOW IT WORKS:
=============
1. Trend Detection: SMA slopes, price vs 200-DMA
2. Volatility Classification: VIX-equivalent, ATR percentile
3. Breadth Analysis: % of stocks above their 50-DMA
4. Hidden Markov Model concept: Probability of being in each regime
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class RegimeState:
    """Current market regime classification."""
    regime: str  # BULL, BEAR, SIDEWAYS, VOLATILE
    confidence: float  # 0-1
    trend_score: float  # -1 (bear) to +1 (bull)
    volatility_percentile: float  # 0-100
    breadth_score: float  # 0-1 (% stocks in uptrend)
    recommendation: str  # Strategy recommendation
    details: Dict


class MarketRegimeDetector:
    """
    Detect the current market regime using multiple signals.
    
    USAGE:
        detector = MarketRegimeDetector()
        regime = detector.detect_regime()
        detector.print_report(regime)
    """

    def __init__(self):
        self.dm = DataManager()
        # NIFTY 50 as market proxy
        self.market_symbol = "^NSEI"
        # Broad market stocks for breadth
        self.breadth_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
            "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "HCLTECH",
            "TATAMOTORS", "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND",
            "TECHM", "AXISBANK", "KOTAKBANK", "ULTRACEMCO", "ONGC",
            "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "CIPLA",
        ]

    def _get_market_data(self, period: str = "2y") -> pd.DataFrame:
        """Get NIFTY 50 data."""
        df = self.dm.get_stock_data(self.market_symbol, period=period)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _detect_trend(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Detect market trend using multiple methods.
        Returns score from -1 (strong bear) to +1 (strong bull).
        """
        if len(df) < 200:
            return 0, {}

        close = df['close']
        signals = {}

        # 1. Price vs key moving averages
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]

        signals['above_50sma'] = 1 if current > sma50 else -1
        signals['above_200sma'] = 1 if current > sma200 else -1
        signals['golden_cross'] = 1 if sma50 > sma200 else -1  # 50 > 200 = bullish

        # 2. SMA slope (is the trend accelerating?)
        sma50_series = close.rolling(50).mean()
        slope_20d = (sma50_series.iloc[-1] - sma50_series.iloc[-20]) / sma50_series.iloc[-20]
        signals['sma50_slope'] = np.clip(slope_20d * 20, -1, 1)  # Normalize

        # 3. Higher highs / lower lows
        recent_high = close.tail(20).max()
        prev_high = close.iloc[-40:-20].max()
        recent_low = close.tail(20).min()
        prev_low = close.iloc[-40:-20].min()

        if recent_high > prev_high and recent_low > prev_low:
            signals['hl_pattern'] = 1  # Higher highs & higher lows = bull
        elif recent_high < prev_high and recent_low < prev_low:
            signals['hl_pattern'] = -1  # Lower highs & lower lows = bear
        else:
            signals['hl_pattern'] = 0  # Mixed

        # 4. Rate of change (momentum)
        roc_20 = (close.iloc[-1] / close.iloc[-20] - 1)
        roc_60 = (close.iloc[-1] / close.iloc[-60] - 1)
        signals['roc_20d'] = np.clip(roc_20 * 10, -1, 1)
        signals['roc_60d'] = np.clip(roc_60 * 5, -1, 1)

        # Weighted average
        weights = {
            'above_50sma': 0.15,
            'above_200sma': 0.20,
            'golden_cross': 0.20,
            'sma50_slope': 0.15,
            'hl_pattern': 0.10,
            'roc_20d': 0.10,
            'roc_60d': 0.10,
        }

        trend_score = sum(signals[k] * weights[k] for k in weights)
        return np.clip(trend_score, -1, 1), signals

    def _detect_volatility(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Classify volatility regime.
        Returns percentile (0-100) of current vol vs historical.
        """
        if len(df) < 60:
            return 50, {}

        # Daily returns volatility
        returns = df['close'].pct_change()
        current_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
        hist_vol = returns.rolling(20).std() * np.sqrt(252)

        # Percentile of current vol in last 1 year
        vol_percentile = (hist_vol < current_vol).tail(252).mean() * 100

        # ATR-based
        from src.strategy.risk_manager import DynamicRiskManager
        rm = DynamicRiskManager()
        current_atr = rm.calculate_atr(df.tail(20))
        hist_atr = rm.calculate_atr(df.tail(60))
        atr_ratio = current_atr / hist_atr if hist_atr > 0 else 1

        details = {
            'current_vol_annualized': round(current_vol * 100, 1),
            'vol_percentile': round(vol_percentile, 1),
            'atr_ratio': round(atr_ratio, 2),
        }

        return vol_percentile, details

    def _detect_breadth(self) -> Tuple[float, Dict]:
        """
        Market breadth: What % of stocks are in their own uptrend?
        
        High breadth (>70%) = healthy bull market
        Low breadth (<30%) = bear market or narrow rally
        """
        above_50sma = 0
        above_200sma = 0
        total = 0

        for symbol in self.breadth_stocks:
            try:
                df = self.dm.get_stock_data(symbol, period="1y")
                if df.empty or len(df) < 200:
                    continue

                close = df['close']
                current = close.iloc[-1]
                sma50 = close.rolling(50).mean().iloc[-1]
                sma200 = close.rolling(200).mean().iloc[-1]

                total += 1
                if current > sma50:
                    above_50sma += 1
                if current > sma200:
                    above_200sma += 1
            except Exception:
                continue

        if total == 0:
            return 0.5, {}

        breadth_50 = above_50sma / total
        breadth_200 = above_200sma / total
        breadth_score = (breadth_50 * 0.6 + breadth_200 * 0.4)

        details = {
            'stocks_above_50sma': f"{above_50sma}/{total} ({breadth_50:.0%})",
            'stocks_above_200sma': f"{above_200sma}/{total} ({breadth_200:.0%})",
            'breadth_score': round(breadth_score, 2),
        }

        return breadth_score, details

    def detect_regime(self) -> RegimeState:
        """
        Detect current market regime combining all signals.
        """
        logger.info("Detecting market regime...")

        # Get market data
        df = self._get_market_data()
        if df.empty:
            return RegimeState("UNKNOWN", 0, 0, 50, 0.5, "Insufficient data", {})

        # Run detectors
        trend_score, trend_details = self._detect_trend(df)
        vol_percentile, vol_details = self._detect_volatility(df)
        breadth_score, breadth_details = self._detect_breadth()

        # Classify regime
        if vol_percentile > 75:
            regime = "VOLATILE"
            confidence = vol_percentile / 100
        elif trend_score > 0.3 and breadth_score > 0.5:
            regime = "BULL"
            confidence = (trend_score + breadth_score) / 2
        elif trend_score < -0.3 and breadth_score < 0.5:
            regime = "BEAR"
            confidence = abs(trend_score + (1 - breadth_score)) / 2
        else:
            regime = "SIDEWAYS"
            confidence = 1 - abs(trend_score)  # More sideways = higher confidence

        # Strategy recommendation
        recommendations = {
            "BULL": "Aggressive momentum. Buy breakouts. Trail stops. Full position sizes.",
            "BEAR": "Defensive. Reduce exposure 50%. Short weak stocks. Buy puts for protection.",
            "SIDEWAYS": "Mean reversion. Pairs trading. Sell options premium. Smaller positions.",
            "VOLATILE": "Cash is king. Reduce position sizes 70%. Widen stops. Buy protection.",
        }

        # Combine details
        all_details = {
            'nifty_price': round(df['close'].iloc[-1], 2),
            'nifty_20d_return': f"{(df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100:.1f}%",
            **trend_details,
            **vol_details,
            **breadth_details,
        }

        return RegimeState(
            regime=regime,
            confidence=round(min(confidence, 1.0), 2),
            trend_score=round(trend_score, 3),
            volatility_percentile=round(vol_percentile, 1),
            breadth_score=round(breadth_score, 3),
            recommendation=recommendations[regime],
            details=all_details,
        )

    def print_report(self, state: RegimeState):
        """Print regime detection report."""
        emoji = {"BULL": "🟢🐂", "BEAR": "🔴🐻", "SIDEWAYS": "🟡↔️", "VOLATILE": "🔴⚡"}.get(state.regime, "⚪")

        print("\n" + "=" * 70)
        print("  MARKET REGIME DETECTION")
        print("=" * 70)

        print(f"\n  {emoji} Current Regime: {state.regime}")
        print(f"  Confidence: {state.confidence:.0%}")
        print(f"\n  📊 Trend Score: {state.trend_score:+.3f} (-1=bear, +1=bull)")
        print(f"  📈 Breadth Score: {state.breadth_score:.1%} (stocks in uptrend)")
        print(f"  🌊 Volatility: {state.volatility_percentile:.0f}th percentile")

        print(f"\n  💡 RECOMMENDATION:")
        print(f"     {state.recommendation}")

        # Details
        print(f"\n  Details:")
        print(f"    NIFTY 50: ₹{state.details.get('nifty_price', 'N/A'):,.0f}")
        print(f"    20-day return: {state.details.get('nifty_20d_return', 'N/A')}")
        print(f"    Annualized Vol: {state.details.get('current_vol_annualized', 'N/A')}%")
        print(f"    Stocks > 50-SMA: {state.details.get('stocks_above_50sma', 'N/A')}")
        print(f"    Stocks > 200-SMA: {state.details.get('stocks_above_200sma', 'N/A')}")

        # What to do
        print(f"\n  STRATEGY ADJUSTMENTS FOR {state.regime} REGIME:")
        adjustments = {
            "BULL": [
                "✅ Full position sizes (100%)",
                "✅ Buy momentum leaders (top ranked)",
                "✅ Use trailing stops (3x ATR)",
                "✅ Add on pullbacks to 20-DMA",
                "❌ Don't short, don't fight the trend",
            ],
            "BEAR": [
                "✅ Reduce position sizes to 50%",
                "✅ Hold more cash (30-50%)",
                "✅ Buy only defensive sectors (pharma, FMCG, utilities)",
                "✅ Consider short positions on weak stocks",
                "❌ Don't buy dips in weak stocks",
            ],
            "SIDEWAYS": [
                "✅ Use mean reversion (buy oversold, sell overbought)",
                "✅ Pairs trading is ideal",
                "✅ Smaller position sizes (70%)",
                "✅ Tight targets, quick profits",
                "❌ Don't expect big trends",
            ],
            "VOLATILE": [
                "✅ Reduce all positions by 70%",
                "✅ Widen stops (use 3x ATR instead of 2x)",
                "✅ Cash is a valid position",
                "✅ Buy puts for portfolio protection",
                "❌ Don't try to catch falling knives",
            ],
        }

        for adj in adjustments.get(state.regime, []):
            print(f"    {adj}")

        print("=" * 70)
