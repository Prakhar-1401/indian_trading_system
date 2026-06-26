"""
Statistical Arbitrage — Pairs Trading Engine

WHAT IS PAIRS TRADING?
======================
Find two stocks that historically move together (e.g., SBIN & ICICIBANK).
When they temporarily diverge, bet that they'll converge back:
  - Stock A goes up too much vs Stock B → Short A, Long B
  - Stock A goes down too much vs Stock B → Long A, Short B

WHY IT WORKS:
- Market-neutral (profits in bull AND bear markets)
- Based on statistical mean-reversion (academic backing)
- Used by every quant fund (Renaissance, Jane Street, Two Sigma)

HOW WE DO IT:
1. Find highly correlated stock pairs (correlation > 0.75)
2. Calculate the "spread" (price ratio or difference)
3. When spread deviates > 2 standard deviations from mean → TRADE
4. Exit when spread reverts to mean
5. Stop-loss if spread goes > 3 standard deviations (pair broke)

INDIAN MARKET PAIRS (Common):
- SBIN / ICICIBANK (both large banks)
- TCS / INFY (both IT services)
- HDFCBANK / KOTAKBANK (private banks)
- SUNPHARMA / CIPLA (pharma)
- TATASTEEL / JSWSTEEL (steel)
- NTPC / POWERGRID (power utilities)
- RELIANCE / ONGC (energy)
- MARUTI / TATAMOTORS (auto — if available)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from loguru import logger
from scipy import stats

from src.data.fetcher import DataManager


@dataclass
class PairSignal:
    """A trading signal for a pair."""
    stock_long: str
    stock_short: str
    zscore: float
    spread_current: float
    spread_mean: float
    spread_std: float
    correlation: float
    signal_type: str  # 'ENTER_LONG_A', 'ENTER_LONG_B', 'EXIT', 'STOP_LOSS'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'


@dataclass
class PairStats:
    """Statistics for a pair."""
    stock_a: str
    stock_b: str
    correlation: float
    cointegration_pvalue: float
    is_cointegrated: bool
    half_life: float  # How many days spread takes to mean-revert
    current_zscore: float
    spread_mean: float
    spread_std: float


class PairsTradingEngine:
    """
    Statistical Arbitrage engine for Indian stock pairs.
    
    USAGE:
        engine = PairsTradingEngine()
        # Find best pairs
        pairs = engine.find_pairs()
        # Get signals
        signals = engine.generate_signals()
        # Backtest a pair
        results = engine.backtest_pair("SBIN", "ICICIBANK")
    """

    # Pre-defined pairs known to be correlated in Indian markets
    CANDIDATE_PAIRS = [
        ("SBIN", "ICICIBANK"),
        ("TCS", "INFY"),
        ("HDFCBANK", "KOTAKBANK"),
        ("SUNPHARMA", "CIPLA"),
        ("TATASTEEL", "JSWSTEEL"),
        ("NTPC", "POWERGRID"),
        ("RELIANCE", "ONGC"),
        ("HCLTECH", "WIPRO"),
        ("TITAN", "BAJFINANCE"),
        ("AXISBANK", "KOTAKBANK"),
        ("BHARTIARTL", "RELIANCE"),
        ("MARUTI", "EICHERMOT"),
        ("DRREDDY", "CIPLA"),
        ("COALINDIA", "NTPC"),
        ("HEROMOTOCO", "BAJAJ-AUTO"),
    ]

    def __init__(self, lookback_days: int = 252, entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5, stop_zscore: float = 3.0):
        """
        Args:
            lookback_days: Days of history for calculating spread statistics
            entry_zscore: Z-score threshold to enter a trade (default 2.0)
            exit_zscore: Z-score threshold to exit (mean reversion)
            stop_zscore: Z-score threshold for stop-loss (pair broke)
        """
        self.dm = DataManager()
        self.lookback_days = lookback_days
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore

    def get_pair_data(self, stock_a: str, stock_b: str, period: str = "2y") -> Tuple[pd.Series, pd.Series]:
        """Get aligned price data for two stocks."""
        df_a = self.dm.get_stock_data(stock_a, period=period)
        df_b = self.dm.get_stock_data(stock_b, period=period)

        if df_a.empty or df_b.empty:
            return pd.Series(), pd.Series()

        # Align on common dates
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < 100:
            return pd.Series(), pd.Series()

        return df_a.loc[common_idx, "close"], df_b.loc[common_idx, "close"]

    def calculate_spread(self, prices_a: pd.Series, prices_b: pd.Series) -> pd.Series:
        """
        Calculate the log price ratio spread.
        Using log ratio is better than simple difference because:
        - It's stationary (required for mean reversion)
        - It accounts for different price levels
        """
        return np.log(prices_a / prices_b)

    def test_cointegration(self, prices_a: pd.Series, prices_b: pd.Series) -> Tuple[float, bool]:
        """
        Engle-Granger cointegration test.
        If p-value < 0.05, the pair is cointegrated (good for pairs trading).
        """
        try:
            spread = self.calculate_spread(prices_a, prices_b)
            # ADF test on the spread
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(spread.dropna(), maxlag=20)
            pvalue = result[1]
            return pvalue, pvalue < 0.05
        except ImportError:
            # Fallback: simple correlation-based test
            correlation = prices_a.pct_change().corr(prices_b.pct_change())
            # If highly correlated returns, assume cointegrated
            return 1 - correlation, correlation > 0.7
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
            return 1.0, False

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion.
        This tells you approximately how many days the spread takes
        to revert halfway back to the mean.
        
        Shorter half-life = faster mean reversion = better for trading
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        # Align
        common = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common]
        spread_diff = spread_diff.loc[common]

        if len(spread_lag) < 30:
            return 999

        # OLS regression: spread_diff = beta * spread_lag + error
        beta = np.polyfit(spread_lag, spread_diff, 1)[0]

        if beta >= 0:
            return 999  # Not mean-reverting

        half_life = -np.log(2) / beta
        return max(half_life, 1)

    def analyze_pair(self, stock_a: str, stock_b: str) -> Optional[PairStats]:
        """Full analysis of a single pair."""
        prices_a, prices_b = self.get_pair_data(stock_a, stock_b)

        if prices_a.empty or prices_b.empty:
            return None

        # Correlation
        correlation = prices_a.pct_change().corr(prices_b.pct_change())

        # Cointegration
        pvalue, is_coint = self.test_cointegration(prices_a, prices_b)

        # Spread statistics
        spread = self.calculate_spread(prices_a, prices_b)
        spread_recent = spread.iloc[-self.lookback_days:] if len(spread) > self.lookback_days else spread
        spread_mean = spread_recent.mean()
        spread_std = spread_recent.std()
        current_zscore = (spread.iloc[-1] - spread_mean) / spread_std if spread_std > 0 else 0

        # Half-life
        half_life = self.calculate_half_life(spread_recent)

        return PairStats(
            stock_a=stock_a,
            stock_b=stock_b,
            correlation=round(correlation, 4),
            cointegration_pvalue=round(pvalue, 4),
            is_cointegrated=is_coint,
            half_life=round(half_life, 1),
            current_zscore=round(current_zscore, 2),
            spread_mean=round(spread_mean, 4),
            spread_std=round(spread_std, 4),
        )

    def find_pairs(self) -> List[PairStats]:
        """
        Analyze all candidate pairs and return valid ones.
        A valid pair has:
        - Correlation > 0.5
        - Cointegrated (or correlation > 0.7 as fallback)
        - Half-life < 60 days (reverts within 2 months)
        """
        logger.info("Analyzing pairs...")
        valid_pairs = []

        for stock_a, stock_b in self.CANDIDATE_PAIRS:
            try:
                stats = self.analyze_pair(stock_a, stock_b)
                if stats is None:
                    continue

                # Filter criteria
                if stats.correlation > 0.5 and stats.half_life < 60:
                    valid_pairs.append(stats)
                    logger.info(
                        f"  ✓ {stock_a}/{stock_b}: corr={stats.correlation:.2f}, "
                        f"half_life={stats.half_life:.0f}d, zscore={stats.current_zscore:+.2f}"
                    )
                else:
                    logger.debug(
                        f"  ✗ {stock_a}/{stock_b}: corr={stats.correlation:.2f}, "
                        f"half_life={stats.half_life:.0f}d (rejected)"
                    )
            except Exception as e:
                logger.warning(f"  Error analyzing {stock_a}/{stock_b}: {e}")

        # Sort by absolute z-score (biggest divergence = best opportunity)
        valid_pairs.sort(key=lambda x: abs(x.current_zscore), reverse=True)
        return valid_pairs

    def generate_signals(self) -> List[PairSignal]:
        """
        Generate trading signals for all valid pairs.
        
        Signal logic:
        - |z-score| > entry_zscore (2.0): ENTER trade
          - z > +2: Short A, Long B (A is overpriced relative to B)
          - z < -2: Long A, Short B (A is underpriced relative to B)
        - |z-score| < exit_zscore (0.5): EXIT (spread reverted)
        - |z-score| > stop_zscore (3.0): STOP LOSS (pair broke)
        """
        pairs = self.find_pairs()
        signals = []

        for pair in pairs:
            zscore = pair.current_zscore

            if abs(zscore) > self.stop_zscore:
                # Pair has diverged too much — stop loss zone
                signals.append(PairSignal(
                    stock_long=pair.stock_b if zscore > 0 else pair.stock_a,
                    stock_short=pair.stock_a if zscore > 0 else pair.stock_b,
                    zscore=zscore,
                    spread_current=pair.spread_mean + zscore * pair.spread_std,
                    spread_mean=pair.spread_mean,
                    spread_std=pair.spread_std,
                    correlation=pair.correlation,
                    signal_type="STOP_LOSS",
                    confidence="LOW",
                ))
            elif abs(zscore) > self.entry_zscore:
                # Good entry opportunity
                confidence = "HIGH" if pair.is_cointegrated else "MEDIUM"
                if zscore > self.entry_zscore:
                    # A is overpriced vs B → Short A, Long B
                    signals.append(PairSignal(
                        stock_long=pair.stock_b,
                        stock_short=pair.stock_a,
                        zscore=zscore,
                        spread_current=pair.spread_mean + zscore * pair.spread_std,
                        spread_mean=pair.spread_mean,
                        spread_std=pair.spread_std,
                        correlation=pair.correlation,
                        signal_type="ENTER",
                        confidence=confidence,
                    ))
                else:
                    # A is underpriced vs B → Long A, Short B
                    signals.append(PairSignal(
                        stock_long=pair.stock_a,
                        stock_short=pair.stock_b,
                        zscore=zscore,
                        spread_current=pair.spread_mean + zscore * pair.spread_std,
                        spread_mean=pair.spread_mean,
                        spread_std=pair.spread_std,
                        correlation=pair.correlation,
                        signal_type="ENTER",
                        confidence=confidence,
                    ))
            elif abs(zscore) < self.exit_zscore:
                # Spread has reverted — exit signal
                signals.append(PairSignal(
                    stock_long="",
                    stock_short="",
                    zscore=zscore,
                    spread_current=pair.spread_mean + zscore * pair.spread_std,
                    spread_mean=pair.spread_mean,
                    spread_std=pair.spread_std,
                    correlation=pair.correlation,
                    signal_type="EXIT",
                    confidence="HIGH",
                ))

        return signals

    def backtest_pair(self, stock_a: str, stock_b: str, capital: float = 1000000) -> dict:
        """
        Backtest pairs trading on a single pair.
        
        Strategy:
        - Enter when |z-score| > 2.0
        - Exit when |z-score| < 0.5 or > 3.0
        - Equal capital allocation to long and short legs
        """
        prices_a, prices_b = self.get_pair_data(stock_a, stock_b, period="5y")
        if prices_a.empty:
            return {}

        spread = self.calculate_spread(prices_a, prices_b)

        # Rolling statistics
        window = min(self.lookback_days, len(spread) // 3)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std

        # Simulate trades
        cash = capital
        position = None  # {'type': 'long_a' or 'long_b', 'entry_date', 'shares_a', 'shares_b', ...}
        trades = []

        for i in range(window, len(zscore)):
            z = zscore.iloc[i]
            date = zscore.index[i]
            price_a = prices_a.iloc[i]
            price_b = prices_b.iloc[i]

            if np.isnan(z):
                continue

            if position is None:
                # Look for entry
                if z > self.entry_zscore:
                    # Short A, Long B
                    alloc = cash * 0.45  # 45% each side, 10% reserve
                    shares_a = int(alloc / price_a)
                    shares_b = int(alloc / price_b)
                    position = {
                        "type": "short_a_long_b",
                        "entry_date": date,
                        "entry_z": z,
                        "entry_price_a": price_a,
                        "entry_price_b": price_b,
                        "shares_a": shares_a,
                        "shares_b": shares_b,
                    }
                elif z < -self.entry_zscore:
                    # Long A, Short B
                    alloc = cash * 0.45
                    shares_a = int(alloc / price_a)
                    shares_b = int(alloc / price_b)
                    position = {
                        "type": "long_a_short_b",
                        "entry_date": date,
                        "entry_z": z,
                        "entry_price_a": price_a,
                        "entry_price_b": price_b,
                        "shares_a": shares_a,
                        "shares_b": shares_b,
                    }
            else:
                # Check for exit
                should_exit = False
                exit_reason = ""

                if abs(z) < self.exit_zscore:
                    should_exit = True
                    exit_reason = "MEAN_REVERT"
                elif abs(z) > self.stop_zscore:
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                if should_exit:
                    # Calculate P&L
                    if position["type"] == "short_a_long_b":
                        pnl_a = (position["entry_price_a"] - price_a) * position["shares_a"]  # Short profit
                        pnl_b = (price_b - position["entry_price_b"]) * position["shares_b"]  # Long profit
                    else:
                        pnl_a = (price_a - position["entry_price_a"]) * position["shares_a"]  # Long profit
                        pnl_b = (position["entry_price_b"] - price_b) * position["shares_b"]  # Short profit

                    total_pnl = pnl_a + pnl_b
                    cash += total_pnl
                    days_held = (date - position["entry_date"]).days

                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": date,
                        "type": position["type"],
                        "entry_z": position["entry_z"],
                        "exit_z": z,
                        "pnl": total_pnl,
                        "pnl_pct": (total_pnl / capital) * 100,
                        "days_held": days_held,
                        "exit_reason": exit_reason,
                    })
                    position = None

        # Summary
        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)

        return {
            "stock_a": stock_a,
            "stock_b": stock_b,
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / max(len(trades), 1) * 100,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / capital) * 100,
            "avg_trade_pnl": total_pnl / max(len(trades), 1),
            "avg_days_held": np.mean([t["days_held"] for t in trades]) if trades else 0,
            "final_capital": cash,
            "trades": trades,
        }

    @staticmethod
    def print_report(pairs: List[PairStats], signals: List[PairSignal]):
        """Print a formatted pairs trading report."""
        print("\n" + "=" * 70)
        print("  PAIRS TRADING — STATISTICAL ARBITRAGE REPORT")
        print("=" * 70)

        print(f"\n  Valid Pairs Found: {len(pairs)}")
        print(f"\n  {'Pair':<25} {'Corr':>6} {'Coint?':>7} {'HalfLife':>9} {'Z-Score':>8} {'Status':<15}")
        print("  " + "-" * 72)

        for p in pairs:
            if abs(p.current_zscore) > 2:
                status = "⚡ DIVERGED"
            elif abs(p.current_zscore) < 0.5:
                status = "✓ At Mean"
            else:
                status = "~ Normal"

            print(
                f"  {p.stock_a + '/' + p.stock_b:<25} "
                f"{p.correlation:>6.3f} "
                f"{'Yes' if p.is_cointegrated else 'No':>7} "
                f"{p.half_life:>7.0f}d "
                f"{p.current_zscore:>+7.2f} "
                f"{status:<15}"
            )

        if signals:
            print(f"\n\n  ACTIVE SIGNALS:")
            print(f"  {'Signal':<10} {'Long':<12} {'Short':<12} {'Z-Score':>8} {'Confidence':<10}")
            print("  " + "-" * 55)

            for s in signals:
                if s.signal_type == "ENTER":
                    print(
                        f"  {'ENTER':<10} {s.stock_long:<12} {s.stock_short:<12} "
                        f"{s.zscore:>+7.2f} {s.confidence:<10}"
                    )
                elif s.signal_type == "STOP_LOSS":
                    print(
                        f"  {'⚠ STOP':<10} {s.stock_long:<12} {s.stock_short:<12} "
                        f"{s.zscore:>+7.2f} {'DANGER':<10}"
                    )

    @staticmethod
    def print_backtest(results: dict):
        """Print backtest results for a pair."""
        if not results:
            print("  No results.")
            return

        print(f"\n  BACKTEST: {results['stock_a']} / {results['stock_b']}")
        print("  " + "-" * 50)
        print(f"  Total Trades:     {results['total_trades']}")
        print(f"  Win Rate:         {results['win_rate']:.1f}%")
        print(f"  Total Return:     {results['total_return_pct']:+.2f}%")
        print(f"  Final Capital:    ₹{results['final_capital']:,.0f}")
        print(f"  Avg Days Held:    {results['avg_days_held']:.0f}")

        if results["trades"]:
            print(f"\n  {'#':<3} {'Entry':<12} {'Exit':<12} {'Type':<18} {'P&L%':>7} {'Days':>5} {'Reason':<12}")
            print("  " + "-" * 72)
            for i, t in enumerate(results["trades"][-10:], 1):  # Last 10 trades
                trade_type = "Long A / Short B" if t["type"] == "long_a_short_b" else "Short A / Long B"
                print(
                    f"  {i:<3} {str(t['entry_date'])[:10]:<12} {str(t['exit_date'])[:10]:<12} "
                    f"{trade_type:<18} {t['pnl_pct']:>+6.2f}% {t['days_held']:>5} {t['exit_reason']:<12}"
                )
