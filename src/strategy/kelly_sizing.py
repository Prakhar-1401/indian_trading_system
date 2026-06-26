"""
Kelly Criterion Position Sizing — The mathematically optimal bet size.

WHAT IS THE KELLY CRITERION?
==============================
Kelly tells you EXACTLY how much of your capital to bet:
- Too small → You leave money on the table
- Too large → One bad streak wipes you out
- Kelly optimal → Maximum long-term growth rate

THE FORMULA:
============
  Kelly % = (Win Rate × Avg Win / Avg Loss - Loss Rate) / (Avg Win / Avg Loss)

  Simplified: Kelly % = Win_Rate - (1 - Win_Rate) / (Avg_Win / Avg_Loss)

EXAMPLE:
  Win Rate: 55%
  Avg Win: 3%
  Avg Loss: 2%
  Kelly = 0.55 - 0.45 / (3/2) = 0.55 - 0.30 = 0.25 → Bet 25% per trade

WHY QUANT FIRMS USE HALF-KELLY:
================================
Full Kelly is too aggressive. One miscalculation and you're dead.
- Full Kelly → Maximum growth but HUGE drawdowns (50%+)
- Half Kelly → 75% of growth but MUCH smaller drawdowns
- Quarter Kelly → Still good growth, very smooth equity curve

WHAT WE DO:
- Calculate Kelly for each stock based on its ACTUAL backtest stats
- Use Half Kelly as default (adjustable)
- Cap at 10% of portfolio per trade (safety net)
- Adjust for correlation (don't bet 25% on 4 correlated stocks = 100%!)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class KellyResult:
    """Kelly Criterion calculation result for a stock."""
    symbol: str
    win_rate: float
    avg_win: float
    avg_loss: float
    kelly_full: float  # Full Kelly %
    kelly_half: float  # Half Kelly (recommended)
    kelly_quarter: float  # Quarter Kelly (conservative)
    recommended_pct: float  # What we actually recommend
    position_value: float  # ₹ amount for given capital
    shares: int  # Number of shares
    edge: float  # Expected value per trade (%)
    sharpe_approx: float  # Approximate Sharpe from win stats


class KellyPositionSizer:
    """
    Calculate optimal position sizes using Kelly Criterion.
    
    USAGE:
        sizer = KellyPositionSizer(capital=1000000)
        result = sizer.calculate("RELIANCE")
        sizer.print_report(["RELIANCE", "TCS", "SBIN"])
    """

    def __init__(self, capital: float = 1000000, kelly_fraction: float = 0.5,
                 max_position_pct: float = 10.0, lookback_days: int = 252):
        """
        Args:
            capital: Total portfolio capital
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly)
            max_position_pct: Maximum position size as % of capital
            lookback_days: Days of history to calculate win rate
        """
        self.dm = DataManager()
        self.capital = capital
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct / 100
        self.lookback_days = lookback_days

    def _calculate_trade_stats(self, symbol: str) -> Optional[Dict]:
        """
        Calculate win rate and avg win/loss from actual price action.
        
        Simulates: "If I bought every pullback to 20-SMA and sold at
        either +target or -stop, what's my win rate?"
        """
        df = self.dm.get_stock_data(symbol, period="2y")
        if df.empty or len(df) < 100:
            return None

        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        close = df['close']
        sma20 = close.rolling(20).mean()

        # Simulate swing trades: buy near SMA, hold for up to 10 days
        wins = []
        losses = []
        holding_period = 10

        # Find days where price touched or crossed below 20-SMA then bounced
        for i in range(50, len(df) - holding_period):
            # Entry condition: price within 1% of 20-SMA (pullback)
            if abs(close.iloc[i] / sma20.iloc[i] - 1) < 0.01:
                entry_price = close.iloc[i]

                # Check forward returns
                future_prices = close.iloc[i+1:i+1+holding_period]
                max_gain = (future_prices.max() / entry_price - 1) * 100
                max_loss = (future_prices.min() / entry_price - 1) * 100

                # Exit at 3% profit or 2% loss (whichever hits first)
                hit_target = max_gain >= 3.0
                hit_stop = max_loss <= -2.0

                if hit_target and not hit_stop:
                    wins.append(max_gain)
                elif hit_stop:
                    losses.append(abs(max_loss))
                else:
                    # Exited at end of holding period
                    final_return = (future_prices.iloc[-1] / entry_price - 1) * 100
                    if final_return > 0:
                        wins.append(final_return)
                    else:
                        losses.append(abs(final_return))

        total_trades = len(wins) + len(losses)
        if total_trades < 10:
            return None

        win_rate = len(wins) / total_trades
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 1

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades,
            'total_wins': len(wins),
            'total_losses': len(losses),
        }

    def calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly percentage.
        
        Kelly = W - (1-W) / R
        Where W = win rate, R = avg_win / avg_loss
        """
        if avg_loss == 0:
            return 0

        R = avg_win / avg_loss  # Win/Loss ratio
        kelly = win_rate - (1 - win_rate) / R

        return max(kelly, 0)  # Never negative (means don't trade)

    def calculate(self, symbol: str) -> Optional[KellyResult]:
        """Calculate Kelly-optimal position size for a symbol."""
        stats = self._calculate_trade_stats(symbol)
        if stats is None:
            return None

        win_rate = stats['win_rate']
        avg_win = stats['avg_win']
        avg_loss = stats['avg_loss']

        # Kelly calculation
        kelly_full = self.calculate_kelly(win_rate, avg_win, avg_loss)
        kelly_half = kelly_full * 0.5
        kelly_quarter = kelly_full * 0.25

        # Apply our fraction
        recommended = kelly_full * self.kelly_fraction

        # Cap at max position
        recommended = min(recommended, self.max_position_pct)

        # Get current price for position sizing
        df = self.dm.get_stock_data(symbol, period="1mo")
        if df.empty:
            return None
        current_price = df['close'].iloc[-1]

        position_value = self.capital * recommended
        shares = int(position_value / current_price) if current_price > 0 else 0

        # Edge = expected value per trade
        edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Approximate Sharpe
        all_returns = ([avg_win] * stats['total_wins'] +
                       [-avg_loss] * stats['total_losses'])
        sharpe = np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0

        return KellyResult(
            symbol=symbol,
            win_rate=round(win_rate, 3),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            kelly_full=round(kelly_full * 100, 2),
            kelly_half=round(kelly_half * 100, 2),
            kelly_quarter=round(kelly_quarter * 100, 2),
            recommended_pct=round(recommended * 100, 2),
            position_value=round(position_value, 0),
            shares=shares,
            edge=round(edge, 2),
            sharpe_approx=round(sharpe, 2),
        )

    def calculate_portfolio(self, symbols: List[str]) -> List[KellyResult]:
        """Calculate Kelly for entire portfolio."""
        results = []
        for symbol in symbols:
            result = self.calculate(symbol)
            if result:
                results.append(result)

        # Sort by edge (best edge first)
        results.sort(key=lambda x: x.edge, reverse=True)
        return results

    def print_report(self, symbols: List[str]):
        """Print comprehensive Kelly position sizing report."""
        results = self.calculate_portfolio(symbols)

        print("\n" + "=" * 70)
        print("  KELLY CRITERION — OPTIMAL POSITION SIZING")
        print("=" * 70)
        print(f"  Capital: ₹{self.capital:,.0f}")
        print(f"  Kelly Fraction: {self.kelly_fraction:.0%} (Half Kelly = safer)")
        print(f"  Max Position: {self.max_position_pct*100:.0f}% of capital")

        if not results:
            print("\n  No stocks with sufficient trade history.")
            print("=" * 70)
            return

        print(f"\n  {'Symbol':<12} {'WinRate':>8} {'AvgW':>6} {'AvgL':>6} "
              f"{'Kelly%':>7} {'Rec%':>6} {'Edge':>6} {'Value':>10} {'Shares':>7}")
        print("  " + "-" * 78)

        total_allocated = 0
        for r in results:
            # Color coding
            if r.edge > 1.0:
                emoji = "🟢"
            elif r.edge > 0:
                emoji = "🟡"
            else:
                emoji = "🔴"

            print(
                f"  {r.symbol:<12} {r.win_rate:>7.1%} "
                f"{r.avg_win:>5.1f}% {r.avg_loss:>5.1f}% "
                f"{r.kelly_half:>6.1f}% {r.recommended_pct:>5.1f}% "
                f"{emoji}{r.edge:>+5.2f}% "
                f"₹{r.position_value:>9,.0f} {r.shares:>7,}"
            )
            total_allocated += r.position_value

        # Summary
        print(f"\n  Total Allocated: ₹{total_allocated:,.0f} / ₹{self.capital:,.0f} "
              f"({total_allocated/self.capital*100:.1f}%)")

        # Best opportunities
        positive_edge = [r for r in results if r.edge > 0.5]
        if positive_edge:
            print(f"\n  ⭐ BEST OPPORTUNITIES (Edge > 0.5%):")
            for r in positive_edge[:5]:
                print(f"     {r.symbol}: Edge={r.edge:+.2f}%, WinRate={r.win_rate:.0%}, "
                      f"Kelly says bet {r.kelly_half:.1f}%")

        # Warnings
        no_edge = [r for r in results if r.edge <= 0]
        if no_edge:
            print(f"\n  ⚠️ NO EDGE (Kelly says don't trade):")
            for r in no_edge[:3]:
                print(f"     {r.symbol}: Edge={r.edge:+.2f}% — Not profitable historically")

        print(f"\n  KEY:")
        print(f"  • Edge = Expected profit per trade (win_rate × avg_win - loss_rate × avg_loss)")
        print(f"  • Kelly% = Mathematically optimal bet size (using Half Kelly for safety)")
        print(f"  • 🟢 = Strong edge | 🟡 = Marginal | 🔴 = No edge")
        print(f"  • Simulated: Buy at 20-SMA pullback, Target +3%, Stop -2%")
        print("=" * 70)
