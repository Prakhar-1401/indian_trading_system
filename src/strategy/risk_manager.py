"""
Dynamic ATR-Based Risk Management

WHAT IS ATR (Average True Range)?
==================================
ATR measures volatility — how much a stock moves per day on average.
- High ATR = volatile stock (e.g., TATAMOTORS moves ₹20/day)
- Low ATR = stable stock (e.g., HDFC moves ₹5/day)

WHY FIXED STOP-LOSSES ARE BAD:
- A 5% stop on a volatile stock gets hit every week (whipsawed out)
- A 5% stop on a stable stock is too loose (lose too much)

ATR-BASED STOPS (What quant firms use):
- Stop = Entry Price - (2 × ATR)
- This means: "exit if the stock moves 2 average days against me"
- Volatile stocks get wider stops (room to breathe)
- Stable stocks get tighter stops (protect capital)

EXAMPLE:
  RELIANCE (ATR = ₹40): Stop = Entry - (2 × ₹40) = Entry - ₹80
  HDFC (ATR = ₹15): Stop = Entry - (2 × ₹15) = Entry - ₹30
  
  Both exit after approximately the same "abnormal" move.

POSITION SIZING WITH ATR (Kelly-inspired):
- Risk per trade = 2% of capital
- Position size = (2% of capital) / (2 × ATR)
- This ensures EVERY trade risks the same ₹ amount regardless of stock
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class RiskParameters:
    """Dynamic risk parameters for a stock."""
    symbol: str
    current_price: float
    atr_14: float  # 14-day ATR
    atr_pct: float  # ATR as % of price
    volatility_regime: str  # 'HIGH', 'NORMAL', 'LOW'
    stop_loss_price: float
    stop_loss_pct: float
    trailing_stop_price: float
    trailing_stop_pct: float
    position_size_shares: int
    position_size_value: float
    risk_per_trade: float


class DynamicRiskManager:
    """
    ATR-based dynamic risk management system.
    
    Instead of fixed 8% stops, it calculates optimal stops based on
    each stock's volatility. More volatile = wider stops.
    
    USAGE:
        rm = DynamicRiskManager(capital=1000000)
        params = rm.calculate_risk("RELIANCE")
        rm.print_risk_report(["RELIANCE", "TCS", "COALINDIA"])
    """

    def __init__(self, capital: float = 1000000, risk_per_trade_pct: float = 2.0,
                 atr_multiplier_stop: float = 2.0, atr_multiplier_trailing: float = 3.0):
        """
        Args:
            capital: Total portfolio capital
            risk_per_trade_pct: Max % of capital to risk per trade (Kelly-inspired)
            atr_multiplier_stop: ATR multiplier for initial stop-loss
            atr_multiplier_trailing: ATR multiplier for trailing stop
        """
        self.dm = DataManager()
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct / 100
        self.atr_mult_stop = atr_multiplier_stop
        self.atr_mult_trail = atr_multiplier_trailing

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        
        True Range = max of:
          - High - Low (today's range)
          - |High - Previous Close| (gap up measurement)
          - |Low - Previous Close| (gap down measurement)
        """
        if len(df) < period + 1:
            return 0

        high = df['high']
        low = df['low']
        close = df['close']

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return atr

    def detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """
        Detect if current volatility is HIGH, NORMAL, or LOW
        compared to historical volatility.
        
        This is important because:
        - In HIGH volatility: widen stops even more, reduce position size
        - In LOW volatility: tighten stops, can increase position size
        """
        if len(df) < 60:
            return "NORMAL"

        # Current ATR vs 60-day historical ATR
        current_atr = self.calculate_atr(df.tail(20), period=14)
        historical_atr = self.calculate_atr(df.tail(60), period=14)

        if historical_atr == 0:
            return "NORMAL"

        ratio = current_atr / historical_atr

        if ratio > 1.5:
            return "HIGH"
        elif ratio < 0.7:
            return "LOW"
        else:
            return "NORMAL"

    def calculate_risk(self, symbol: str, entry_price: float = None) -> Optional[RiskParameters]:
        """
        Calculate complete risk parameters for a stock.
        
        Returns optimal: stop-loss, trailing stop, position size.
        """
        df = self.dm.get_stock_data(symbol, period="6mo")
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        current_price = entry_price or df['close'].iloc[-1]
        atr = self.calculate_atr(df)
        atr_pct = (atr / current_price) * 100

        # Volatility regime
        regime = self.detect_volatility_regime(df)

        # Adjust ATR multiplier based on regime
        regime_adjustment = {"HIGH": 1.5, "NORMAL": 1.0, "LOW": 0.75}
        adj = regime_adjustment[regime]

        # Stop-loss
        stop_distance = atr * self.atr_mult_stop * adj
        stop_loss_price = current_price - stop_distance
        stop_loss_pct = (stop_distance / current_price) * 100

        # Trailing stop (wider than initial)
        trail_distance = atr * self.atr_mult_trail * adj
        trailing_stop_price = current_price - trail_distance
        trailing_stop_pct = (trail_distance / current_price) * 100

        # Position sizing (risk-based)
        risk_amount = self.capital * self.risk_per_trade_pct
        position_size_shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        position_size_value = position_size_shares * current_price

        # Cap at 10% of portfolio
        max_position = self.capital * 0.10
        if position_size_value > max_position:
            position_size_shares = int(max_position / current_price)
            position_size_value = position_size_shares * current_price

        return RiskParameters(
            symbol=symbol,
            current_price=round(current_price, 2),
            atr_14=round(atr, 2),
            atr_pct=round(atr_pct, 2),
            volatility_regime=regime,
            stop_loss_price=round(stop_loss_price, 2),
            stop_loss_pct=round(stop_loss_pct, 2),
            trailing_stop_price=round(trailing_stop_price, 2),
            trailing_stop_pct=round(trailing_stop_pct, 2),
            position_size_shares=position_size_shares,
            position_size_value=round(position_size_value, 2),
            risk_per_trade=round(risk_amount, 2),
        )

    def print_risk_report(self, symbols: list, capital: float = None):
        """Print a comprehensive risk report for multiple stocks."""
        if capital:
            self.capital = capital

        print("\n" + "=" * 70)
        print("  DYNAMIC RISK MANAGEMENT (ATR-Based)")
        print("=" * 70)
        print(f"  Portfolio Capital: ₹{self.capital:,.0f}")
        print(f"  Risk per Trade: {self.risk_per_trade_pct*100:.1f}% (₹{self.capital * self.risk_per_trade_pct:,.0f})")
        print(f"  ATR Stop Multiplier: {self.atr_mult_stop}x | Trailing: {self.atr_mult_trail}x")

        print(f"\n  {'Symbol':<12} {'Price':>8} {'ATR':>7} {'ATR%':>6} {'Regime':<7} "
              f"{'Stop':>8} {'Stop%':>6} {'Trail%':>7} {'Shares':>7} {'Value':>10}")
        print("  " + "-" * 85)

        total_allocated = 0
        for symbol in symbols:
            params = self.calculate_risk(symbol)
            if params is None:
                print(f"  {symbol:<12} {'No data':<60}")
                continue

            regime_emoji = {"HIGH": "🔴", "NORMAL": "🟡", "LOW": "🟢"}
            emoji = regime_emoji.get(params.volatility_regime, "⚪")

            print(
                f"  {params.symbol:<12} "
                f"₹{params.current_price:>7,.0f} "
                f"₹{params.atr_14:>5,.0f} "
                f"{params.atr_pct:>5.1f}% "
                f"{emoji}{params.volatility_regime:<6} "
                f"₹{params.stop_loss_price:>7,.0f} "
                f"{params.stop_loss_pct:>5.1f}% "
                f"{params.trailing_stop_pct:>6.1f}% "
                f"{params.position_size_shares:>7,} "
                f"₹{params.position_size_value:>9,.0f}"
            )
            total_allocated += params.position_size_value

        print(f"\n  Total Allocated: ₹{total_allocated:,.0f} / ₹{self.capital:,.0f} "
              f"({total_allocated/self.capital*100:.1f}%)")
        print(f"  Cash Reserve: ₹{self.capital - total_allocated:,.0f}")

        print("\n  KEY:")
        print("  • ATR% = daily volatility as % of price")
        print("  • Stop = ATR × 2 below entry (adaptive to volatility)")
        print("  • Shares = max shares to risk only 2% of capital")
        print("  • 🔴 HIGH vol = wider stops, smaller position")
        print("  • 🟢 LOW vol = tighter stops, larger position")
        print("=" * 70)
