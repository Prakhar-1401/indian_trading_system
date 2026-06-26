"""
Scheduled Daily Rebalance — Automatically adjust portfolio weights.

WHAT IS REBALANCING?
=====================
Over time, winning stocks grow to dominate your portfolio.
If RELIANCE goes up 50%, it might become 30% of your portfolio.
That's too much risk in one stock.

Rebalancing means: periodically reset back to TARGET weights.

TYPES:
======
1. CALENDAR: Rebalance every week/month regardless
2. THRESHOLD: Rebalance only when a position drifts >5% from target
3. SIGNAL-BASED: Rebalance when strategy signals change

WE USE: Threshold-based with signal confirmation.
- If position > 12% of portfolio → trim
- If position < 3% of target weight → add
- Only rebalance on signal change or weekly
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger

from src.data.fetcher import DataManager


@dataclass
class RebalanceAction:
    """A rebalancing action to take."""
    symbol: str
    action: str  # BUY_MORE, TRIM, EXIT, NEW_ENTRY
    current_weight: float
    target_weight: float
    shares_to_trade: int
    value_to_trade: float
    reason: str


class PortfolioRebalancer:
    """
    Automated portfolio rebalancing engine.
    
    USAGE:
        rebalancer = PortfolioRebalancer(capital=1000000)
        actions = rebalancer.calculate_rebalance(current_positions, target_allocation)
        rebalancer.print_rebalance_plan(actions)
    """

    def __init__(self, capital: float = 1000000, 
                 max_position_pct: float = 12.0,
                 min_position_pct: float = 3.0,
                 rebalance_threshold: float = 3.0):
        """
        Args:
            capital: Total portfolio value
            max_position_pct: Maximum single position (% of portfolio)
            min_position_pct: Minimum meaningful position (%)
            rebalance_threshold: Drift threshold to trigger rebalance (%)
        """
        self.dm = DataManager()
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.rebalance_threshold = rebalance_threshold

    def get_target_allocation(self, rankings: pd.DataFrame = None,
                              num_stocks: int = 10) -> Dict[str, float]:
        """
        Calculate target portfolio allocation based on rankings.
        
        Uses score-weighted allocation:
        - Higher ranked stocks get larger weights
        - Cap at max_position_pct
        - Equal weight fallback if no rankings
        """
        if rankings is None or rankings.empty:
            # Generate fresh rankings
            from src.ranking.ranker import StockRanker
            symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN",
                "BHARTIARTL", "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE",
                "ITC", "LT", "TITAN", "MARUTI", "HCLTECH",
            ]
            ranker = StockRanker()
            rankings = ranker.rank_quick(symbols)

        if rankings.empty:
            return {}

        # Take top N stocks
        top = rankings.head(num_stocks).copy()

        # Score-weighted allocation
        if 'composite_score' in top.columns:
            scores = top['composite_score'].clip(lower=0)
            total_score = scores.sum()
            if total_score > 0:
                weights = scores / total_score
            else:
                weights = pd.Series([1/num_stocks] * len(top), index=top.index)
        else:
            weights = pd.Series([1/num_stocks] * len(top), index=top.index)

        # Cap individual weights
        max_w = self.max_position_pct / 100
        weights = weights.clip(upper=max_w)

        # Renormalize to sum to ~90% (keep 10% cash)
        weights = weights / weights.sum() * 0.90

        allocation = {}
        for idx, row in top.iterrows():
            symbol = row['symbol']
            allocation[symbol] = round(float(weights.iloc[top.index.get_loc(idx)]), 4)

        return allocation

    def calculate_rebalance(self, current_positions: Dict[str, Dict],
                           target_allocation: Dict[str, float]) -> List[RebalanceAction]:
        """
        Calculate what trades are needed to rebalance.
        
        Args:
            current_positions: {symbol: {'quantity': int, 'current_price': float}}
            target_allocation: {symbol: target_weight (0-1)}
        """
        actions = []

        # Calculate current weights
        total_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in current_positions.values()
        )
        # Add cash
        invested = total_value
        total_portfolio = self.capital  # Use total capital as denominator

        current_weights = {}
        for symbol, pos in current_positions.items():
            pos_value = pos['quantity'] * pos['current_price']
            current_weights[symbol] = pos_value / total_portfolio if total_portfolio > 0 else 0

        # Check each target position
        all_symbols = set(list(target_allocation.keys()) + list(current_positions.keys()))

        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0)
            target_w = target_allocation.get(symbol, 0)
            drift = (current_w - target_w) * 100  # In percentage points

            # Get current price
            if symbol in current_positions:
                price = current_positions[symbol]['current_price']
            else:
                price = self._get_price(symbol)
                if price == 0:
                    continue

            # Decide action
            if target_w == 0 and current_w > 0:
                # Exit entirely
                shares = current_positions[symbol]['quantity']
                actions.append(RebalanceAction(
                    symbol=symbol,
                    action="EXIT",
                    current_weight=round(current_w * 100, 2),
                    target_weight=0,
                    shares_to_trade=shares,
                    value_to_trade=round(shares * price, 0),
                    reason="No longer in target allocation"
                ))

            elif current_w == 0 and target_w > 0:
                # New entry
                value = target_w * total_portfolio
                shares = int(value / price)
                if shares > 0:
                    actions.append(RebalanceAction(
                        symbol=symbol,
                        action="NEW_ENTRY",
                        current_weight=0,
                        target_weight=round(target_w * 100, 2),
                        shares_to_trade=shares,
                        value_to_trade=round(shares * price, 0),
                        reason="New addition to portfolio"
                    ))

            elif drift > self.rebalance_threshold:
                # Overweight → Trim
                trim_value = drift / 100 * total_portfolio
                shares = int(trim_value / price)
                if shares > 0:
                    actions.append(RebalanceAction(
                        symbol=symbol,
                        action="TRIM",
                        current_weight=round(current_w * 100, 2),
                        target_weight=round(target_w * 100, 2),
                        shares_to_trade=shares,
                        value_to_trade=round(shares * price, 0),
                        reason=f"Overweight by {drift:.1f}pp"
                    ))

            elif drift < -self.rebalance_threshold:
                # Underweight → Buy more
                add_value = abs(drift) / 100 * total_portfolio
                shares = int(add_value / price)
                if shares > 0:
                    actions.append(RebalanceAction(
                        symbol=symbol,
                        action="BUY_MORE",
                        current_weight=round(current_w * 100, 2),
                        target_weight=round(target_w * 100, 2),
                        shares_to_trade=shares,
                        value_to_trade=round(shares * price, 0),
                        reason=f"Underweight by {abs(drift):.1f}pp"
                    ))

        # Sort: exits first, then trims, then buys
        priority = {"EXIT": 0, "TRIM": 1, "BUY_MORE": 2, "NEW_ENTRY": 3}
        actions.sort(key=lambda x: priority.get(x.action, 9))

        return actions

    def _get_price(self, symbol: str) -> float:
        """Get current price."""
        df = self.dm.get_stock_data(symbol, period="5d")
        return df['close'].iloc[-1] if not df.empty else 0

    def print_rebalance_plan(self, actions: List[RebalanceAction]):
        """Print the rebalance plan."""
        print("\n" + "=" * 70)
        print("  📊 PORTFOLIO REBALANCE PLAN")
        print("=" * 70)

        if not actions:
            print("\n  ✅ Portfolio is balanced. No actions needed.")
            print(f"  (Drift threshold: ±{self.rebalance_threshold}%)")
            print("=" * 70)
            return

        total_buys = sum(a.value_to_trade for a in actions if a.action in ["BUY_MORE", "NEW_ENTRY"])
        total_sells = sum(a.value_to_trade for a in actions if a.action in ["TRIM", "EXIT"])

        print(f"\n  Actions Required: {len(actions)}")
        print(f"  Total to Buy: ₹{total_buys:,.0f}")
        print(f"  Total to Sell: ₹{total_sells:,.0f}")
        print(f"  Net Cash Flow: ₹{total_sells - total_buys:,.0f}")

        print(f"\n  {'Symbol':<12} {'Action':<10} {'CurrWt':>7} {'TgtWt':>7} "
              f"{'Shares':>7} {'Value':>10} Reason")
        print("  " + "-" * 75)

        for a in actions:
            emoji = {"EXIT": "🔴", "TRIM": "🟡", "BUY_MORE": "🟢", "NEW_ENTRY": "🆕"}.get(a.action, "⚪")
            print(
                f"  {a.symbol:<12} {emoji}{a.action:<9} "
                f"{a.current_weight:>6.1f}% {a.target_weight:>6.1f}% "
                f"{a.shares_to_trade:>7} ₹{a.value_to_trade:>9,.0f} "
                f"{a.reason}"
            )

        print(f"\n  ⚠️ Review before executing. Use paper trading mode to test.")
        print("=" * 70)
