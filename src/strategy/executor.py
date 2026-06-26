"""
Strategy Executor — Generates BUY/SELL signals and manages the portfolio.

THIS IS THE "BRAIN" THAT DECIDES:
- What to buy
- How much to buy (position sizing)
- When to sell
- How to handle risk

SIGNAL GENERATION FLOW:
1. Run the StockRanker to get top stocks with composite scores
2. Compare with current holdings
3. Generate signals:
   - BUY: Stock in top 15 but not in portfolio
   - SELL: Stock in portfolio but dropped out of top 15
   - HOLD: Stock still in top 15 and in portfolio
   - STOP_LOSS: Stock hit stop-loss level
"""
import pandas as pd
from datetime import datetime
from loguru import logger
from dataclasses import dataclass

from src.utils.helpers import load_config
from src.ranking.ranker import StockRanker


@dataclass
class Signal:
    """A trading signal."""
    symbol: str
    action: str       # 'BUY', 'SELL', 'HOLD'
    reason: str        # Why this signal was generated
    score: float       # Composite score
    target_pct: float  # Target portfolio percentage
    current_price: float = 0.0
    stop_loss: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StrategyExecutor:
    """
    Runs the strategy and generates actionable signals.
    
    USAGE:
        executor = StrategyExecutor()
        signals = executor.generate_signals()
        for s in signals:
            print(f"{s.action} {s.symbol} @ ₹{s.current_price}")
    """

    def __init__(self, use_breeze: bool = False):
        self.config = load_config()
        self.ranker = StockRanker(use_breeze=use_breeze)
        self.portfolio_config = self.config.get("portfolio", {})
        self.risk_config = self.config.get("risk", {})

    def generate_signals(
        self, current_holdings: dict = None, symbols: list = None
    ) -> list:
        """
        Generate trading signals.
        
        Args:
            current_holdings: {symbol: {shares, entry_price}} or None for fresh start
            symbols: Custom symbol list (default: Nifty 500)
        
        Returns:
            List of Signal objects
        """
        if current_holdings is None:
            current_holdings = {}

        # 1. Rank stocks
        logger.info("Ranking stocks...")
        rankings = self.ranker.rank_stocks(symbols=symbols)
        if rankings.empty:
            return []

        max_stocks = self.portfolio_config.get("max_stocks", 15)
        top_symbols = set(rankings.head(max_stocks)["symbol"].tolist())
        current_symbols = set(current_holdings.keys())

        signals = []

        # 2. SELL signals — stocks that dropped out of top N
        for symbol in current_symbols - top_symbols:
            entry = current_holdings[symbol]
            signals.append(Signal(
                symbol=symbol,
                action="SELL",
                reason=f"Dropped out of top {max_stocks}",
                score=0,
                target_pct=0,
                current_price=entry.get("current_price", entry.get("entry_price", 0)),
            ))

        # 3. BUY signals — new stocks entering top N
        for _, row in rankings.head(max_stocks).iterrows():
            symbol = row["symbol"]
            if symbol not in current_symbols:
                stop_loss_price = (
                    row.get("current_price", 0) *
                    (1 - self.risk_config.get("stop_loss_pct", 8) / 100)
                )
                signals.append(Signal(
                    symbol=symbol,
                    action="BUY",
                    reason=f"Rank #{int(row.get('rank', 0))} (Score: {row['composite_score']:.2f})",
                    score=row["composite_score"],
                    target_pct=self.portfolio_config.get("max_position_pct", 7),
                    current_price=row.get("current_price", 0) or 0,
                    stop_loss=stop_loss_price,
                ))

        # 4. HOLD signals — stocks still in top N
        for _, row in rankings.head(max_stocks).iterrows():
            symbol = row["symbol"]
            if symbol in current_symbols:
                signals.append(Signal(
                    symbol=symbol,
                    action="HOLD",
                    reason=f"Still in top {max_stocks} (Rank #{int(row.get('rank', 0))})",
                    score=row["composite_score"],
                    target_pct=self.portfolio_config.get("max_position_pct", 7),
                ))

        # Sort: SELL first, then BUY, then HOLD
        order = {"SELL": 0, "BUY": 1, "HOLD": 2}
        signals.sort(key=lambda s: order.get(s.action, 3))

        return signals

    def print_signals(self, signals: list):
        """Pretty-print signals."""
        print("\n" + "=" * 70)
        print(f"  TRADING SIGNALS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)

        for s in signals:
            icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(s.action, "⚪")
            price_str = f"₹{s.current_price:,.1f}" if s.current_price else ""
            sl_str = f"SL: ₹{s.stop_loss:,.1f}" if s.stop_loss else ""

            print(f"  {icon} {s.action:4} | {s.symbol:15} | {price_str:>12} | {sl_str:>14} | {s.reason}")

        buy_count = sum(1 for s in signals if s.action == "BUY")
        sell_count = sum(1 for s in signals if s.action == "SELL")
        hold_count = sum(1 for s in signals if s.action == "HOLD")
        print("-" * 70)
        print(f"  Summary: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD")
        print("=" * 70)


def generate_quick_signals(symbols: list = None) -> list:
    """
    Quick signal generation for a small set of stocks.
    Uses only momentum + quality (no web scraping).
    """
    if symbols is None:
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
            "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "HCLTECH",
            "TATAMOTORS", "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND",
        ]

    ranker = StockRanker()
    rankings = ranker.rank_quick(symbols)

    if rankings.empty:
        return []

    signals = []
    for _, row in rankings.head(15).iterrows():
        signals.append(Signal(
            symbol=row["symbol"],
            action="BUY" if row["composite_score"] > 5 else "HOLD",
            reason=f"Score: {row['composite_score']:.2f} (M:{row['momentum_score']:.1f} Q:{row['quality_score']:.1f})",
            score=row["composite_score"],
            target_pct=7.0,
        ))

    return signals
