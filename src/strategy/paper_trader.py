"""
Paper Trading Mode — Practice with real data, fake money.

WHAT IS PAPER TRADING?
=======================
Execute trades with REAL market data but SIMULATED capital.
- Track your P&L as if you were trading live
- Build confidence before going live
- Test strategies in real-time without risk

HOW IT WORKS:
=============
1. Start with virtual capital (e.g., ₹10,00,000)
2. System generates signals → you "execute" them
3. Trades are logged with entry price, quantity, stops
4. Positions are tracked at real market prices
5. Daily P&L calculated automatically
6. Full trade journal maintained

THIS IS HOW QUANT FIRMS TEST:
- No strategy goes live without 3+ months of paper trading
- Must beat benchmark (NIFTY) in paper mode first
- Track Sharpe ratio, max drawdown, win rate in real-time
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from src.data.fetcher import DataManager

PAPER_TRADE_FILE = "data/paper_trades.json"
PORTFOLIO_FILE = "data/paper_portfolio.json"


@dataclass
class PaperTrade:
    """A simulated trade."""
    trade_id: str
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    quantity: int
    entry_date: str
    stop_loss: float
    target: float
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED_OUT


class PaperTrader:
    """
    Paper trading engine — simulate real trading.
    
    USAGE:
        trader = PaperTrader(capital=1000000)
        trader.buy("RELIANCE", quantity=50, stop_loss=1400, target=1550)
        trader.update_positions()  # Updates P&L with current prices
        trader.show_portfolio()
        trader.show_trade_journal()
    """

    def __init__(self, capital: float = 1000000):
        self.dm = DataManager()
        self.initial_capital = capital
        self.portfolio = self._load_portfolio()
        self.trades = self._load_trades()

        # Initialize if first time
        if 'cash' not in self.portfolio:
            self.portfolio = {
                'cash': capital,
                'initial_capital': capital,
                'positions': {},
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
            }
            self._save_portfolio()

    def _load_portfolio(self) -> Dict:
        """Load portfolio from disk."""
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _save_portfolio(self):
        """Save portfolio to disk."""
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

    def _load_trades(self) -> List[Dict]:
        """Load trade journal from disk."""
        if os.path.exists(PAPER_TRADE_FILE):
            with open(PAPER_TRADE_FILE, 'r') as f:
                return json.load(f)
        return []

    def _save_trades(self):
        """Save trade journal to disk."""
        os.makedirs(os.path.dirname(PAPER_TRADE_FILE), exist_ok=True)
        with open(PAPER_TRADE_FILE, 'w') as f:
            json.dump(self.trades, f, indent=2)

    def _get_current_price(self, symbol: str) -> float:
        """Get current/latest price for a symbol."""
        df = self.dm.get_stock_data(symbol, period="5d")
        if df.empty:
            return 0
        return df['close'].iloc[-1]

    def buy(self, symbol: str, quantity: int = None, stop_loss: float = None,
            target: float = None, reason: str = "Signal") -> Optional[PaperTrade]:
        """Execute a paper BUY trade."""
        price = self._get_current_price(symbol)
        if price == 0:
            print(f"  ❌ Could not get price for {symbol}")
            return None

        # Auto-calculate quantity if not provided (use 10% of capital)
        if quantity is None:
            max_value = self.portfolio['cash'] * 0.10
            quantity = int(max_value / price)

        trade_value = price * quantity

        # Check if we have enough cash
        if trade_value > self.portfolio['cash']:
            print(f"  ❌ Insufficient cash. Need ₹{trade_value:,.0f}, have ₹{self.portfolio['cash']:,.0f}")
            return None

        # Auto stop-loss (2x ATR) if not provided
        if stop_loss is None:
            df = self.dm.get_stock_data(symbol, period="3mo")
            if not df.empty:
                from src.strategy.risk_manager import DynamicRiskManager
                rm = DynamicRiskManager()
                atr = rm.calculate_atr(df)
                stop_loss = round(price - 2 * atr, 2)
            else:
                stop_loss = round(price * 0.95, 2)  # Default 5%

        # Auto target (3x risk) if not provided
        if target is None:
            risk = price - stop_loss
            target = round(price + 3 * risk, 2)

        # Create trade
        trade_id = f"PT-{len(self.trades)+1:04d}"
        trade = PaperTrade(
            trade_id=trade_id,
            symbol=symbol,
            action="BUY",
            entry_price=round(price, 2),
            quantity=quantity,
            entry_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            stop_loss=stop_loss,
            target=target,
        )

        # Update portfolio
        self.portfolio['cash'] -= trade_value
        self.portfolio['positions'][symbol] = {
            'trade_id': trade_id,
            'quantity': quantity,
            'entry_price': round(price, 2),
            'current_price': round(price, 2),
            'stop_loss': stop_loss,
            'target': target,
            'unrealized_pnl': 0,
        }
        self.portfolio['total_trades'] += 1

        # Save
        self.trades.append(asdict(trade))
        self._save_trades()
        self._save_portfolio()

        print(f"  ✅ BOUGHT {quantity} × {symbol} @ ₹{price:,.2f}")
        print(f"     Value: ₹{trade_value:,.0f} | Stop: ₹{stop_loss:,.2f} | Target: ₹{target:,.2f}")
        print(f"     Trade ID: {trade_id}")

        return trade

    def sell(self, symbol: str, reason: str = "Manual"):
        """Close a paper position."""
        if symbol not in self.portfolio.get('positions', {}):
            print(f"  ❌ No open position in {symbol}")
            return

        pos = self.portfolio['positions'][symbol]
        current_price = self._get_current_price(symbol)
        if current_price == 0:
            current_price = pos['current_price']

        # Calculate P&L
        pnl = (current_price - pos['entry_price']) * pos['quantity']
        pnl_pct = (current_price / pos['entry_price'] - 1) * 100

        # Update trade record
        for trade in self.trades:
            if trade['trade_id'] == pos['trade_id'] and trade['status'] == 'OPEN':
                trade['exit_price'] = round(current_price, 2)
                trade['exit_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                trade['exit_reason'] = reason
                trade['pnl'] = round(pnl, 2)
                trade['pnl_pct'] = round(pnl_pct, 2)
                trade['status'] = "CLOSED"
                break

        # Update portfolio
        self.portfolio['cash'] += current_price * pos['quantity']
        if pnl > 0:
            self.portfolio['winning_trades'] += 1
        else:
            self.portfolio['losing_trades'] += 1
        del self.portfolio['positions'][symbol]

        self._save_trades()
        self._save_portfolio()

        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"  {emoji} SOLD {pos['quantity']} × {symbol} @ ₹{current_price:,.2f}")
        print(f"     P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%) | Reason: {reason}")

    def update_positions(self):
        """Update all open positions with current prices. Check stops/targets."""
        positions = self.portfolio.get('positions', {})
        if not positions:
            return

        stopped = []
        targeted = []

        for symbol, pos in positions.items():
            price = self._get_current_price(symbol)
            if price == 0:
                continue

            pos['current_price'] = round(price, 2)
            pos['unrealized_pnl'] = round((price - pos['entry_price']) * pos['quantity'], 2)

            # Check stop-loss
            if price <= pos['stop_loss']:
                stopped.append(symbol)
            # Check target
            elif price >= pos['target']:
                targeted.append(symbol)

        self._save_portfolio()

        # Execute stops and targets
        for sym in stopped:
            print(f"  🛑 STOP-LOSS HIT: {sym}")
            self.sell(sym, reason="STOP_LOSS")

        for sym in targeted:
            print(f"  🎯 TARGET HIT: {sym}")
            self.sell(sym, reason="TARGET_HIT")

    def show_portfolio(self):
        """Show current portfolio status."""
        self.update_positions()

        positions = self.portfolio.get('positions', {})
        cash = self.portfolio.get('cash', 0)
        initial = self.portfolio.get('initial_capital', self.initial_capital)

        # Calculate total value
        total_invested = sum(
            pos['current_price'] * pos['quantity'] for pos in positions.values()
        )
        total_value = cash + total_invested
        total_pnl = total_value - initial
        total_pnl_pct = (total_value / initial - 1) * 100

        print("\n" + "=" * 70)
        print("  📋 PAPER TRADING PORTFOLIO")
        print("=" * 70)
        print(f"  Start Date: {self.portfolio.get('start_date', 'N/A')}")
        print(f"  Initial Capital: ₹{initial:,.0f}")

        print(f"\n  💰 Portfolio Value: ₹{total_value:,.0f}")
        emoji = "🟢" if total_pnl >= 0 else "🔴"
        print(f"  {emoji} Total P&L: ₹{total_pnl:,.0f} ({total_pnl_pct:+.2f}%)")
        print(f"  💵 Cash Available: ₹{cash:,.0f}")
        print(f"  📊 Invested: ₹{total_invested:,.0f}")

        # Win/Loss stats
        total_trades = self.portfolio.get('total_trades', 0)
        wins = self.portfolio.get('winning_trades', 0)
        losses = self.portfolio.get('losing_trades', 0)
        closed = wins + losses
        win_rate = wins / closed * 100 if closed > 0 else 0

        print(f"\n  📈 Stats: {total_trades} trades | {wins}W / {losses}L | "
              f"Win Rate: {win_rate:.0f}%")

        # Open positions
        if positions:
            print(f"\n  {'Symbol':<12} {'Qty':>5} {'Entry':>8} {'Current':>8} "
                  f"{'P&L':>10} {'P&L%':>7} {'Stop':>8} {'Target':>8}")
            print("  " + "-" * 75)

            for symbol, pos in positions.items():
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = (pos['current_price'] / pos['entry_price'] - 1) * 100
                emoji = "🟢" if pnl >= 0 else "🔴"

                print(
                    f"  {symbol:<12} {pos['quantity']:>5} "
                    f"₹{pos['entry_price']:>7,.0f} ₹{pos['current_price']:>7,.0f} "
                    f"{emoji}₹{pnl:>8,.0f} {pnl_pct:>+6.1f}% "
                    f"₹{pos['stop_loss']:>7,.0f} ₹{pos['target']:>7,.0f}"
                )
        else:
            print("\n  No open positions.")

        print("=" * 70)

    def show_trade_journal(self):
        """Show full trade history."""
        if not self.trades:
            print("\n  No trades yet. Use 'python main.py paper buy SYMBOL' to start.")
            return

        print("\n" + "=" * 70)
        print("  📓 TRADE JOURNAL")
        print("=" * 70)

        print(f"  {'ID':<8} {'Symbol':<10} {'Entry':>7} {'Exit':>7} "
              f"{'P&L%':>7} {'Status':<12} {'Reason':<15}")
        print("  " + "-" * 70)

        for t in self.trades[-20:]:  # Last 20 trades
            exit_price = t.get('exit_price', '-')
            pnl_str = f"{t['pnl_pct']:+.1f}%" if t.get('pnl_pct') is not None else "  -"
            status_emoji = {"OPEN": "🔵", "CLOSED": "✅", "STOPPED_OUT": "🛑"}.get(t['status'], "⚪")

            print(
                f"  {t['trade_id']:<8} {t['symbol']:<10} "
                f"₹{t['entry_price']:>6,.0f} "
                f"{'₹' + str(int(exit_price)) if exit_price != '-' else '   -':>7} "
                f"{pnl_str:>7} {status_emoji}{t['status']:<11} "
                f"{t.get('exit_reason', 'OPEN'):<15}"
            )

        print("=" * 70)

    def reset(self):
        """Reset paper trading (start fresh)."""
        self.portfolio = {
            'cash': self.initial_capital,
            'initial_capital': self.initial_capital,
            'positions': {},
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
        }
        self.trades = []
        self._save_portfolio()
        self._save_trades()
        print("  🔄 Paper trading reset. Fresh start with ₹{:,.0f}".format(self.initial_capital))
