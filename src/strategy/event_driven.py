"""
Event-Driven Trading — Trade around corporate events automatically.

WHAT IS EVENT-DRIVEN TRADING?
==============================
Stocks move predictably around certain events:
1. EARNINGS: Stock gaps ±5-15% after results. Patterns exist:
   - Stocks that beat expectations 3 quarters in a row tend to beat again
   - Pre-earnings drift: stocks drift toward the earnings direction before release
   
2. DIVIDENDS: Ex-dividend date = stock drops by dividend amount
   - But high-dividend stocks often recover within 2-3 days
   - Dividend capture strategy: buy before ex-date, sell after recovery

3. SPLITS/BONUS: Stocks rally 10-20% in the weeks before a split
   - Post-split: often flat or slight decline (sell the news)

4. INDEX ADDITION/REMOVAL:
   - Added to NIFTY 50 → forced buying by index funds → rallies 5-10%
   - Removed → forced selling → drops 5-10%

HOW QUANT FIRMS TRADE EVENTS:
==============================
- Pre-position before the event (if there's a pattern)
- Use options for earnings (straddles for expected big moves)
- Fade the gap if it's overdone (mean reversion after gap)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import yfinance as yf

from src.data.fetcher import DataManager


@dataclass
class CorporateEvent:
    """A detected corporate event."""
    symbol: str
    event_type: str  # EARNINGS, DIVIDEND, SPLIT, AGM
    event_date: Optional[datetime]
    details: str
    trading_strategy: str
    confidence: str  # HIGH, MEDIUM, LOW


@dataclass 
class EventSignal:
    """Trading signal generated from an event."""
    symbol: str
    action: str  # BUY, SELL, STRADDLE, HOLD
    reason: str
    days_to_event: int
    expected_move: float  # Expected % move
    risk_reward: str


class EventDrivenTrader:
    """
    Detect and trade around corporate events.
    
    USAGE:
        trader = EventDrivenTrader()
        events = trader.scan_events()
        signals = trader.generate_signals(events)
        trader.print_report(events, signals)
    """

    def __init__(self):
        self.dm = DataManager()
        self.watchlist = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
            "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "HCLTECH",
            "TATAMOTORS", "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND",
        ]

    def _get_earnings_info(self, symbol: str) -> Optional[CorporateEvent]:
        """Check for upcoming earnings using yfinance calendar."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            calendar = ticker.calendar

            if calendar is None or calendar.empty if hasattr(calendar, 'empty') else not calendar:
                return None

            # Try to get earnings date
            earnings_date = None
            if isinstance(calendar, pd.DataFrame):
                if 'Earnings Date' in calendar.columns:
                    earnings_date = calendar['Earnings Date'].iloc[0]
                elif 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
            elif isinstance(calendar, dict):
                earnings_date = calendar.get('Earnings Date', [None])[0]

            if earnings_date is None:
                return None

            if isinstance(earnings_date, str):
                earnings_date = pd.to_datetime(earnings_date)
            elif not isinstance(earnings_date, pd.Timestamp):
                earnings_date = pd.Timestamp(earnings_date)

            days_to = (earnings_date - pd.Timestamp.now()).days

            if days_to < -7 or days_to > 60:
                return None

            return CorporateEvent(
                symbol=symbol,
                event_type="EARNINGS",
                event_date=earnings_date,
                details=f"Results in {days_to} days",
                trading_strategy=self._earnings_strategy(symbol, days_to),
                confidence="MEDIUM",
            )
        except Exception as e:
            logger.debug(f"No earnings info for {symbol}: {e}")
            return None

    def _get_dividend_info(self, symbol: str) -> Optional[CorporateEvent]:
        """Check for upcoming/recent dividends."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            divs = ticker.dividends

            if divs.empty:
                return None

            # Get most recent dividend
            last_div = divs.iloc[-1]
            last_div_date = divs.index[-1]

            if hasattr(last_div_date, 'tz') and last_div_date.tz is not None:
                last_div_date = last_div_date.tz_localize(None)

            days_since = (pd.Timestamp.now() - last_div_date).days

            # Only report if recent (within 30 days) or if pattern suggests upcoming
            if days_since > 90:
                # Check if there's a yearly pattern
                if len(divs) >= 4:
                    # Estimate next dividend date
                    intervals = divs.index.to_series().diff().dropna()
                    avg_interval = intervals.mean()
                    expected_next = last_div_date + avg_interval

                    days_to_next = (expected_next - pd.Timestamp.now()).days
                    if 0 < days_to_next < 45:
                        return CorporateEvent(
                            symbol=symbol,
                            event_type="DIVIDEND",
                            event_date=expected_next,
                            details=f"Expected ₹{last_div:.1f} dividend in ~{days_to_next} days",
                            trading_strategy="Buy 5 days before ex-date. Hold through record date.",
                            confidence="LOW",
                        )
                return None

            # Recent dividend
            info = ticker.info
            div_yield = info.get('dividendYield', 0) or 0

            return CorporateEvent(
                symbol=symbol,
                event_type="DIVIDEND",
                event_date=last_div_date,
                details=f"₹{last_div:.1f}/share ({div_yield*100:.1f}% yield). {days_since}d ago.",
                trading_strategy="If ex-date upcoming: buy before, sell after recovery (2-3 days).",
                confidence="MEDIUM" if div_yield > 0.03 else "LOW",
            )
        except Exception as e:
            logger.debug(f"No dividend info for {symbol}: {e}")
            return None

    def _detect_pre_earnings_drift(self, symbol: str) -> Optional[Dict]:
        """
        Detect pre-earnings drift pattern.
        
        If a stock has beaten estimates 3+ times in a row,
        it tends to drift UP before the next earnings.
        """
        df = self.dm.get_stock_data(symbol, period="1y")
        if df.empty or len(df) < 60:
            return None

        # Look for earnings gaps (>3% gap up/down on high volume)
        returns = df['close'].pct_change()
        vol_avg = df['volume'].rolling(20).mean()
        vol_ratio = df['volume'] / vol_avg

        # Find earnings days (big move + high volume)
        earnings_days = df[(abs(returns) > 0.03) & (vol_ratio > 2.0)]

        if len(earnings_days) < 2:
            return None

        # Check if recent earnings were positive
        recent_gaps = returns[earnings_days.index].tail(4)
        positive_beats = (recent_gaps > 0).sum()

        return {
            'recent_beats': positive_beats,
            'total_events': len(recent_gaps),
            'avg_gap': recent_gaps.mean() * 100,
            'pattern': 'POSITIVE_STREAK' if positive_beats >= 3 else 'MIXED',
        }

    def _earnings_strategy(self, symbol: str, days_to: int) -> str:
        """Determine earnings trading strategy."""
        drift = self._detect_pre_earnings_drift(symbol)

        if days_to > 15:
            if drift and drift['pattern'] == 'POSITIVE_STREAK':
                return f"PRE-EARNINGS DRIFT: {drift['recent_beats']}/{drift['total_events']} recent beats. Buy now for drift."
            return "Too far out. Monitor for pre-earnings positioning."
        elif days_to > 5:
            return "STRADDLE: Buy ATM call + put for expected big move."
        elif days_to > 0:
            return "HIGH RISK: Don't enter new. If holding, tighten stops or hedge with puts."
        else:
            return "POST-EARNINGS: Evaluate gap. Fade if overdone (>8% gap)."

    def _detect_volume_events(self, symbol: str) -> Optional[CorporateEvent]:
        """Detect unusual volume that might indicate upcoming event."""
        df = self.dm.get_stock_data(symbol, period="3mo")
        if df.empty or len(df) < 30:
            return None

        vol_avg = df['volume'].rolling(20).mean()
        recent_vol = df['volume'].tail(3).mean()
        vol_ratio = recent_vol / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1

        if vol_ratio > 3.0:
            recent_return = (df['close'].iloc[-1] / df['close'].iloc[-3] - 1) * 100
            direction = "accumulation (bullish)" if recent_return > 0 else "distribution (bearish)"

            return CorporateEvent(
                symbol=symbol,
                event_type="VOLUME_SPIKE",
                event_date=df.index[-1],
                details=f"Volume {vol_ratio:.1f}x normal. Price {recent_return:+.1f}%. Likely {direction}.",
                trading_strategy=f"{'Follow the move (buy)' if recent_return > 0 else 'Wait for confirmation or short'}",
                confidence="MEDIUM",
            )
        return None

    def scan_events(self) -> List[CorporateEvent]:
        """Scan all watchlist stocks for upcoming events."""
        events = []

        for symbol in self.watchlist:
            logger.debug(f"Scanning events for {symbol}...")

            # Check earnings
            e = self._get_earnings_info(symbol)
            if e:
                events.append(e)

            # Check dividends
            d = self._get_dividend_info(symbol)
            if d:
                events.append(d)

            # Check volume spikes
            v = self._detect_volume_events(symbol)
            if v:
                events.append(v)

        # Sort by date
        events.sort(key=lambda x: x.event_date if x.event_date else datetime.max)
        return events

    def generate_signals(self, events: List[CorporateEvent]) -> List[EventSignal]:
        """Generate trading signals from detected events."""
        signals = []

        for event in events:
            if event.event_date is None:
                continue

            days_to = (event.event_date - pd.Timestamp.now()).days

            if event.event_type == "EARNINGS":
                if 5 < days_to < 20:
                    drift = self._detect_pre_earnings_drift(event.symbol)
                    if drift and drift['pattern'] == 'POSITIVE_STREAK':
                        signals.append(EventSignal(
                            symbol=event.symbol,
                            action="BUY",
                            reason=f"Pre-earnings drift: {drift['recent_beats']} consecutive beats",
                            days_to_event=days_to,
                            expected_move=drift['avg_gap'],
                            risk_reward="2:1",
                        ))

            elif event.event_type == "DIVIDEND":
                if 0 < days_to < 10:
                    signals.append(EventSignal(
                        symbol=event.symbol,
                        action="BUY",
                        reason="Dividend capture: buy before ex-date",
                        days_to_event=days_to,
                        expected_move=1.5,
                        risk_reward="1.5:1",
                    ))

            elif event.event_type == "VOLUME_SPIKE":
                if "accumulation" in event.details:
                    signals.append(EventSignal(
                        symbol=event.symbol,
                        action="BUY",
                        reason="Unusual volume accumulation detected",
                        days_to_event=0,
                        expected_move=3.0,
                        risk_reward="2:1",
                    ))

        return signals

    def print_report(self, events: List[CorporateEvent], signals: List[EventSignal]):
        """Print event-driven analysis report."""
        print("\n" + "=" * 70)
        print("  EVENT-DRIVEN TRADING SCANNER")
        print("=" * 70)

        if not events:
            print("\n  No significant events detected in watchlist.")
            print("=" * 70)
            return

        # Group by type
        by_type = {}
        for e in events:
            by_type.setdefault(e.event_type, []).append(e)

        for etype, elist in by_type.items():
            emoji = {"EARNINGS": "📊", "DIVIDEND": "💰", "SPLIT": "✂️",
                     "VOLUME_SPIKE": "📈", "AGM": "🏢"}.get(etype, "📌")
            print(f"\n  {emoji} {etype} ({len(elist)} events):")
            print(f"  {'Symbol':<12} {'Date':<12} {'Confidence':<10} Details")
            print(f"  {'-'*65}")

            for e in elist:
                date_str = e.event_date.strftime('%Y-%m-%d') if e.event_date else "Unknown"
                print(f"  {e.symbol:<12} {date_str:<12} {e.confidence:<10} {e.details[:40]}")
                print(f"  {'':12} Strategy: {e.trading_strategy[:55]}")

        # Signals
        if signals:
            print(f"\n\n  ⚡ ACTIONABLE SIGNALS:")
            print(f"  {'Symbol':<12} {'Action':<8} {'Days':>5} {'Exp Move':>9} Reason")
            print(f"  {'-'*70}")
            for s in signals:
                emoji = {"BUY": "🟢", "SELL": "🔴", "STRADDLE": "🟡"}.get(s.action, "⚪")
                print(
                    f"  {s.symbol:<12} {emoji}{s.action:<7} {s.days_to_event:>5}d "
                    f"{s.expected_move:>+7.1f}% {s.reason}"
                )
        else:
            print("\n  No actionable signals at this time.")

        print("=" * 70)
