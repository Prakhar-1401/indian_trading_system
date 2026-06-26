"""
Daily Morning Automation Script
================================
Runs automatically via Windows Task Scheduler at 8:45 AM IST (before market opens at 9:15).

What it does:
1. Pre-market report (global cues, regime, events, signals)
2. Sends summary to Telegram
3. Logs everything

Schedule: Mon-Fri at 8:45 AM
"""

import sys
import os
from datetime import datetime, date

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from src.data.fetcher import DataManager
from src.utils.telegram_alerts import TelegramAlerts

# Configure logging
logger.add("logs/daily_morning_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")


def run_morning_routine():
    """Complete morning routine before market opens."""
    logger.info("=" * 60)
    logger.info(f"DAILY MORNING ROUTINE — {datetime.now().strftime('%A, %B %d, %Y %H:%M')}")
    logger.info("=" * 60)

    telegram = TelegramAlerts()
    dm = DataManager()
    messages = []
    messages.append(f"☀️ *MORNING REPORT — {date.today().strftime('%b %d, %Y')}*\n")

    # 1. MARKET REGIME
    try:
        from src.strategy.regime_detector import MarketRegimeDetector
        detector = MarketRegimeDetector()
        result = detector.detect_regime()
        regime = result.regime
        confidence = result.confidence
        trend = result.trend_score
        vol_pct = result.volatility_percentile

        regime_emoji = {'BULL': '🟢', 'BEAR': '🔴', 'SIDEWAYS': '🟡', 'VOLATILE': '⚡'}.get(regime, '⚪')
        messages.append(f"{regime_emoji} *Regime*: {regime} ({confidence}% conf)")
        messages.append(f"   Trend: {trend:+.2f} | Vol: {vol_pct}th pctl")

        if regime == 'VOLATILE':
            messages.append("   ⚠️ REDUCE POSITIONS 70%")
        elif regime == 'BEAR':
            messages.append("   ⚠️ REDUCE EXPOSURE 50%")

        logger.info(f"Regime: {regime} (confidence={confidence}%)")
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        messages.append(f"⚠️ Regime: Error - {e}")

    # 2. TOP SIGNALS
    try:
        from src.strategy.executor import generate_quick_signals
        signals = generate_quick_signals()
        buy_signals = [s for s in signals if s.action == 'BUY']

        if buy_signals:
            messages.append(f"\n🎯 *BUY SIGNALS* ({len(buy_signals)}):")
            for s in buy_signals[:5]:
                messages.append(f"   • {s.symbol}: Score {s.score:.1f}")
        else:
            messages.append("\n🎯 No BUY signals today")

        logger.info(f"Signals: {len(buy_signals)} BUY out of {len(signals)} total")
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")

    # 3. KELLY — BEST EDGE STOCKS
    try:
        from src.strategy.kelly_sizing import KellyPositionSizer
        kelly = KellyPositionSizer()
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
                   "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC"]
        results = kelly.calculate_portfolio(symbols)
        positive_edge = [r for r in results if r.edge > 0]

        if positive_edge:
            messages.append(f"\n💰 *POSITIVE EDGE* ({len(positive_edge)}):")
            for r in sorted(positive_edge, key=lambda x: x.edge, reverse=True)[:3]:
                messages.append(f"   • {r.symbol}: Edge {r.edge:+.2f}%, Kelly {r.kelly_half:.1f}%")
        else:
            messages.append("\n💰 No stocks with positive edge today")

        logger.info(f"Kelly: {len(positive_edge)} stocks with positive edge")
    except Exception as e:
        logger.error(f"Kelly analysis failed: {e}")

    # 4. EVENTS WARNING
    try:
        from src.strategy.event_driven import EventDrivenTrader
        trader = EventDrivenTrader()
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN",
                   "BHARTIARTL", "ITC", "LT", "BAJFINANCE", "SUNPHARMA"]
        all_events = []
        for sym in symbols:
            try:
                events = trader.scan_events(sym)
                all_events.extend(events)
            except Exception:
                pass

        # Filter only upcoming events (next 5 days)
        upcoming = [e for e in all_events if hasattr(e, 'days_until') and 0 <= e.days_until <= 5]
        if upcoming:
            messages.append(f"\n📅 *EVENTS (next 5 days)*:")
            for ev in upcoming[:5]:
                messages.append(f"   • {ev.symbol}: {ev.event_type} in {ev.days_until}d")
    except Exception as e:
        logger.error(f"Events scan failed: {e}")

    # 5. GEO RISK
    try:
        from src.sentiment.geopolitical import GeopoliticalMonitor
        geo = GeopoliticalMonitor()
        risk = geo.get_risk_report()
        risk_level = risk.get('risk_level', 'UNKNOWN')
        risk_score = risk.get('risk_score', 0)

        if risk_level in ['HIGH', 'CRITICAL']:
            messages.append(f"\n🌐 *GEO RISK: {risk_level}* ({risk_score}/10)")
            messages.append("   ⚠️ Consider reducing exposure!")
    except Exception as e:
        logger.error(f"Geo risk failed: {e}")

    # 6. PAPER PORTFOLIO UPDATE
    try:
        import json
        portfolio_file = os.path.join("data", "paper_portfolio.json")
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)
            cash = portfolio.get('cash', 0)
            positions = portfolio.get('positions', {})
            initial = portfolio.get('initial_capital', 1000000)
            total_invested = sum(p.get('quantity', 0) * p.get('entry_price', 0) for p in positions.values())
            total_value = cash + total_invested
            pnl_pct = ((total_value - initial) / initial) * 100

            messages.append(f"\n📋 *PAPER PORTFOLIO*:")
            messages.append(f"   Value: ₹{total_value:,.0f} ({pnl_pct:+.2f}%)")
            messages.append(f"   Positions: {len(positions)} | Cash: ₹{cash:,.0f}")
    except Exception as e:
        logger.error(f"Portfolio read failed: {e}")

    # SEND TO TELEGRAM
    full_message = "\n".join(messages)
    logger.info(f"Sending Telegram message ({len(full_message)} chars)")

    try:
        telegram.send_message(full_message)
        logger.info("✅ Telegram sent successfully")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")

    # Also print to console
    print(full_message)
    print("\n" + "=" * 60)
    print("Morning routine complete!")


def run_evening_routine():
    """Evening routine after market closes (3:30 PM+)."""
    logger.info("EVENING ROUTINE starting...")
    telegram = TelegramAlerts()
    messages = [f"🌙 *EVENING SUMMARY — {date.today().strftime('%b %d')}*\n"]

    # Check paper portfolio P&L with current prices
    try:
        import json
        portfolio_file = os.path.join("data", "paper_portfolio.json")
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)

            dm = DataManager()
            positions = portfolio.get('positions', {})
            if positions:
                messages.append("📊 *Position Updates*:")
                for sym, pos in positions.items():
                    try:
                        df = dm.get_stock_data(sym, period="1d")
                        if not df.empty:
                            current = df['close'].iloc[-1]
                            entry = pos['entry_price']
                            pnl_pct = ((current - entry) / entry) * 100
                            emoji = "🟢" if pnl_pct > 0 else "🔴"
                            messages.append(f"   {emoji} {sym}: ₹{current:.0f} ({pnl_pct:+.1f}%)")

                            # Check stop-loss hit
                            stop = pos.get('stop_loss', 0)
                            if stop and current <= stop:
                                messages.append(f"   🚨 {sym} HIT STOP-LOSS! Consider selling.")
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"Evening portfolio check failed: {e}")

    full_message = "\n".join(messages)
    try:
        telegram.send_message(full_message)
        logger.info("✅ Evening Telegram sent")
    except Exception as e:
        logger.error(f"Telegram failed: {e}")

    print(full_message)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily Trading Automation")
    parser.add_argument("routine", choices=["morning", "evening"], default="morning", nargs="?",
                       help="Which routine to run (morning or evening)")
    args = parser.parse_args()

    if args.routine == "morning":
        run_morning_routine()
    elif args.routine == "evening":
        run_evening_routine()
