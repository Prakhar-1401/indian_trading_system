"""
Morning Pre-Market Report — Everything you need before 9:15 AM.

WHAT THIS DOES:
===============
Generates a complete pre-market briefing combining ALL modules:
1. Global market overnight moves (US, Asia, Europe)
2. Market regime + volatility state
3. Geopolitical risk level
4. Top signals from ML + traditional indicators
5. Pairs trading opportunities
6. Event calendar (earnings, dividends today/this week)
7. Optimal position sizes (Kelly)
8. Risk parameters (ATR stops)

RUN THIS: Every morning at 8:30 AM before market opens.
Can be scheduled via scheduler.py or Windows Task Scheduler.

SENDS TO: Console + Telegram (if configured)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger
import yfinance as yf

from src.data.fetcher import DataManager


class PreMarketReport:
    """
    Generate comprehensive morning pre-market report.
    
    USAGE:
        report = PreMarketReport()
        report.generate()  # Prints + sends to Telegram
    """

    def __init__(self):
        self.dm = DataManager()

    def _get_global_markets(self) -> Dict:
        """Get overnight performance of global markets."""
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Nikkei 225": "^N225",
            "Hang Seng": "^HSI",
            "FTSE 100": "^FTSE",
            "SGX Nifty": "^STI",  # Proxy
            "Gold": "GC=F",
            "Crude Oil": "CL=F",
            "USD/INR": "INR=X",
        }

        results = {}
        for name, ticker in indices.items():
            try:
                data = yf.download(ticker, period="5d", interval="1d", progress=False)
                if data is not None and len(data) >= 2:
                    # Handle multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        close_col = ('Close', ticker)
                        if close_col in data.columns:
                            prev = data[close_col].iloc[-2]
                            last = data[close_col].iloc[-1]
                        else:
                            close_data = data['Close']
                            if isinstance(close_data, pd.DataFrame):
                                close_data = close_data.iloc[:, 0]
                            prev = close_data.iloc[-2]
                            last = close_data.iloc[-1]
                    else:
                        prev = data['Close'].iloc[-2]
                        last = data['Close'].iloc[-1]
                    
                    change_pct = (last / prev - 1) * 100
                    results[name] = {
                        'price': round(float(last), 2),
                        'change_pct': round(float(change_pct), 2),
                    }
            except Exception as e:
                logger.debug(f"Failed to get {name}: {e}")

        return results

    def _get_india_futures(self) -> Dict:
        """Get NIFTY/BANKNIFTY pre-market indicators."""
        result = {}
        try:
            nifty = self.dm.get_stock_data("^NSEI", period="5d")
            if not nifty.empty:
                result['nifty_prev_close'] = round(nifty['close'].iloc[-1], 2)
                result['nifty_5d_return'] = round(
                    (nifty['close'].iloc[-1] / nifty['close'].iloc[-5] - 1) * 100, 2
                )
        except Exception:
            pass
        return result

    def _get_fii_dii_proxy(self) -> Dict:
        """
        Estimate FII/DII flow from market breadth.
        (Real FII data needs NSE website scraping or paid API)
        """
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                   "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE"]
        
        up_vol = 0
        down_vol = 0
        
        for sym in symbols:
            try:
                df = self.dm.get_stock_data(sym, period="5d")
                if df.empty or len(df) < 2:
                    continue
                last_return = df['close'].iloc[-1] / df['close'].iloc[-2] - 1
                last_vol = df['volume'].iloc[-1]
                if last_return > 0:
                    up_vol += last_vol
                else:
                    down_vol += last_vol
            except Exception:
                continue

        total = up_vol + down_vol
        if total > 0:
            buy_ratio = up_vol / total
            if buy_ratio > 0.6:
                flow = "NET BUYING"
            elif buy_ratio < 0.4:
                flow = "NET SELLING"
            else:
                flow = "MIXED"
        else:
            flow = "N/A"
            buy_ratio = 0.5

        return {'flow_direction': flow, 'buy_ratio': round(buy_ratio * 100, 1)}

    def generate(self, send_telegram: bool = True):
        """Generate and print the full pre-market report."""
        now = datetime.now()
        
        print("\n" + "═" * 70)
        print(f"  ☀️  MORNING PRE-MARKET REPORT — {now.strftime('%A, %B %d, %Y')}")
        print("═" * 70)

        # 1. Global Markets
        print("\n  🌍 GLOBAL MARKETS (Overnight)")
        print("  " + "-" * 50)
        global_mkts = self._get_global_markets()
        
        for name, data in global_mkts.items():
            emoji = "🟢" if data['change_pct'] > 0 else "🔴" if data['change_pct'] < 0 else "⚪"
            print(f"    {emoji} {name:<14} {data['price']:>12,.2f}  ({data['change_pct']:+.2f}%)")

        # 2. India Pre-Market
        print("\n  🇮🇳 INDIA INDICATORS")
        print("  " + "-" * 50)
        india = self._get_india_futures()
        if india:
            prev_close = india.get('nifty_prev_close')
            ret_5d = india.get('nifty_5d_return')
            if prev_close is not None:
                print(f"    NIFTY 50 Prev Close: ₹{prev_close:,.2f}")
            if ret_5d is not None:
                print(f"    NIFTY 5-day Return: {ret_5d:+.2f}%")

        # FII/DII Flow
        flow = self._get_fii_dii_proxy()
        flow_emoji = "🟢" if flow['flow_direction'] == "NET BUYING" else "🔴" if flow['flow_direction'] == "NET SELLING" else "🟡"
        print(f"    {flow_emoji} Institutional Flow: {flow['flow_direction']} (Buy ratio: {flow['buy_ratio']}%)")

        # 3. Market Regime
        print("\n  📊 MARKET REGIME")
        print("  " + "-" * 50)
        try:
            from src.strategy.regime_detector import MarketRegimeDetector
            detector = MarketRegimeDetector()
            regime = detector.detect_regime()
            regime_emoji = {"BULL": "🐂", "BEAR": "🐻", "SIDEWAYS": "↔️", "VOLATILE": "⚡"}.get(regime.regime, "?")
            print(f"    {regime_emoji} Regime: {regime.regime} (confidence: {regime.confidence:.0%})")
            print(f"    💡 {regime.recommendation}")
        except Exception as e:
            print(f"    ⚠️ Could not detect regime: {e}")

        # 4. Geopolitical Risk
        print("\n  🌐 GEOPOLITICAL RISK")
        print("  " + "-" * 50)
        try:
            from src.sentiment.geopolitical import GeopoliticalMonitor
            monitor = GeopoliticalMonitor()
            report = monitor.get_risk_report()
            risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(report.get('risk_level', ''), "⚪")
            print(f"    {risk_emoji} Risk Level: {report.get('risk_level', 'N/A')} ({report.get('risk_score', 0)}/10)")
            if report.get('top_events'):
                print(f"    Top Event: {report['top_events'][0].get('headline', '')[:60]}")
        except Exception as e:
            print(f"    ⚠️ Could not assess: {e}")

        # 5. Top Signals
        print("\n  🎯 TOP SIGNALS TODAY")
        print("  " + "-" * 50)
        try:
            from src.strategy.executor import generate_quick_signals
            signals = generate_quick_signals()
            if signals:
                for s in signals[:5]:
                    action = getattr(s, 'action', 'HOLD')
                    symbol = getattr(s, 'symbol', 'N/A')
                    score = getattr(s, 'score', 0)
                    emoji = "🟢" if action == 'BUY' else "🔴" if action == 'SELL' else "⚪"
                    print(f"    {emoji} {symbol:<12} {action:<5} Score: {score:.1f}")
            else:
                print("    No strong signals today.")
        except Exception as e:
            print(f"    ⚠️ Could not generate: {e}")

        # 6. Events This Week
        print("\n  📅 EVENTS THIS WEEK")
        print("  " + "-" * 50)
        try:
            from src.strategy.event_driven import EventDrivenTrader
            trader = EventDrivenTrader()
            events = trader.scan_events()
            upcoming = [e for e in events if e.event_date and 
                       0 <= (e.event_date - pd.Timestamp.now()).days <= 7]
            if upcoming:
                for e in upcoming[:5]:
                    days = (e.event_date - pd.Timestamp.now()).days
                    print(f"    📌 {e.symbol:<12} {e.event_type:<12} in {days}d — {e.details[:35]}")
            else:
                print("    No major events in next 7 days.")
        except Exception as e:
            print(f"    ⚠️ Could not scan events: {e}")

        # 7. Pairs Trading
        print("\n  ⚖️ PAIRS TRADING")
        print("  " + "-" * 50)
        try:
            from src.strategy.pairs_trading import PairsTradingEngine
            engine = PairsTradingEngine()
            signals = engine.generate_signals()
            active = [s for s in signals if s.signal_type != 'HOLD']
            if active:
                for s in active[:3]:
                    print(f"    ⚡ {s.stock_long}/{s.stock_short}: {s.signal_type} (z-score: {s.zscore:+.2f})")
            else:
                print("    No active pairs signals. All spreads in normal range.")
        except Exception as e:
            print(f"    ⚠️ {e}")

        # 8. Summary & Action Items
        print("\n  ✅ ACTION ITEMS FOR TODAY")
        print("  " + "-" * 50)
        
        # Compile actions based on all data
        actions = []
        if global_mkts:
            us_change = global_mkts.get("S&P 500", {}).get('change_pct', 0)
            if us_change < -1:
                actions.append("⚠️ US markets down — expect gap-down opening")
            elif us_change > 1:
                actions.append("📈 US markets up — expect positive opening")

        if flow['flow_direction'] == "NET SELLING":
            actions.append("🔴 Institutional selling — be cautious with new longs")
        
        try:
            if regime.regime == "VOLATILE":
                actions.append("⚡ VOLATILE regime — reduce position sizes 70%")
            elif regime.regime == "BEAR":
                actions.append("🐻 BEAR regime — defensive stocks only")
        except Exception:
            pass

        if not actions:
            actions.append("✅ No special actions — follow normal strategy")

        for action in actions:
            print(f"    {action}")

        print("\n" + "═" * 70)
        print(f"  Report generated at {now.strftime('%H:%M:%S')}")
        print("═" * 70)

        # Send to Telegram if configured
        if send_telegram:
            self._send_telegram_summary(global_mkts, flow, actions)

    def _send_telegram_summary(self, global_mkts: Dict, flow: Dict, actions: List):
        """Send condensed version to Telegram."""
        try:
            from src.utils.telegram_alerts import TelegramAlerts
            bot = TelegramAlerts()
            if not bot.enabled:
                return

            us_sp500 = global_mkts.get("S&P 500", {})
            gold = global_mkts.get("Gold", {})
            oil = global_mkts.get("Crude Oil", {})

            text = f"""
☀️ <b>PRE-MARKET REPORT — {datetime.now().strftime('%b %d')}</b>

🌍 S&P 500: {us_sp500.get('change_pct', 0):+.2f}%
🪙 Gold: {gold.get('change_pct', 0):+.2f}%
🛢️ Oil: {oil.get('change_pct', 0):+.2f}%
🏦 Inst. Flow: {flow['flow_direction']}

📋 Actions:
{chr(10).join(actions[:3])}
"""
            bot.send_message(text)
        except Exception as e:
            logger.debug(f"Telegram summary failed: {e}")
