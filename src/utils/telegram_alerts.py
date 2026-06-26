"""
Telegram Alert Bot — Get trading signals on your phone.

SETUP (One-time, takes 2 minutes):
===================================
1. Open Telegram, search for @BotFather
2. Send: /newbot
3. Name it: "My Trading Alerts" (or whatever you want)
4. BotFather gives you a TOKEN like: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
5. Save that token in your .env file as: TELEGRAM_BOT_TOKEN=your_token

6. Create a channel or group in Telegram for alerts
7. Add your bot to that channel as admin
8. Get your chat ID:
   - Send any message to your bot
   - Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   - Find "chat":{"id": XXXXX} — that's your CHAT_ID
9. Save in .env: TELEGRAM_CHAT_ID=your_chat_id

That's it! Now signals get sent to your phone automatically.
"""
import os
import requests
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class TelegramAlerts:
    """
    Send trading alerts to Telegram.
    
    USAGE:
        bot = TelegramAlerts()
        bot.send_signal("BUY", "COALINDIA", 480.0, "Momentum score 6.7/10")
        bot.send_daily_report(rankings, signals)
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

        if not self.enabled:
            logger.info("Telegram alerts disabled (no TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID in .env)")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the Telegram channel."""
        if not self.enabled:
            logger.debug(f"[Telegram disabled] Would send: {text[:50]}...")
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                logger.debug("Telegram message sent successfully")
                return True
            else:
                logger.warning(f"Telegram API error: {response.status_code} — {response.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_signal(self, action: str, symbol: str, price: float,
                    reason: str, score: float = None):
        """Send a trading signal alert."""
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")

        text = f"""
{emoji} <b>{action} SIGNAL — {symbol}</b>

💰 Price: ₹{price:,.2f}
📊 Score: {score:.2f}/10
📝 Reason: {reason}
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return self.send_message(text)

    def send_pairs_signal(self, long_stock: str, short_stock: str,
                          zscore: float, confidence: str):
        """Send pairs trading signal."""
        text = f"""
⚡ <b>PAIRS TRADE SIGNAL</b>

📈 LONG: {long_stock}
📉 SHORT: {short_stock}
📊 Z-Score: {zscore:+.2f}
🎯 Confidence: {confidence}
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

<i>Spread has diverged — expecting mean reversion</i>
"""
        return self.send_message(text)

    def send_risk_alert(self, symbol: str, alert_type: str, details: str):
        """Send risk management alert (stop-loss hit, drawdown, etc.)."""
        text = f"""
🚨 <b>RISK ALERT — {alert_type}</b>

📌 Stock: {symbol}
⚠️ {details}
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return self.send_message(text)

    def send_geopolitical_alert(self, risk_level: str, headline: str,
                                 sectors_positive: List[str], sectors_negative: List[str]):
        """Send geopolitical risk alert."""
        emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")

        pos_text = ", ".join(sectors_positive[:3]) if sectors_positive else "None"
        neg_text = ", ".join(sectors_negative[:3]) if sectors_negative else "None"

        text = f"""
{emoji} <b>GEOPOLITICAL ALERT — {risk_level} RISK</b>

📰 {headline}

📈 Sectors to BUY: {pos_text}
📉 Sectors to AVOID: {neg_text}
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return self.send_message(text)

    def send_daily_report(self, top_stocks: List[Dict], market_mood: str,
                          risk_level: str):
        """Send morning daily report with all signals."""
        now = datetime.now().strftime('%Y-%m-%d')

        stocks_text = ""
        for i, s in enumerate(top_stocks[:5], 1):
            emoji = "🟢" if s.get("action") == "BUY" else "🟡"
            stocks_text += f"  {i}. {emoji} {s['symbol']} — Score: {s.get('score', 0):.2f}\n"

        text = f"""
📊 <b>DAILY MARKET REPORT — {now}</b>

🌡️ Market Mood: {market_mood}
🌍 Geo Risk: {risk_level}

<b>Top 5 Stocks:</b>
{stocks_text}
<i>Run full analysis for detailed signals</i>
"""
        return self.send_message(text)

    def test_connection(self) -> bool:
        """Test if bot is properly configured."""
        if not self.enabled:
            print("\n  ❌ Telegram not configured.")
            print("  Add these to your .env file:")
            print("    TELEGRAM_BOT_TOKEN=your_bot_token")
            print("    TELEGRAM_CHAT_ID=your_chat_id")
            print("\n  See the setup instructions at top of this file.")
            return False

        success = self.send_message("✅ Trading bot connected successfully!")
        if success:
            print("  ✅ Telegram alert sent! Check your phone.")
        else:
            print("  ❌ Failed to send. Check your token and chat ID.")
        return success
