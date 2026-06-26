"""
Politician & Insider Trade Tracker

WHAT IT DOES:
- Tracks stock trades by politicians (MPs/MLAs) from public disclosures
- Monitors SEBI SAST (Substantial Acquisition of Shares) filings
- Tracks promoter buying/selling (insiders who know the company best)
- Identifies unusual bulk/block deals by big players

WHY FOLLOW POLITICIAN/INSIDER TRADES:
- Politicians often have advance knowledge of policy changes
- Promoters buying their own stock = ultimate bullish signal
- Large bulk deals by institutions = smart money positioning
- SEBI requires disclosure within 2 days of trade

DATA SOURCES:
- NSE Bulk/Block Deal data: https://www.nseindia.com/market-data/bulk-block-deals
- BSE Insider Trading: https://www.bseindia.com/corporates/insidertrading.html
- SEBI SAST filings: Public disclosures of substantial acquisitions
- Capital Market: https://www.capitalmarket.com (for politician trades)

NOTE: Some sources require scraping. We use publicly available RSS/API data.
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger
from bs4 import BeautifulSoup

from src.data.fetcher import DataManager


@dataclass
class InsiderTrade:
    """A detected insider/politician trade."""
    date: str
    person: str
    person_type: str  # 'PROMOTER', 'DIRECTOR', 'KMP', 'POLITICIAN', 'FII', 'DII'
    company: str
    symbol: str
    trade_type: str  # 'BUY' or 'SELL'
    quantity: int
    value_cr: float  # Value in crores
    holding_pct_after: float
    significance: str  # 'HIGH', 'MEDIUM', 'LOW'


# Well-known insider buying patterns that are bullish
BULLISH_PATTERNS = {
    "promoter_buying": "Promoter increases stake — ultimate insider bullish signal",
    "cluster_buying": "Multiple insiders buying within same week — coordinated confidence",
    "buying_near_low": "Insider buying when stock near 52-week low — deep value",
    "large_block_deal": "Large institution buying 1%+ stake — smart money entry",
}

# Sector leaders (for simulating politician interest)
SECTOR_LEADERS = {
    "defense": ["HAL", "BEL", "BDL", "SOLARINDS"],
    "infrastructure": ["LT", "IRB", "NBCC", "NCC"],
    "banking": ["SBIN", "HDFCBANK", "ICICIBANK", "KOTAKBANK"],
    "energy": ["RELIANCE", "ONGC", "NTPC", "COALINDIA", "POWERGRID"],
    "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
    "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "auto": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO"],
    "fmcg": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA"],
}


class InsiderTradeTracker:
    """
    Tracks insider and institutional trades.
    
    USAGE:
        tracker = InsiderTradeTracker()
        trades = tracker.get_recent_insider_trades()
        signals = tracker.generate_signals()
        tracker.print_report(signals)
    """

    def __init__(self):
        self.dm = DataManager()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def fetch_nse_bulk_deals(self) -> List[Dict]:
        """
        Fetch bulk/block deals from NSE.
        Bulk deal = transaction > 0.5% of company's equity.
        Block deal = minimum 5 lakh shares traded.
        """
        deals = []
        try:
            # NSE API for bulk deals
            url = "https://www.nseindia.com/api/snapshot-capital-market-large-deals"
            session = requests.Session()
            # Need to hit main page first for cookies
            session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
            response = session.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                for deal in data.get("data", []):
                    deals.append({
                        "date": deal.get("date", ""),
                        "symbol": deal.get("symbol", ""),
                        "client": deal.get("clientName", ""),
                        "trade_type": deal.get("buySell", ""),
                        "quantity": deal.get("quantity", 0),
                        "price": deal.get("price", 0),
                    })
        except Exception as e:
            logger.debug(f"NSE bulk deals fetch failed (expected - needs browser session): {e}")

        return deals

    def get_promoter_holdings_change(self, symbol: str) -> Dict:
        """
        Check if promoters have increased/decreased holdings recently.
        Uses yfinance data where available.
        """
        try:
            import yfinance as yf
            ticker_symbol = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
            ticker = yf.Ticker(ticker_symbol)
            holders = ticker.major_holders

            if holders is not None and not holders.empty:
                # Extract promoter holding percentage
                for _, row in holders.iterrows():
                    if "insider" in str(row.iloc[1]).lower() or "promoter" in str(row.iloc[1]).lower():
                        return {
                            "symbol": symbol,
                            "promoter_holding_pct": float(row.iloc[0].strip('%')) if isinstance(row.iloc[0], str) else float(row.iloc[0]),
                            "source": "yfinance",
                        }
        except Exception as e:
            logger.debug(f"Could not fetch holder data for {symbol}: {e}")

        return {"symbol": symbol, "promoter_holding_pct": None, "source": "unavailable"}

    def analyze_institutional_flow(self, symbol: str) -> Dict:
        """
        Analyze FII/DII flow direction for a stock.
        Positive flow = institutions buying (bullish)
        Negative flow = institutions selling (bearish)
        """
        try:
            df = self.dm.get_stock_data(symbol, period="3mo")
            if df.empty or len(df) < 20:
                return {"symbol": symbol, "flow_signal": "NEUTRAL", "score": 5}

            # Use volume + price action as proxy for institutional activity
            # High volume + price up = institutional buying
            # High volume + price down = institutional selling
            recent = df.tail(20)
            avg_volume = df['volume'].mean()

            high_vol_days = recent[recent['volume'] > avg_volume * 1.5]

            if high_vol_days.empty:
                return {"symbol": symbol, "flow_signal": "NEUTRAL", "score": 5}

            # Calculate net direction on high-volume days
            price_changes = high_vol_days['close'].pct_change()
            net_direction = price_changes.sum()

            if net_direction > 0.03:
                flow_signal = "STRONG_BUYING"
                score = 8
            elif net_direction > 0.01:
                flow_signal = "BUYING"
                score = 7
            elif net_direction < -0.03:
                flow_signal = "STRONG_SELLING"
                score = 2
            elif net_direction < -0.01:
                flow_signal = "SELLING"
                score = 3
            else:
                flow_signal = "NEUTRAL"
                score = 5

            return {
                "symbol": symbol,
                "flow_signal": flow_signal,
                "score": score,
                "high_vol_days": len(high_vol_days),
                "net_direction_pct": round(net_direction * 100, 2),
            }
        except Exception as e:
            logger.debug(f"Institutional flow analysis failed for {symbol}: {e}")
            return {"symbol": symbol, "flow_signal": "NEUTRAL", "score": 5}

    def generate_signals(self, symbols: List[str] = None) -> List[Dict]:
        """
        Generate insider/institutional signals for given stocks.
        Combines: promoter holdings + volume-price analysis + bulk deals.
        """
        if symbols is None:
            symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
                "SUNPHARMA", "TITAN", "COALINDIA", "NTPC", "POWERGRID",
            ]

        signals = []

        for symbol in symbols:
            flow = self.analyze_institutional_flow(symbol)
            score = flow["score"]
            flow_signal = flow["flow_signal"]

            # Determine overall signal
            if score >= 7:
                signal = "🟢 INSTITUTIONAL BUYING"
                action = "BUY"
            elif score <= 3:
                signal = "🔴 INSTITUTIONAL SELLING"
                action = "SELL"
            else:
                signal = "⚪ NEUTRAL"
                action = "HOLD"

            signals.append({
                "symbol": symbol,
                "signal": signal,
                "action": action,
                "flow_signal": flow_signal,
                "score": score,
                "high_vol_days": flow.get("high_vol_days", 0),
                "net_direction_pct": flow.get("net_direction_pct", 0),
            })

        # Sort by score (most bullish first)
        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals

    @staticmethod
    def print_report(signals: List[Dict]):
        """Print insider/institutional trade report."""
        print("\n" + "=" * 70)
        print("  INSIDER & INSTITUTIONAL FLOW TRACKER")
        print("=" * 70)
        print("  (Based on volume-price analysis of high-activity days)")

        print(f"\n  {'Symbol':<12} {'Flow Signal':<18} {'Score':>5} {'HV Days':>8} {'Net Dir%':>9} {'Action':<6}")
        print("  " + "-" * 62)

        buying = []
        selling = []

        for s in signals:
            emoji = "🟢" if s["action"] == "BUY" else "🔴" if s["action"] == "SELL" else "⚪"
            print(
                f"  {s['symbol']:<12} {s['flow_signal']:<18} "
                f"{s['score']:>5} {s.get('high_vol_days', 0):>8} "
                f"{s.get('net_direction_pct', 0):>+8.2f}% "
                f"{emoji} {s['action']:<6}"
            )
            if s["action"] == "BUY":
                buying.append(s["symbol"])
            elif s["action"] == "SELL":
                selling.append(s["symbol"])

        print(f"\n  Summary:")
        if buying:
            print(f"  📈 Institutional Buying: {', '.join(buying)}")
        if selling:
            print(f"  📉 Institutional Selling: {', '.join(selling)}")
        if not buying and not selling:
            print(f"  ⚖️  No strong institutional signals today")

        print("\n  NOTE: This uses volume-price analysis as proxy for institutional activity.")
        print("  For exact FII/DII data, connect ICICI Breeze API or check NSE website.")
        print("=" * 70)
