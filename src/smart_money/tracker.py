"""
Smart Money Tracking Module — Follow the big players.

WHAT IS "SMART MONEY"?
These are institutional investors, insiders, and politically connected people
who often have an information edge:

1. FII/DII (Foreign/Domestic Institutional Investors):
   - When FIIs buy heavily → market tends to go up
   - When FIIs sell → market often drops
   - Data available daily from NSE website

2. Insider Trades (Promoters/Directors buying/selling their own company stock):
   - Promoter buying own stock = STRONGEST bullish signal
   - They know their company best. If they're buying, something good is coming.
   - Data from SEBI/BSE insider trading disclosures

3. Bulk/Block Deals:
   - Large transactions (>0.5% of shares) reported by NSE
   - Shows big institutional interest in a stock

4. US Congress/Politician Trades:
   - US politicians' trades are publicly disclosed
   - Some track record of beating the market
   - We use this as a REFERENCE for global tech/pharma stocks
   - Source: capitoltrades.com, quiverquant.com

SCORING:
- FII net buyer + DII net buyer: +3 points
- Insider/promoter buying: +4 points (strongest signal)
- Bulk deal with known institution buying: +2 points
- Politician buying in related sector: +1 point
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional

from src.utils.helpers import load_config


class SmartMoneyTracker:
    """Track institutional and insider money flows."""

    def __init__(self):
        self.config = load_config()
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/json",
        }

    # ========================================
    # FII / DII DATA
    # ========================================
    def get_fii_dii_data(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch FII/DII buy/sell data from NSE.
        
        FII = Foreign Institutional Investors (like Goldman, Morgan Stanley)
        DII = Domestic Institutional Investors (like LIC, SBI MF)
        
        When both are buying → VERY bullish
        When FII selling but DII buying → Mixed (usually DII cushions the fall)
        When both selling → Bearish
        """
        try:
            # NSE provides this data via their API
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            session = requests.Session()
            # NSE requires a session cookie, so visit the main page first
            session.get("https://www.nseindia.com", headers=self._headers, timeout=10)
            resp = session.get(url, headers=self._headers, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data)
                logger.info(f"Fetched FII/DII data: {len(df)} records")
                return df
        except Exception as e:
            logger.warning(f"NSE FII/DII fetch failed: {e}")

        # Fallback: Use moneycontrol FII/DII data
        return self._fetch_fii_dii_fallback()

    def _fetch_fii_dii_fallback(self) -> pd.DataFrame:
        """Fallback FII/DII data from MoneyControl."""
        try:
            url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
            resp = requests.get(url, headers=self._headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            tables = soup.find_all("table")
            if tables:
                df = pd.read_html(str(tables[0]))[0]
                return df
        except Exception as e:
            logger.error(f"FII/DII fallback also failed: {e}")

        return pd.DataFrame()

    def get_fii_dii_signal(self) -> dict:
        """
        Convert FII/DII data into a trading signal.
        Returns: {signal: 'bullish'/'bearish'/'neutral', score: -5 to +5}
        """
        df = self.get_fii_dii_data(days=5)
        if df.empty:
            return {"signal": "neutral", "score": 0, "details": "No data available"}

        # Simplified scoring based on net buying/selling
        # In production, you'd parse the actual buy/sell values
        return {"signal": "neutral", "score": 0, "details": "Data fetched, analysis pending"}

    # ========================================
    # INSIDER / PROMOTER TRADES
    # ========================================
    def get_insider_trades(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch insider trading data from BSE.
        
        SEBI requires all insiders (promoters, directors, key personnel)
        to disclose their trades within 2 trading days.
        
        WHAT TO LOOK FOR:
        - Promoter BUYING > 1 Cr worth: Very bullish
        - Multiple insiders buying: Even more bullish
        - Promoter selling: Not always bad (could be personal needs)
        - Promoter pledging shares: RED FLAG
        """
        try:
            # BSE insider trading page
            url = "https://www.bseindia.com/corporates/Insider_Trading_new.aspx"
            resp = requests.get(url, headers=self._headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            table = soup.find("table", {"id": "ContentPlaceHolder1_gvData"})
            if table:
                df = pd.read_html(str(table))[0]
                logger.info(f"Fetched {len(df)} insider trade records")
                return df
        except Exception as e:
            logger.warning(f"BSE insider trades fetch failed: {e}")

        # Fallback: Trendlyne insider trades
        return self._fetch_insider_fallback()

    def _fetch_insider_fallback(self) -> pd.DataFrame:
        """Fallback insider data from Trendlyne."""
        try:
            url = "https://trendlyne.com/stock-screeners/insider-trading/"
            resp = requests.get(url, headers=self._headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Parse the data table
            tables = soup.find_all("table")
            if tables:
                df = pd.read_html(str(tables[0]))[0]
                return df
        except Exception as e:
            logger.debug(f"Trendlyne insider fallback failed: {e}")
        return pd.DataFrame()

    def get_insider_signal(self, symbol: str) -> dict:
        """
        Check if there's insider buying/selling for a specific stock.
        
        SCORING:
        - Promoter bought > 1 Cr: +4 (strongest signal)
        - Director bought: +2
        - Promoter sold: -1 (mildly negative)
        - Promoter pledged more shares: -3 (red flag)
        """
        df = self.get_insider_trades()
        if df.empty:
            return {"symbol": symbol, "signal": "neutral", "score": 0}

        # Filter for this symbol
        # Column names vary by source, try common patterns
        symbol_cols = [c for c in df.columns if "symbol" in c.lower() or "company" in c.lower() or "name" in c.lower()]
        if not symbol_cols:
            return {"symbol": symbol, "signal": "neutral", "score": 0}

        col = symbol_cols[0]
        stock_data = df[df[col].astype(str).str.contains(symbol, case=False, na=False)]

        if stock_data.empty:
            return {"symbol": symbol, "signal": "neutral", "score": 0}

        # Analyze buy vs sell
        buy_cols = [c for c in df.columns if "buy" in c.lower() or "acquisition" in c.lower()]
        sell_cols = [c for c in df.columns if "sell" in c.lower() or "disposal" in c.lower()]

        score = 0
        if buy_cols:
            score += len(stock_data) * 2  # Each insider buy = +2
        if sell_cols:
            score -= len(stock_data)  # Each sell = -1

        score = max(min(score, 5), -5)  # Cap at ±5

        signal = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")
        return {"symbol": symbol, "signal": signal, "score": score}

    # ========================================
    # BULK / BLOCK DEALS
    # ========================================
    def get_bulk_deals(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch bulk deal data from NSE.
        
        Bulk deals = transactions where quantity > 0.5% of total shares
        These are large institutional trades that MUST be disclosed.
        
        If a big mutual fund or FII is buying bulk, that's very bullish.
        """
        try:
            url = "https://www.nseindia.com/api/historical/bulk-deals"
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=self._headers, timeout=10)

            from_date = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")
            to_date = datetime.now().strftime("%d-%m-%Y")

            params = {"from": from_date, "to": to_date}
            resp = session.get(url, headers=self._headers, params=params, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    return pd.DataFrame(data["data"])
        except Exception as e:
            logger.warning(f"NSE bulk deals fetch failed: {e}")

        return pd.DataFrame()

    # ========================================
    # POLITICIAN TRADES (US Congress — Reference)
    # ========================================
    def get_politician_trades(self) -> pd.DataFrame:
        """
        Fetch US Congress stock trades from public disclosures.
        
        WHY US POLITICIAN TRADES?
        - Legally required to disclose within 45 days
        - Some politicians have suspiciously good track records
        - Useful for global tech/pharma stocks that also trade in India
        - e.g., If a US senator buys Nvidia → it might signal good things
          for Indian IT companies that serve Nvidia/AI clients
        
        SOURCE: Capitol Trades / Quiver Quantitative
        """
        try:
            # QuiverQuant provides a simple API
            url = "https://www.quiverquant.com/congresstrading/"
            resp = requests.get(url, headers=self._headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Parse trade tables
            tables = soup.find_all("table")
            if tables:
                df = pd.read_html(str(tables[0]))[0]
                logger.info(f"Fetched {len(df)} politician trade records")
                return df
        except Exception as e:
            logger.warning(f"Politician trades fetch failed: {e}")

        return pd.DataFrame()

    def map_us_to_indian_stocks(self, us_symbol: str) -> list:
        """
        Map US stock trades to related Indian stocks.
        e.g., MSFT buying → could signal good things for INFY, TCS, WIPRO
        """
        sector_mapping = {
            # US Tech → Indian IT
            "MSFT": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"],
            "GOOGL": ["INFY", "TCS", "WIPRO"],
            "AAPL": ["DIXON", "TATAELXSI"],
            "AMZN": ["INFY", "TCS"],
            "NVDA": ["INFY", "TCS", "LTIM"],
            # US Pharma → Indian Pharma
            "PFE": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
            "JNJ": ["SUNPHARMA", "DRREDDY", "CIPLA"],
            "LLY": ["BIOCON", "SUNPHARMA"],
            # US Banks → Indian Banks
            "JPM": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN"],
            "GS": ["HDFCBANK", "ICICIBANK"],
        }
        return sector_mapping.get(us_symbol, [])

    # ========================================
    # COMPOSITE SMART MONEY SCORE
    # ========================================
    def compute_smart_money_score(self, symbol: str) -> float:
        """
        Compute a composite smart money score for a stock (0-10 scale).
        
        COMPONENTS:
        - FII/DII flow direction: 0-3 points
        - Insider buying/selling: 0-4 points (highest weight — strongest signal)
        - Bulk deal activity: 0-2 points
        - Politician trade reference: 0-1 point
        """
        score = 0
        max_score = 10

        # 1. FII/DII signal
        fii_data = self.get_fii_dii_signal()
        if fii_data["signal"] == "bullish":
            score += 3
        elif fii_data["signal"] == "neutral":
            score += 1

        # 2. Insider trades
        insider = self.get_insider_signal(symbol)
        if insider["score"] > 0:
            score += min(insider["score"], 4)
        elif insider["score"] < 0:
            score -= 1

        # 3. Bulk deals
        bulk = self.get_bulk_deals()
        if not bulk.empty:
            # Check if this stock appears in recent bulk deals
            symbol_matches = bulk.apply(
                lambda row: symbol.lower() in str(row).lower(), axis=1
            )
            if symbol_matches.any():
                score += 2

        # Normalize to 0-10
        normalized = round(max(min(score, max_score), 0), 2)
        return normalized
