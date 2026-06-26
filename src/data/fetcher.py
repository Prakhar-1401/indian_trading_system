"""
Data Fetching Layer — ICICI Direct Breeze API + yfinance fallback.

WHY TWO SOURCES?
- ICICI Direct Breeze API: Real-time data, your actual broker, can place orders
- yfinance: Free historical data, great for backtesting, no API key needed

HOW IT WORKS:
1. For backtesting → yfinance (free, unlimited historical data)
2. For live trading → ICICI Breeze API (real-time, connected to your account)
3. For fundamentals → screener.in scraping + yfinance
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from src.utils.helpers import load_config, get_env, get_project_root

# Cache directory for downloaded data
CACHE_DIR = get_project_root() / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BreezeDataFetcher:
    """
    Fetches data from ICICI Direct Breeze API.
    
    TO SET UP:
    1. Go to https://api.icicidirect.com/apiuser/home
    2. Log in with your ICICI Direct credentials
    3. Click 'Register Now' if first time, or 'View' for existing keys
    4. Copy API Key and Secret to your .env file
    5. Session token must be generated DAILY (it expires every day)
       - Log in to ICICI Direct website
       - The session token is in the URL after login
       - Or use the generate_session() helper below
    """

    def __init__(self):
        self._client = None

    def connect(self):
        """Initialize Breeze API connection."""
        try:
            from breeze_connect import BreezeConnect

            api_key = get_env("ICICI_API_KEY")
            api_secret = get_env("ICICI_API_SECRET")
            session_token = get_env("ICICI_SESSION_TOKEN")

            self._client = BreezeConnect(api_key=api_key)
            self._client.generate_session(
                api_secret=api_secret, session_token=session_token
            )
            logger.info("Connected to ICICI Direct Breeze API successfully.")
            return True
        except Exception as e:
            logger.warning(f"Breeze API connection failed: {e}. Falling back to yfinance.")
            return False

    def get_historical_data(
        self, symbol: str, interval: str = "1day",
        from_date: str = None, to_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from ICICI Breeze.
        
        Args:
            symbol: NSE symbol (e.g., 'RELIANCE', 'TCS')
            interval: '1minute', '5minute', '1day', etc.
            from_date: Start date 'YYYY-MM-DD'
            to_date: End date 'YYYY-MM-DD'
        """
        if not self._client:
            raise ConnectionError("Not connected. Call connect() first.")

        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%dT07:00:00.000Z")
        else:
            to_date = f"{to_date}T07:00:00.000Z"

        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime(
                "%Y-%m-%dT07:00:00.000Z"
            )
        else:
            from_date = f"{from_date}T07:00:00.000Z"

        data = self._client.get_historical_data_v2(
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            stock_code=symbol,
            exchange_code="NSE",
            product_type="cash",
        )

        if data and "Success" in data.get("Status", ""):
            df = pd.DataFrame(data["Success"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        else:
            logger.error(f"No data returned for {symbol}: {data}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> dict:
        """Get real-time quote for a symbol."""
        if not self._client:
            raise ConnectionError("Not connected. Call connect() first.")
        return self._client.get_quotes(
            stock_code=symbol, exchange_code="NSE", product_type="cash"
        )


class YFinanceDataFetcher:
    """
    Fetches data from Yahoo Finance — FREE, no API key needed.
    Best for backtesting and historical analysis.
    
    Indian stocks on yfinance use '.NS' suffix: 'RELIANCE.NS', 'TCS.NS'
    """

    @staticmethod
    def get_historical_data(
        symbol: str, period: str = "2y", interval: str = "1d",
        start: str = None, end: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: NSE symbol (auto-appends .NS if needed)
            period: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
            interval: '1d', '1wk', '1mo'
            start/end: 'YYYY-MM-DD' (overrides period if both provided)
        """
        # Auto-append .NS for Indian stocks
        if not symbol.endswith(".NS") and not symbol.startswith("^"):
            symbol = f"{symbol}.NS"

        # Cache key must distinguish period-based pulls from explicit date-range pulls.
        # Otherwise multi-window backtests can accidentally reuse the wrong cached slice.
        if start and end:
            cache_key = f"{symbol.replace('.', '_')}_{interval}_{start}_{end}"
        else:
            cache_key = f"{symbol.replace('.', '_')}_{interval}_{period}"
        cache_file = CACHE_DIR / f"{cache_key}.csv"

        # Check cache (data less than 4 hours old)
        if cache_file.exists():
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 14400:  # 4 hours
                logger.debug(f"Loading cached data for {symbol}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df

        logger.info(f"Downloading data for {symbol} ({period}, {interval})")
        ticker = yf.Ticker(symbol)

        if start and end:
            df = ticker.history(start=start, end=end, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        # Standardize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Drop rows with NaN close (yfinance returns a partial row for an
        # incomplete/non-trading current day, which poisons every downstream calc)
        if "close" in df.columns:
            df = df[df["close"].notna()]

        if df.empty:
            logger.warning(f"All rows NaN after cleaning for {symbol}")
            return df

        # Cache it
        df.to_csv(cache_file)
        return df

    @staticmethod
    def get_fundamentals(symbol: str) -> dict:
        """
        Get fundamental data: PE, PB, ROE, market cap, etc.
        This is KEY for the Quality factor in our ranking.
        """
        if not symbol.endswith(".NS"):
            symbol = f"{symbol}.NS"

        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol.replace(".NS", ""),
            "market_cap_cr": info.get("marketCap", 0) / 1e7,  # Convert to Crores
            "pe_ratio": info.get("trailingPE", None),
            "pb_ratio": info.get("priceToBook", None),
            "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
            "debt_to_equity": info.get("debtToEquity", None),
            "profit_margins": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
            "revenue_growth": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
            "promoter_holding": None,  # Not available on yfinance, scraped separately
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": info.get("currentPrice", None),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", 0),
        }

    @staticmethod
    def get_nifty500_symbols() -> list:
        """
        Get list of Nifty 500 stocks.
        We scrape it from NSE India's website.
        """
        try:
            url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
            df = pd.read_csv(url)
            symbols = df["Symbol"].tolist()
            logger.info(f"Loaded {len(symbols)} Nifty 500 symbols")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch Nifty 500 list: {e}")
            # Fallback: Top 50 most traded stocks
            return [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
                "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
                "NESTLEIND", "TECHM", "POWERGRID", "NTPC", "TATAMOTORS",
                "ONGC", "JSWSTEEL", "M&M", "ADANIENT", "ADANIPORTS",
                "TATASTEEL", "BAJAJFINSV", "COALINDIA", "GRASIM", "DIVISLAB",
                "BPCL", "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO",
                "APOLLOHOSP", "BRITANNIA", "HINDALCO", "INDUSINDBK", "SBILIFE",
                "TATACONSUM", "DABUR", "PIDILITIND", "HAVELLS", "GODREJCP",
            ]


class DataManager:
    """
    Unified data interface. Tries ICICI Breeze first, falls back to yfinance.
    
    USAGE:
        dm = DataManager()
        df = dm.get_stock_data("RELIANCE", period="1y")
        fundamentals = dm.get_fundamentals("TCS")
    """

    def __init__(self, use_breeze: bool = False):
        self.yf = YFinanceDataFetcher()
        self.breeze = None

        if use_breeze:
            self.breeze = BreezeDataFetcher()
            if not self.breeze.connect():
                self.breeze = None

    def get_stock_data(
        self, symbol: str, period: str = "2y", interval: str = "1d",
        start: str = None, end: str = None, source: str = "auto"
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a stock.
        source: 'auto' (try breeze, fall back to yf), 'breeze', 'yfinance'
        """
        if source == "breeze" or (source == "auto" and self.breeze):
            try:
                return self.breeze.get_historical_data(
                    symbol, from_date=start, to_date=end
                )
            except Exception as e:
                logger.warning(f"Breeze failed for {symbol}: {e}, using yfinance")

        return self.yf.get_historical_data(
            symbol, period=period, interval=interval, start=start, end=end
        )

    def get_fundamentals(self, symbol: str) -> dict:
        return self.yf.get_fundamentals(symbol)

    def get_universe(self) -> list:
        """Get the stock universe (Nifty 500)."""
        return self.yf.get_nifty500_symbols()

    def get_bulk_data(self, symbols: list, period: str = "1y") -> dict:
        """Download data for multiple stocks. Returns {symbol: DataFrame}."""
        result = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching {symbol} ({i+1}/{len(symbols)})")
            try:
                df = self.get_stock_data(symbol, period=period)
                if not df.empty:
                    result[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return result
