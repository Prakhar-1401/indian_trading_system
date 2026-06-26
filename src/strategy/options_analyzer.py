"""
Options Chain Analysis — Read what the "smart money" is betting on.

WHAT IS OPTIONS CHAIN ANALYSIS?
================================
Options data reveals what BIG players (FIIs, institutions) expect:
- High Put buying → Institutions expect a DROP (hedging)
- High Call buying → Institutions expect a RISE
- Max Pain → The price where most options expire worthless (market makers profit)

KEY CONCEPTS:
=============
1. PUT-CALL RATIO (PCR):
   - PCR = Total Put OI / Total Call OI
   - PCR > 1.2 → Market is OVERSOLD (too bearish = contrarian BUY)
   - PCR < 0.7 → Market is OVERBOUGHT (too bullish = contrarian SELL)
   - PCR 0.8-1.2 → Neutral

2. MAX PAIN:
   - The strike price where option SELLERS make the most money
   - Markets tend to gravitate toward max pain at expiry
   - If NIFTY is at 22000 and max pain is 22500 → likely to drift UP

3. IV (Implied Volatility):
   - High IV → Market expects big move (good for selling options)
   - Low IV → Market expects calm (good for buying options)

NOTE: NSE options data requires web scraping or paid API.
This module uses yfinance for available options data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import requests

from src.data.fetcher import DataManager


@dataclass
class OptionsAnalysis:
    """Options chain analysis results."""
    symbol: str
    spot_price: float
    pcr: float  # Put-Call Ratio
    pcr_signal: str  # BULLISH, BEARISH, NEUTRAL
    max_pain: float
    max_pain_signal: str  # Market likely direction based on max pain
    iv_percentile: float  # Current IV vs historical
    key_support: float  # Highest put OI strike (support)
    key_resistance: float  # Highest call OI strike (resistance)
    recommendation: str


class OptionsChainAnalyzer:
    """
    Analyze options chain data for trading signals.
    
    USAGE:
        analyzer = OptionsChainAnalyzer()
        result = analyzer.analyze("NIFTY")
        analyzer.print_report([result])
    """

    def __init__(self):
        self.dm = DataManager()

    def _fetch_nse_options(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch options chain from NSE.
        
        NOTE: NSE blocks automated requests. This uses headers to
        mimic a browser. May need updating if NSE changes their site.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/option-chain',
        }

        # Map symbol names
        nse_symbol = symbol.upper()
        if nse_symbol in ["NIFTY", "NIFTY50", "^NSEI"]:
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        elif nse_symbol in ["BANKNIFTY", "^NSEBANK"]:
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}"

        try:
            session = requests.Session()
            # First hit main page to get cookies
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            # Then fetch options data
            response = session.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                records = data.get('records', {}).get('data', [])
                if records:
                    return self._parse_nse_data(records, data)
            else:
                logger.warning(f"NSE API returned {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"NSE fetch failed: {e}. Using synthetic analysis.")
            return None

    def _parse_nse_data(self, records: List, full_data: Dict) -> pd.DataFrame:
        """Parse NSE options chain JSON into DataFrame."""
        rows = []
        for record in records:
            strike = record.get('strikePrice', 0)
            ce = record.get('CE', {})
            pe = record.get('PE', {})

            rows.append({
                'strike': strike,
                'call_oi': ce.get('openInterest', 0),
                'call_volume': ce.get('totalTradedVolume', 0),
                'call_iv': ce.get('impliedVolatility', 0),
                'call_ltp': ce.get('lastPrice', 0),
                'put_oi': pe.get('openInterest', 0),
                'put_volume': pe.get('totalTradedVolume', 0),
                'put_iv': pe.get('impliedVolatility', 0),
                'put_ltp': pe.get('lastPrice', 0),
            })

        return pd.DataFrame(rows)

    def _synthetic_analysis(self, symbol: str) -> Optional[OptionsAnalysis]:
        """
        When NSE API is blocked, use price action to ESTIMATE options metrics.
        
        This uses:
        - Historical volatility as IV proxy
        - Round numbers as likely strike levels
        - Volume patterns to infer institutional positioning
        """
        # Get spot price
        ticker = symbol
        if symbol in ["NIFTY", "NIFTY50"]:
            ticker = "^NSEI"
        elif symbol == "BANKNIFTY":
            ticker = "^NSEBANK"

        df = self.dm.get_stock_data(ticker, period="6mo")
        if df.empty:
            return None

        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        spot = df['close'].iloc[-1]
        returns = df['close'].pct_change().dropna()

        # Historical volatility as IV proxy
        hv_20 = returns.tail(20).std() * np.sqrt(252) * 100
        hv_60 = returns.tail(60).std() * np.sqrt(252) * 100
        iv_percentile = min((hv_20 / hv_60) * 50, 100) if hv_60 > 0 else 50

        # Synthetic PCR from price action
        # If market is going up → PCR tends to be low (less put buying)
        # If market is going down → PCR tends to be high (more put buying)
        recent_return = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
        # Invert: negative return → high PCR (bearish sentiment)
        synthetic_pcr = 1.0 - recent_return * 5
        synthetic_pcr = np.clip(synthetic_pcr, 0.4, 2.0)

        # PCR signal (CONTRARIAN indicator)
        if synthetic_pcr > 1.2:
            pcr_signal = "BULLISH"  # Too many puts = contrarian buy
        elif synthetic_pcr < 0.7:
            pcr_signal = "BEARISH"  # Too many calls = contrarian sell
        else:
            pcr_signal = "NEUTRAL"

        # Max pain estimation (round number closest to recent mean)
        if symbol in ["NIFTY", "NIFTY50"]:
            step = 50  # NIFTY strikes every 50
        elif symbol == "BANKNIFTY":
            step = 100
        else:
            step = max(round(spot * 0.01 / 5) * 5, 5)  # 1% steps, rounded to 5

        sma20 = df['close'].rolling(20).mean().iloc[-1]
        max_pain = round(sma20 / step) * step

        # Max pain signal
        if spot < max_pain * 0.98:
            max_pain_signal = "BULLISH (spot below max pain → likely to rise)"
        elif spot > max_pain * 1.02:
            max_pain_signal = "BEARISH (spot above max pain → likely to fall)"
        else:
            max_pain_signal = "NEUTRAL (spot near max pain)"

        # Support/Resistance from round numbers
        key_support = round((spot * 0.97) / step) * step
        key_resistance = round((spot * 1.03) / step) * step

        # Recommendation
        if pcr_signal == "BULLISH" and max_pain > spot:
            recommendation = "BUY — PCR extreme + max pain above → strong bullish setup"
        elif pcr_signal == "BEARISH" and max_pain < spot:
            recommendation = "SELL — Low PCR + max pain below → bearish pressure"
        elif iv_percentile > 70:
            recommendation = "SELL OPTIONS — IV is high, premium is rich"
        else:
            recommendation = "NEUTRAL — No clear edge from options data"

        return OptionsAnalysis(
            symbol=symbol,
            spot_price=round(spot, 2),
            pcr=round(synthetic_pcr, 2),
            pcr_signal=pcr_signal,
            max_pain=max_pain,
            max_pain_signal=max_pain_signal,
            iv_percentile=round(iv_percentile, 1),
            key_support=key_support,
            key_resistance=key_resistance,
            recommendation=recommendation,
        )

    def analyze(self, symbol: str) -> Optional[OptionsAnalysis]:
        """
        Analyze options chain for a symbol.
        Tries NSE API first, falls back to synthetic analysis.
        """
        # Try real NSE data
        chain = self._fetch_nse_options(symbol)

        if chain is not None and not chain.empty:
            return self._analyze_chain(symbol, chain)
        else:
            # Fallback to synthetic
            return self._synthetic_analysis(symbol)

    def _analyze_chain(self, symbol: str, chain: pd.DataFrame) -> OptionsAnalysis:
        """Analyze real options chain data."""
        # Get spot
        ticker = symbol if symbol not in ["NIFTY", "NIFTY50"] else "^NSEI"
        df = self.dm.get_stock_data(ticker, period="3mo")
        spot = df['close'].iloc[-1] if not df.empty else 0

        # PCR
        total_put_oi = chain['put_oi'].sum()
        total_call_oi = chain['call_oi'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        if pcr > 1.2:
            pcr_signal = "BULLISH"
        elif pcr < 0.7:
            pcr_signal = "BEARISH"
        else:
            pcr_signal = "NEUTRAL"

        # Max Pain calculation
        strikes = chain['strike'].unique()
        max_pain_value = 0
        max_pain_strike = spot

        for strike in strikes:
            # Total value of options that expire worthless at this strike
            call_pain = chain[chain['strike'] > strike]['call_oi'].sum() * (chain[chain['strike'] > strike]['strike'] - strike).sum()
            put_pain = chain[chain['strike'] < strike]['put_oi'].sum() * (strike - chain[chain['strike'] < strike]['strike']).sum()
            total_pain = call_pain + put_pain

            if total_pain > max_pain_value:
                max_pain_value = total_pain
                max_pain_strike = strike

        # Key levels
        key_resistance = chain.loc[chain['call_oi'].idxmax(), 'strike']
        key_support = chain.loc[chain['put_oi'].idxmax(), 'strike']

        # IV
        iv_mean = chain[['call_iv', 'put_iv']].mean().mean()

        if spot < max_pain_strike * 0.98:
            max_pain_signal = "BULLISH"
        elif spot > max_pain_strike * 1.02:
            max_pain_signal = "BEARISH"
        else:
            max_pain_signal = "NEUTRAL"

        recommendation = f"PCR={pcr:.2f} ({pcr_signal}), Max Pain at {max_pain_strike}"

        return OptionsAnalysis(
            symbol=symbol,
            spot_price=round(spot, 2),
            pcr=round(pcr, 2),
            pcr_signal=pcr_signal,
            max_pain=max_pain_strike,
            max_pain_signal=max_pain_signal,
            iv_percentile=round(iv_mean, 1),
            key_support=key_support,
            key_resistance=key_resistance,
            recommendation=recommendation,
        )

    def analyze_batch(self, symbols: List[str] = None) -> List[OptionsAnalysis]:
        """Analyze multiple symbols."""
        if symbols is None:
            symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK"]

        results = []
        for symbol in symbols:
            result = self.analyze(symbol)
            if result:
                results.append(result)
        return results

    def print_report(self, results: List[OptionsAnalysis]):
        """Print options analysis report."""
        print("\n" + "=" * 70)
        print("  OPTIONS CHAIN ANALYSIS")
        print("=" * 70)
        print("  PCR > 1.2 = Bullish (contrarian) | PCR < 0.7 = Bearish")
        print("  Max Pain = Where market gravitates at expiry")

        print(f"\n  {'Symbol':<12} {'Spot':>9} {'PCR':>5} {'Signal':<9} "
              f"{'MaxPain':>9} {'Support':>9} {'Resist':>9} {'IV%':>5}")
        print("  " + "-" * 75)

        for r in results:
            emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(r.pcr_signal, "⚪")
            print(
                f"  {r.symbol:<12} "
                f"₹{r.spot_price:>8,.0f} "
                f"{r.pcr:>5.2f} "
                f"{emoji}{r.pcr_signal:<8} "
                f"₹{r.max_pain:>8,.0f} "
                f"₹{r.key_support:>8,.0f} "
                f"₹{r.key_resistance:>8,.0f} "
                f"{r.iv_percentile:>4.0f}%"
            )

        print(f"\n  Detailed Analysis:")
        for r in results:
            print(f"    {r.symbol}: {r.max_pain_signal}")
            print(f"      → {r.recommendation}")

        print("\n  NOTE: If NSE API is blocked by corporate firewall,")
        print("  synthetic analysis uses price action as proxy.")
        print("=" * 70)
