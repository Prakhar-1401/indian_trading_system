"""
Multi-Factor Stock Ranking Engine — THE CORE OF THE STRATEGY.

HOW IT WORKS (The Big Picture):
=================================

Step 1: Get all Nifty 500 stocks
Step 2: For each stock, calculate 4 factor scores:
        - Momentum Score (0-10): From technical indicators
        - Quality Score (0-10): From fundamentals
        - Sentiment Score (0-10): From news analysis
        - Smart Money Score (0-10): From institutional/insider flows
Step 3: Compute weighted composite score:
        Composite = (0.40 × Momentum) + (0.25 × Quality) + 
                    (0.20 × Sentiment) + (0.15 × Smart Money)
Step 4: Rank all stocks by composite score (highest = best)
Step 5: Apply filters (min market cap, max debt, etc.)
Step 6: Pick top 15 stocks for the portfolio

WHY THIS APPROACH?
- No single indicator is reliable alone
- Combining multiple factors reduces false signals
- Different market conditions favor different factors
- Weights are tunable through backtesting

WHEN INDICATORS CONFLICT (Common Question):
============================================
Example: Stock has great momentum (RSI strong, above 200 DMA) BUT 
         poor quality (high debt, low ROE) AND negative news.

In our system:
  Momentum: 8/10 × 0.40 = 3.2
  Quality:  2/10 × 0.25 = 0.5
  Sentiment: 1/10 × 0.20 = 0.2
  Smart Money: 5/10 × 0.15 = 0.75
  TOTAL: 4.65/10

This stock would rank BELOW a stock with balanced scores:
  Momentum: 6/10 × 0.40 = 2.4
  Quality:  7/10 × 0.25 = 1.75
  Sentiment: 6/10 × 0.20 = 1.2
  Smart Money: 6/10 × 0.15 = 0.9
  TOTAL: 6.25/10

So the balanced stock wins! The composite score naturally resolves conflicts.
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

from src.utils.helpers import load_config
from src.data.fetcher import DataManager
from src.indicators.technical import compute_momentum_score
from src.sentiment.news_analyzer import NewsSentimentAnalyzer
from src.smart_money.tracker import SmartMoneyTracker


class QualityScorer:
    """
    Scores stocks on fundamental quality (0-10).
    
    WHAT MAKES A "QUALITY" COMPANY:
    - High ROE (Return on Equity) → efficient use of shareholder money
    - Low Debt/Equity → not over-leveraged
    - Consistent profit growth → business is growing
    - High promoter holding → promoters believe in the company
    - Pays dividends → returns money to shareholders
    """

    @staticmethod
    def compute(fundamentals: dict) -> float:
        score = 0
        max_score = 10

        # ROE Score (0-3)
        roe = fundamentals.get("roe")
        if roe is not None:
            if roe > 20:
                score += 3
            elif roe > 15:
                score += 2
            elif roe > 12:
                score += 1

        # Debt/Equity Score (0-2)
        de = fundamentals.get("debt_to_equity")
        if de is not None:
            if de < 0.3:
                score += 2  # Very low debt
            elif de < 0.8:
                score += 1  # Manageable debt
            # High debt = 0 points

        # Profit Margins (0-2)
        margins = fundamentals.get("profit_margins")
        if margins is not None:
            if margins > 20:
                score += 2
            elif margins > 10:
                score += 1

        # PE Ratio (0-1) — Lower is better (but not too low = value trap)
        pe = fundamentals.get("pe_ratio")
        if pe is not None:
            if 10 < pe < 25:
                score += 1  # Reasonable valuation

        # Revenue Growth (0-1)
        growth = fundamentals.get("revenue_growth")
        if growth is not None:
            if growth > 15:
                score += 1

        # Dividend (0-1)
        div = fundamentals.get("dividend_yield")
        if div is not None and div > 1:
            score += 1

        return round(min(score, max_score), 2)


class StockRanker:
    """
    The main ranking engine. Combines all factors into a composite score.
    """

    def __init__(self, use_breeze: bool = False):
        self.config = load_config()
        self.dm = DataManager(use_breeze=use_breeze)
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.smart_money = SmartMoneyTracker()
        self.quality_scorer = QualityScorer()

        # Load factor weights from config
        weights = self.config.get("factor_weights", {})
        self.w_momentum = weights.get("momentum", 0.40)
        self.w_quality = weights.get("quality", 0.25)
        self.w_sentiment = weights.get("sentiment", 0.20)
        self.w_smart_money = weights.get("smart_money", 0.15)

    def score_stock(self, symbol: str) -> dict:
        """
        Calculate ALL factor scores for a single stock.
        Returns a dict with individual + composite scores.
        """
        logger.info(f"Scoring {symbol}...")

        result = {
            "symbol": symbol,
            "momentum_score": 0,
            "quality_score": 0,
            "sentiment_score": 0,
            "smart_money_score": 0,
            "composite_score": 0,
        }

        try:
            # --- Momentum ---
            df = self.dm.get_stock_data(symbol, period="2y")
            if not df.empty:
                result["momentum_score"] = compute_momentum_score(df)

            # --- Quality ---
            fundamentals = self.dm.get_fundamentals(symbol)
            result["quality_score"] = self.quality_scorer.compute(fundamentals)

            # Add fundamental data for reference
            result["pe_ratio"] = fundamentals.get("pe_ratio")
            result["roe"] = fundamentals.get("roe")
            result["debt_to_equity"] = fundamentals.get("debt_to_equity")
            result["market_cap_cr"] = fundamentals.get("market_cap_cr")
            result["sector"] = fundamentals.get("sector")

            # --- Sentiment ---
            sent_data = self.sentiment_analyzer.get_stock_sentiment(symbol)
            # Convert from -10..+10 to 0..10 scale
            raw_sent = sent_data["sentiment_score"]
            result["sentiment_score"] = round((raw_sent + 10) / 2, 2)  # Normalize

            # --- Smart Money ---
            result["smart_money_score"] = self.smart_money.compute_smart_money_score(symbol)

        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")

        # --- Composite Score ---
        result["composite_score"] = round(
            self.w_momentum * result["momentum_score"]
            + self.w_quality * result["quality_score"]
            + self.w_sentiment * result["sentiment_score"]
            + self.w_smart_money * result["smart_money_score"],
            2,
        )

        return result

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply universe filters BEFORE ranking.
        Remove stocks that don't meet our minimum criteria.
        """
        config = self.config.get("universe", {})
        filtered = df.copy()

        # Min market cap
        min_mcap = config.get("min_market_cap_cr", 500)
        if "market_cap_cr" in filtered.columns:
            filtered = filtered[
                (filtered["market_cap_cr"].isna()) | (filtered["market_cap_cr"] >= min_mcap)
            ]

        # Max debt/equity
        max_de = config.get("max_debt_equity", 1.5)
        if "debt_to_equity" in filtered.columns:
            filtered = filtered[
                (filtered["debt_to_equity"].isna()) | (filtered["debt_to_equity"] <= max_de * 100)
            ]
            # yfinance returns D/E as percentage sometimes

        # Exclude sectors
        exclude = config.get("exclude_sectors", [])
        if exclude and "sector" in filtered.columns:
            filtered = filtered[~filtered["sector"].isin(exclude)]

        return filtered

    def rank_stocks(
        self, symbols: list = None, top_n: int = None
    ) -> pd.DataFrame:
        """
        MAIN METHOD: Score and rank all stocks.
        
        Args:
            symbols: List of symbols to rank (default: Nifty 500)
            top_n: Return only top N stocks (default: from config)
        
        Returns:
            DataFrame sorted by composite_score (highest first)
        """
        if symbols is None:
            symbols = self.dm.get_universe()

        if top_n is None:
            top_n = self.config.get("portfolio", {}).get("max_stocks", 15)

        logger.info(f"Ranking {len(symbols)} stocks...")

        # Score all stocks
        scores = []
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Scoring {symbol}")
            try:
                score = self.score_stock(symbol)
                scores.append(score)
            except Exception as e:
                logger.error(f"Skipping {symbol}: {e}")

        df = pd.DataFrame(scores)

        if df.empty:
            logger.warning("No stocks scored!")
            return df

        # Apply filters
        df = self.apply_filters(df)

        # Sort by composite score
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        logger.info(f"Top {top_n} stocks:")
        for _, row in df.head(top_n).iterrows():
            logger.info(
                f"  #{row['rank']} {row['symbol']}: "
                f"Composite={row['composite_score']:.2f} "
                f"(M={row['momentum_score']:.1f}, Q={row['quality_score']:.1f}, "
                f"S={row['sentiment_score']:.1f}, SM={row['smart_money_score']:.1f})"
            )

        return df.head(top_n) if top_n else df

    def rank_quick(self, symbols: list) -> pd.DataFrame:
        """
        Quick ranking using ONLY momentum + quality (no web scraping).
        Much faster — good for initial screening and backtesting.
        """
        scores = []
        for symbol in symbols:
            try:
                df = self.dm.get_stock_data(symbol, period="1y")
                fundamentals = self.dm.get_fundamentals(symbol)

                m_score = compute_momentum_score(df) if not df.empty else 0
                q_score = self.quality_scorer.compute(fundamentals)

                composite = 0.60 * m_score + 0.40 * q_score

                scores.append({
                    "symbol": symbol,
                    "momentum_score": m_score,
                    "quality_score": q_score,
                    "composite_score": round(composite, 2),
                    "pe_ratio": fundamentals.get("pe_ratio"),
                    "market_cap_cr": fundamentals.get("market_cap_cr"),
                    "sector": fundamentals.get("sector"),
                })
            except Exception as e:
                logger.error(f"Quick rank failed for {symbol}: {e}")

        df = pd.DataFrame(scores)
        if not df.empty:
            df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
        return df
