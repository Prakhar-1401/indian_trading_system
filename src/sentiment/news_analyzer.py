"""
News Sentiment Analysis Module.

WHAT THIS DOES:
1. Scrapes financial news from Indian sources (MoneyControl, ET, LiveMint)
2. Filters articles mentioning specific stock names/companies
3. Runs VADER sentiment analysis on each article
4. Returns a sentiment score per stock (-10 to +10)

WHY SENTIMENT MATTERS:
- Positive news = institutional buying → price goes up
- Negative news (scam, earnings miss) = panic selling → price drops
- News often LEADS price action by 1-3 days
- Combined with technical momentum, it's very powerful

SENTIMENT SCORING:
- VADER gives compound score from -1 to +1
- We average across all articles about a stock
- Then scale to -10 to +10 for our ranking system
"""
import re
import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
from typing import Optional

from src.utils.helpers import load_config, get_env


class NewsSentimentAnalyzer:
    """Scrapes and analyzes news sentiment for Indian stocks."""

    def __init__(self):
        self.config = load_config()
        self.analyzer = SentimentIntensityAnalyzer()
        self.rss_feeds = self.config.get("sentiment", {}).get("news_sources", [
            "https://economictimes.indiatimes.com/markets/rss",
            "https://www.moneycontrol.com/rss/latestnews.xml",
            "https://www.livemint.com/rss/markets",
        ])
        self.lookback_days = self.config.get("sentiment", {}).get("lookback_days", 7)
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        # Company name mappings for better article matching
        self._company_aliases = {
            "RELIANCE": ["reliance", "reliance industries", "ril", "jio", "mukesh ambani"],
            "TCS": ["tcs", "tata consultancy", "tata consulting"],
            "HDFCBANK": ["hdfc bank", "hdfc"],
            "INFY": ["infosys", "infy"],
            "ICICIBANK": ["icici bank", "icici"],
            "HINDUNILVR": ["hindustan unilever", "hul"],
            "SBIN": ["sbi", "state bank of india", "state bank"],
            "BHARTIARTL": ["bharti airtel", "airtel"],
            "ITC": ["itc limited", "itc"],
            "KOTAKBANK": ["kotak mahindra", "kotak bank"],
            "BAJFINANCE": ["bajaj finance", "bajaj finserv"],
            "TATAMOTORS": ["tata motors", "tata motor"],
            "WIPRO": ["wipro"],
            "MARUTI": ["maruti suzuki", "maruti"],
            "ADANIENT": ["adani enterprises", "adani"],
        }

    def fetch_rss_articles(self) -> list:
        """
        Fetch articles from RSS feeds.
        RSS = Really Simple Syndication. It's a standardized way websites
        publish their latest articles. We parse these XML feeds to get headlines.
        """
        articles = []
        cutoff = datetime.now() - timedelta(days=self.lookback_days)

        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])

                    if pub_date and pub_date < cutoff:
                        continue

                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "source": feed_url,
                        "date": pub_date or datetime.now(),
                    })
                logger.info(f"Fetched {len(feed.entries)} articles from {feed_url}")
            except Exception as e:
                logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")

        return articles

    def scrape_moneycontrol_news(self, symbol: str) -> list:
        """
        Scrape stock-specific news from MoneyControl.
        MoneyControl has per-stock news pages that are very useful.
        """
        articles = []
        try:
            # MoneyControl search
            search_url = f"https://www.moneycontrol.com/news/tags/{symbol.lower()}.html"
            resp = requests.get(search_url, headers=self._headers, timeout=10)
            if resp.status_code != 200:
                return articles

            soup = BeautifulSoup(resp.text, "html.parser")
            news_items = soup.find_all("li", class_="clearfix")[:10]

            for item in news_items:
                title_tag = item.find("h2") or item.find("a")
                if title_tag:
                    articles.append({
                        "title": title_tag.get_text(strip=True),
                        "summary": "",
                        "source": "moneycontrol",
                        "symbol": symbol,
                        "date": datetime.now(),  # Approximate
                    })
        except Exception as e:
            logger.debug(f"MoneyControl scrape failed for {symbol}: {e}")

        return articles

    def scrape_google_news(self, query: str, num_results: int = 10) -> list:
        """
        Fetch news via Google News RSS for a specific query.
        This gives broader coverage beyond our fixed RSS feeds.
        """
        articles = []
        try:
            encoded_query = requests.utils.quote(f"{query} stock India NSE")
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)

            for entry in feed.entries[:num_results]:
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])

                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "source": "google_news",
                    "date": pub_date or datetime.now(),
                })
        except Exception as e:
            logger.error(f"Google News fetch failed for {query}: {e}")

        return articles

    def analyze_sentiment(self, text: str) -> dict:
        """
        Run VADER sentiment analysis on text.
        
        VADER is specifically tuned for social media/news text.
        Returns: {neg, neu, pos, compound}
        compound is -1 (most negative) to +1 (most positive)
        """
        if not text or not text.strip():
            return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}
        
        # Clean text
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = re.sub(r"\s+", " ", text).strip()
        
        return self.analyzer.polarity_scores(text)

    def get_stock_sentiment(self, symbol: str) -> dict:
        """
        Get sentiment score for a specific stock.
        
        PROCESS:
        1. Search RSS feeds for articles mentioning this stock
        2. Scrape stock-specific pages
        3. Analyze each article's sentiment
        4. Average the compound scores
        5. Return normalized score
        """
        all_articles = []

        # Get aliases for this stock
        aliases = self._company_aliases.get(
            symbol, [symbol.lower()]
        )

        # 1. Filter RSS articles that mention this stock
        rss_articles = self.fetch_rss_articles()
        for article in rss_articles:
            text = f"{article['title']} {article['summary']}".lower()
            if any(alias in text for alias in aliases):
                article["symbol"] = symbol
                all_articles.append(article)

        # 2. Google News for this stock
        google_articles = self.scrape_google_news(symbol)
        all_articles.extend(google_articles)

        # 3. MoneyControl stock-specific
        mc_articles = self.scrape_moneycontrol_news(symbol)
        all_articles.extend(mc_articles)

        if not all_articles:
            return {
                "symbol": symbol,
                "sentiment_score": 0,
                "num_articles": 0,
                "avg_compound": 0,
                "positive_pct": 0,
                "negative_pct": 0,
                "articles": [],
            }

        # 4. Analyze each article
        sentiments = []
        for article in all_articles:
            text = f"{article['title']}. {article.get('summary', '')}"
            scores = self.analyze_sentiment(text)
            article["sentiment"] = scores
            sentiments.append(scores["compound"])

        # 5. Calculate aggregate score
        avg_compound = sum(sentiments) / len(sentiments)
        positive_pct = sum(1 for s in sentiments if s > 0.2) / len(sentiments) * 100
        negative_pct = sum(1 for s in sentiments if s < -0.2) / len(sentiments) * 100

        # Normalize to -10 to +10 scale
        sentiment_score = round(avg_compound * 10, 2)

        return {
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "num_articles": len(all_articles),
            "avg_compound": round(avg_compound, 4),
            "positive_pct": round(positive_pct, 1),
            "negative_pct": round(negative_pct, 1),
            "articles": all_articles[:5],  # Top 5 articles for reference
        }

    def get_market_sentiment(self) -> dict:
        """
        Overall market mood from broad financial news.
        Useful for deciding if it's a good time to be fully invested or cautious.
        """
        articles = self.fetch_rss_articles()
        if not articles:
            return {"market_mood": "neutral", "score": 0}

        sentiments = []
        for article in articles:
            text = f"{article['title']}. {article.get('summary', '')}"
            scores = self.analyze_sentiment(text)
            sentiments.append(scores["compound"])

        avg = sum(sentiments) / len(sentiments)

        if avg > 0.15:
            mood = "bullish"
        elif avg > 0.05:
            mood = "mildly_bullish"
        elif avg > -0.05:
            mood = "neutral"
        elif avg > -0.15:
            mood = "mildly_bearish"
        else:
            mood = "bearish"

        return {
            "market_mood": mood,
            "score": round(avg * 10, 2),
            "num_articles": len(articles),
        }

    def get_bulk_sentiment(self, symbols: list) -> pd.DataFrame:
        """Get sentiment for multiple stocks. Returns a DataFrame."""
        results = []
        for symbol in symbols:
            logger.info(f"Analyzing sentiment for {symbol}")
            data = self.get_stock_sentiment(symbol)
            results.append({
                "symbol": data["symbol"],
                "sentiment_score": data["sentiment_score"],
                "num_articles": data["num_articles"],
                "positive_pct": data["positive_pct"],
                "negative_pct": data["negative_pct"],
            })
        return pd.DataFrame(results)
