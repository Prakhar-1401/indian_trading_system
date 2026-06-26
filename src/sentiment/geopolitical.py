"""
Geopolitical Risk Monitor — Track war/conflict/global events impact on markets.

WHAT IT DOES:
- Monitors news RSS feeds for geopolitical events (war, sanctions, conflicts)
- Maps events to affected sectors (defense, oil, banking, etc.)
- Generates risk scores and sector impact predictions

HOW GEOPOLITICS AFFECTS INDIAN MARKETS:
- India-Pakistan tension → Defense stocks UP (HAL, BEL), Market DOWN
- Middle East conflict → Oil UP → ONGC/Oil India UP, Airlines DOWN
- US-China trade war → IT stocks impacted, pharma benefits
- Russia-Ukraine → Oil/gas/fertilizer prices spike
- Global recession fear → Safe havens (gold, FMCG) UP, cyclicals DOWN

SECTOR SENSITIVITY MAP:
  Event Type          | Positive Sectors          | Negative Sectors
  --------------------|---------------------------|------------------
  War/Conflict        | Defense, Oil, Gold        | Airlines, Tourism, Auto
  Sanctions           | Pharma (generics), IT     | Banks (trade finance)
  Oil Price Spike     | ONGC, Oil India, Coal     | Airlines, Paints, FMCG
  Currency Crisis     | IT (dollar earners)       | Importers (electronics)
  Rate Hike (US)      | Banking (NIM expansion)   | Real Estate, NBFC
  Pandemic/Health     | Pharma, Diagnostics       | Hotels, Aviation, Retail
"""
import feedparser
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class GeoEvent:
    """A geopolitical event detected from news."""
    headline: str
    source: str
    date: str
    event_type: str  # 'WAR', 'SANCTIONS', 'OIL_CRISIS', 'TRADE_WAR', 'PANDEMIC', 'POLITICAL'
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    affected_sectors_positive: List[str]
    affected_sectors_negative: List[str]
    risk_score: float  # 0-10 (10 = maximum risk)


# Keywords that indicate geopolitical risk
EVENT_KEYWORDS = {
    "WAR": [
        "war", "military", "strike", "missile", "bomb", "attack", "invasion",
        "troops", "army", "navy", "airforce", "ceasefire", "conflict",
        "nuclear", "escalation", "border tension", "surgical strike",
        "drone strike", "casualties", "warship", "defense alert",
    ],
    "SANCTIONS": [
        "sanctions", "embargo", "ban", "trade restriction", "blacklist",
        "export ban", "import duty", "tariff war", "trade war",
    ],
    "OIL_CRISIS": [
        "oil price", "crude oil", "opec", "oil supply", "petroleum",
        "natural gas", "energy crisis", "fuel price", "oil production cut",
        "brent crude", "wti crude",
    ],
    "TRADE_WAR": [
        "trade war", "tariff", "import duty", "export restriction",
        "china trade", "us-china", "protectionism", "trade deficit",
    ],
    "PANDEMIC": [
        "pandemic", "virus", "outbreak", "lockdown", "covid", "epidemic",
        "quarantine", "health emergency", "WHO alert",
    ],
    "POLITICAL": [
        "election", "government collapse", "coup", "regime change",
        "political crisis", "parliament dissolved", "no confidence",
        "budget", "rbi policy", "rate cut", "rate hike",
    ],
}

# Sector impact mapping
SECTOR_IMPACT = {
    "WAR": {
        "positive": ["Defense (HAL, BEL, BDL)", "Oil (ONGC, OIL)", "Gold (GOLDBEES)"],
        "negative": ["Airlines (INDIGO)", "Tourism (IHCL)", "Auto (MARUTI)", "Banking (broad)"],
    },
    "SANCTIONS": {
        "positive": ["Pharma (SUNPHARMA, DRREDDY)", "IT (TCS, INFY)"],
        "negative": ["Banks (trade finance)", "Importers"],
    },
    "OIL_CRISIS": {
        "positive": ["Oil (ONGC, OIL, COALINDIA)", "Gas (GAIL, IGL)"],
        "negative": ["Airlines (INDIGO)", "Paints (ASIANPAINT)", "FMCG (transport cost)"],
    },
    "TRADE_WAR": {
        "positive": ["IT (TCS, INFY - domestic consumption)", "Pharma (generic demand)"],
        "negative": ["Export-heavy (textiles)", "Metal (TATASTEEL, JSWSTEEL)"],
    },
    "PANDEMIC": {
        "positive": ["Pharma (CIPLA, DRREDDY)", "Diagnostics (METROPOLIS)", "IT (WFH)"],
        "negative": ["Hotels (IHCL)", "Aviation (INDIGO)", "Retail (DMART)"],
    },
    "POLITICAL": {
        "positive": ["Infra (LT, if govt spending)", "Banking (if rate cut)"],
        "negative": ["Market broad (uncertainty)", "NBFC (if rate hike)"],
    },
}

# RSS feeds for geopolitical news
GEO_RSS_FEEDS = [
    "https://www.livemint.com/rss/markets",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://economictimes.indiatimes.com/news/defence/rssfeeds/47139498.cms",
]

# India-specific geopolitical watchlist
INDIA_WATCHLIST = [
    "india pakistan", "india china", "loc", "line of control",
    "ladakh", "galwan", "arunachal", "kashmir", "doklam",
    "indian ocean", "south china sea", "bangladesh border",
]


class GeopoliticalMonitor:
    """
    Monitors geopolitical events and maps them to market impact.
    
    USAGE:
        monitor = GeopoliticalMonitor()
        events = monitor.scan_news()
        report = monitor.get_risk_report()
        monitor.print_report(report)
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_articles(self) -> List[Dict]:
        """Fetch articles from all geopolitical news feeds."""
        articles = []

        for feed_url in GEO_RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Last 20 articles per feed
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "source": feed_url.split("/")[2],
                        "date": entry.get("published", str(datetime.now())),
                    })
            except Exception as e:
                logger.debug(f"Error fetching {feed_url}: {e}")

        logger.info(f"Fetched {len(articles)} articles from {len(GEO_RSS_FEEDS)} feeds")
        return articles

    def classify_event(self, text: str) -> tuple:
        """Classify text into event type and calculate severity."""
        text_lower = text.lower()

        matches = {}
        for event_type, keywords in EVENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                matches[event_type] = score

        if not matches:
            return None, "LOW", 0

        # Best match
        best_type = max(matches, key=matches.get)
        keyword_hits = matches[best_type]

        # Severity based on keyword density and sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        neg_score = sentiment["neg"]

        if keyword_hits >= 3 or neg_score > 0.5:
            severity = "HIGH"
            risk_score = min(8 + keyword_hits, 10)
        elif keyword_hits >= 2 or neg_score > 0.3:
            severity = "MEDIUM"
            risk_score = min(5 + keyword_hits, 8)
        else:
            severity = "LOW"
            risk_score = min(2 + keyword_hits, 5)

        # India-specific boost
        for term in INDIA_WATCHLIST:
            if term in text_lower:
                risk_score = min(risk_score + 2, 10)
                if severity == "LOW":
                    severity = "MEDIUM"
                elif severity == "MEDIUM":
                    severity = "HIGH"
                break

        return best_type, severity, risk_score

    def scan_news(self) -> List[GeoEvent]:
        """Scan all feeds and identify geopolitical events."""
        articles = self.fetch_articles()
        events = []

        for article in articles:
            text = f"{article['title']} {article['summary']}"
            event_type, severity, risk_score = self.classify_event(text)

            if event_type is None:
                continue

            impact = SECTOR_IMPACT.get(event_type, {"positive": [], "negative": []})

            events.append(GeoEvent(
                headline=article["title"][:100],
                source=article["source"],
                date=article["date"][:25],
                event_type=event_type,
                severity=severity,
                affected_sectors_positive=impact["positive"],
                affected_sectors_negative=impact["negative"],
                risk_score=risk_score,
            ))

        # Sort by risk score
        events.sort(key=lambda x: x.risk_score, reverse=True)
        return events

    def get_risk_report(self) -> Dict:
        """Generate a comprehensive risk report."""
        events = self.scan_news()

        if not events:
            return {
                "overall_risk": "LOW",
                "risk_score": 0,
                "events": [],
                "sectors_at_risk": [],
                "sectors_to_buy": [],
                "recommendation": "No significant geopolitical risks detected. Normal trading.",
            }

        # Overall risk = average of top 3 events
        top_scores = [e.risk_score for e in events[:3]]
        avg_risk = sum(top_scores) / len(top_scores)

        if avg_risk >= 7:
            overall_risk = "HIGH"
            recommendation = "⚠️ HIGH RISK — Consider reducing exposure. Move to defensives/cash."
        elif avg_risk >= 4:
            overall_risk = "MEDIUM"
            recommendation = "⚡ MODERATE RISK — Be cautious. Tighten stop-losses."
        else:
            overall_risk = "LOW"
            recommendation = "✅ LOW RISK — Normal trading conditions."

        # Aggregate affected sectors
        sectors_negative = set()
        sectors_positive = set()
        for e in events[:5]:
            for s in e.affected_sectors_negative:
                sectors_negative.add(s)
            for s in e.affected_sectors_positive:
                sectors_positive.add(s)

        # Event type breakdown
        type_counts = {}
        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        return {
            "overall_risk": overall_risk,
            "risk_score": round(avg_risk, 1),
            "total_events": len(events),
            "high_severity_count": sum(1 for e in events if e.severity == "HIGH"),
            "event_types": type_counts,
            "events": events[:10],  # Top 10 by risk
            "sectors_at_risk": list(sectors_negative),
            "sectors_to_buy": list(sectors_positive),
            "recommendation": recommendation,
        }

    @staticmethod
    def print_report(report: Dict):
        """Print formatted geopolitical risk report."""
        print("\n" + "=" * 70)
        print("  GEOPOLITICAL RISK MONITOR")
        print("=" * 70)

        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        emoji = risk_emoji.get(report["overall_risk"], "⚪")

        print(f"\n  {emoji} Overall Risk Level: {report['overall_risk']} ({report['risk_score']}/10)")
        print(f"  {report['recommendation']}")
        print(f"\n  Events Detected: {report['total_events']}")
        print(f"  High Severity: {report.get('high_severity_count', 0)}")

        # Event type breakdown
        if report.get("event_types"):
            print(f"\n  Event Types:")
            for etype, count in report["event_types"].items():
                print(f"    {etype:<15} : {count} events")

        # Top events
        if report["events"]:
            print(f"\n  {'Risk':>4} {'Type':<12} {'Severity':<8} {'Headline':<50}")
            print("  " + "-" * 76)
            for e in report["events"][:8]:
                print(
                    f"  {e.risk_score:>4.0f} {e.event_type:<12} "
                    f"{e.severity:<8} {e.headline[:48]:<50}"
                )

        # Sector impact
        if report["sectors_to_buy"]:
            print(f"\n  📈 Sectors to BUY (benefit from events):")
            for s in report["sectors_to_buy"]:
                print(f"     + {s}")

        if report["sectors_at_risk"]:
            print(f"\n  📉 Sectors at RISK (hurt by events):")
            for s in report["sectors_at_risk"]:
                print(f"     - {s}")

        print("\n" + "=" * 70)
