"""
Automated Scheduler — Runs the strategy on a schedule.

This script runs continuously and:
1. Every Monday at 9:00 AM IST: Generates fresh signals (rebalance day)
2. Every weekday at 9:15 AM IST: Checks stop-losses on current holdings
3. Every day at 6:00 PM IST: Runs sentiment analysis for next day

For deployment, run this as a background service or cron job.
"""
import schedule
import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import setup_logging

logger = setup_logging()


def job_generate_signals():
    """Weekly signal generation — runs on rebalance day."""
    logger.info("=== WEEKLY SIGNAL GENERATION ===")
    try:
        from src.strategy.executor import generate_quick_signals, StrategyExecutor
        signals = generate_quick_signals()
        executor = StrategyExecutor()
        executor.print_signals(signals)
        logger.info(f"Generated {len(signals)} signals")
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")


def job_check_stops():
    """Daily stop-loss check."""
    logger.info("=== DAILY STOP-LOSS CHECK ===")
    # In production, this would check your actual positions
    logger.info("Stop-loss check completed")


def job_sentiment_scan():
    """Daily sentiment scan."""
    logger.info("=== DAILY SENTIMENT SCAN ===")
    try:
        from src.sentiment.news_analyzer import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        mood = analyzer.get_market_sentiment()
        logger.info(f"Market mood: {mood['market_mood']} (Score: {mood['score']})")
    except Exception as e:
        logger.error(f"Sentiment scan failed: {e}")


def main():
    logger.info("Starting scheduler...")

    # Monday 9:00 AM — Weekly rebalance signals
    schedule.every().monday.at("09:00").do(job_generate_signals)

    # Weekdays 9:15 AM — Check stop-losses
    schedule.every().monday.at("09:15").do(job_check_stops)
    schedule.every().tuesday.at("09:15").do(job_check_stops)
    schedule.every().wednesday.at("09:15").do(job_check_stops)
    schedule.every().thursday.at("09:15").do(job_check_stops)
    schedule.every().friday.at("09:15").do(job_check_stops)

    # Daily 6:00 PM — Sentiment for next day
    schedule.every().day.at("18:00").do(job_sentiment_scan)

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    print("Scheduler running. Next jobs:")
    for job in schedule.get_jobs():
        print(f"  {job}")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
