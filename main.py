"""
Main entry point — Run the trading system.

USAGE:
    python main.py rank          # Quick rank top stocks
    python main.py signals       # Generate trading signals
    python main.py backtest      # Run backtest
    python main.py sentiment     # Check market sentiment
    python main.py pairs         # Pairs trading (stat arb) signals
    python main.py geo           # Geopolitical risk monitor
    python main.py insider       # Insider/institutional flow tracker
    python main.py risk          # Dynamic ATR risk management
    python main.py telegram      # Test Telegram bot connection
    python main.py ml            # ML ensemble predictions
    python main.py regime        # Market regime detection
    python main.py options       # Options chain analysis
    python main.py events        # Event-driven trading scanner
    python main.py kelly         # Kelly Criterion position sizing
    python main.py premarket     # Morning pre-market report
    python main.py paper         # Paper trading portfolio
    python main.py montecarlo    # Monte Carlo simulation
    python main.py wfo           # Walk-forward optimization
    python main.py factors       # Factor attribution analysis
    python main.py sectors       # Sector heatmap & correlation
    python main.py full          # Run EVERYTHING (daily report)
    python main.py decide        # Decision engine (TAKE/WATCH/SKIP)
    python main.py statval       # Statistical validation (IC, Fama-MacBeth, t-stats)
    python main.py optimize hrp  # Portfolio optimization (max_sharpe/min_variance/risk_parity/hrp/black_litterman)
    python main.py impact        # Market impact + Almgren-Chriss execution
    python main.py dashboard     # Launch Streamlit dashboard
    python main.py terminal      # Launch Bloomberg-style terminal
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import setup_logging, load_config

logger = setup_logging()


def cmd_rank():
    """Quick rank top stocks by momentum + quality."""
    from src.ranking.ranker import StockRanker

    print("\n📊 Quick Stock Ranking (Momentum + Quality)")
    print("=" * 50)

    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
        "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "HCLTECH",
        "TATAMOTORS", "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND",
        "HCLTECH", "TECHM", "AXISBANK", "KOTAKBANK", "ULTRACEMCO",
    ]

    ranker = StockRanker()
    rankings = ranker.rank_quick(symbols)

    if not rankings.empty:
        print(f"\n{'Rank':>4} {'Symbol':<15} {'Composite':>10} {'Momentum':>10} {'Quality':>10} {'Sector':<20}")
        print("-" * 75)
        for _, row in rankings.iterrows():
            print(
                f"{int(row.get('rank', 0)):>4} {row['symbol']:<15} "
                f"{row['composite_score']:>10.2f} {row['momentum_score']:>10.1f} "
                f"{row['quality_score']:>10.1f} {str(row.get('sector', 'N/A')):<20}"
            )


def cmd_signals():
    """Generate trading signals."""
    from src.strategy.executor import StrategyExecutor, generate_quick_signals

    print("\n🎯 Generating Trading Signals...")
    signals = generate_quick_signals()

    if signals:
        executor = StrategyExecutor()
        executor.print_signals(signals)
    else:
        print("No signals generated.")


def cmd_backtest():
    """Run strategy backtest."""
    from src.backtest.engine import BacktestEngine

    print("\n🔄 Running Backtest...")
    engine = BacktestEngine()
    results = engine.run()
    engine.print_report(results)


def cmd_sentiment():
    """Check market and stock sentiment."""
    from src.sentiment.news_analyzer import NewsSentimentAnalyzer

    analyzer = NewsSentimentAnalyzer()

    print("\n📰 Market Sentiment Analysis")
    print("=" * 40)

    # Overall market
    mood = analyzer.get_market_sentiment()
    print(f"\nMarket Mood: {mood['market_mood'].upper()}")
    print(f"Score: {mood['score']}/10")
    print(f"Articles analyzed: {mood['num_articles']}")

    # Top stocks sentiment
    print("\n\nStock-Specific Sentiment:")
    print("-" * 40)
    top_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    for symbol in top_stocks:
        result = analyzer.get_stock_sentiment(symbol)
        print(f"{symbol:>12}: Score={result['sentiment_score']:>6.1f}, Articles={result['num_articles']}")


def cmd_pairs():
    """Run pairs trading analysis."""
    from src.strategy.pairs_trading import PairsTradingEngine

    print("\n⚖️ Pairs Trading — Statistical Arbitrage")
    print("=" * 50)

    engine = PairsTradingEngine()
    pairs = engine.find_pairs()
    signals = engine.generate_signals()

    engine.print_report(pairs, signals)

    # Backtest the best pair
    if pairs:
        best = pairs[0]
        print(f"\n\n  Backtesting best pair: {best.stock_a}/{best.stock_b}...")
        results = engine.backtest_pair(best.stock_a, best.stock_b)
        engine.print_backtest(results)


def cmd_geo():
    """Run geopolitical risk monitor."""
    from src.sentiment.geopolitical import GeopoliticalMonitor

    print("\n🌍 Geopolitical Risk Monitor")
    print("=" * 50)

    monitor = GeopoliticalMonitor()
    report = monitor.get_risk_report()
    monitor.print_report(report)


def cmd_insider():
    """Run insider/institutional flow tracker."""
    from src.smart_money.insider_tracker import InsiderTradeTracker

    print("\n💰 Insider & Institutional Flow Tracker")
    print("=" * 50)

    tracker = InsiderTradeTracker()
    signals = tracker.generate_signals()
    tracker.print_report(signals)


def cmd_risk():
    """Show dynamic ATR-based risk management."""
    from src.strategy.risk_manager import DynamicRiskManager

    symbols = [
        "COALINDIA", "NTPC", "POWERGRID", "SUNPHARMA", "BHARTIARTL",
        "BAJFINANCE", "SBIN", "TCS", "RELIANCE", "HDFCBANK",
    ]

    rm = DynamicRiskManager(capital=1000000)
    rm.print_risk_report(symbols)


def cmd_telegram():
    """Test Telegram bot connection."""
    from src.utils.telegram_alerts import TelegramAlerts

    print("\n📡 Testing Telegram Connection...")
    bot = TelegramAlerts()
    bot.test_connection()


def cmd_full():
    """Run the full daily analysis (all modules)."""
    print("\n" + "🟦" * 35)
    print("  FULL DAILY MARKET ANALYSIS")
    print("🟦" * 35)

    cmd_rank()
    print()
    cmd_signals()
    print()
    cmd_pairs()
    print()
    cmd_geo()
    print()
    cmd_insider()
    print()
    cmd_risk()

    print("\n" + "=" * 70)
    print("  ✅ DAILY ANALYSIS COMPLETE")
    print("=" * 70)


def cmd_ml():
    """Run ML ensemble predictions."""
    from src.strategy.ml_ensemble import MLEnsembleModel

    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN",
        "BHARTIARTL", "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE",
    ]

    model = MLEnsembleModel()
    signals = model.predict_batch(symbols)
    model.print_report(signals)


def cmd_regime():
    """Detect current market regime."""
    from src.strategy.regime_detector import MarketRegimeDetector

    detector = MarketRegimeDetector()
    regime = detector.detect_regime()
    detector.print_report(regime)


def cmd_options():
    """Options chain analysis."""
    from src.strategy.options_analyzer import OptionsChainAnalyzer

    analyzer = OptionsChainAnalyzer()
    results = analyzer.analyze_batch(["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK"])
    analyzer.print_report(results)


def cmd_events():
    """Event-driven trading scanner."""
    from src.strategy.event_driven import EventDrivenTrader

    trader = EventDrivenTrader()
    events = trader.scan_events()
    signals = trader.generate_signals(events)
    trader.print_report(events, signals)


def cmd_kelly():
    """Kelly Criterion position sizing."""
    from src.strategy.kelly_sizing import KellyPositionSizer

    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN",
        "BHARTIARTL", "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE",
    ]

    sizer = KellyPositionSizer(capital=1000000)
    sizer.print_report(symbols)


def cmd_premarket():
    """Generate morning pre-market report."""
    from src.strategy.premarket_report import PreMarketReport

    report = PreMarketReport()
    report.generate()


def cmd_paper():
    """Paper trading commands."""
    from src.strategy.paper_trader import PaperTrader

    trader = PaperTrader(capital=1000000)

    if len(sys.argv) < 3 or sys.argv[2] == "show":
        trader.show_portfolio()
    elif sys.argv[2] == "buy" and len(sys.argv) >= 4:
        symbol = sys.argv[3].upper()
        qty = int(sys.argv[4]) if len(sys.argv) >= 5 else None
        trader.buy(symbol, quantity=qty)
    elif sys.argv[2] == "sell" and len(sys.argv) >= 4:
        symbol = sys.argv[3].upper()
        trader.sell(symbol)
    elif sys.argv[2] == "journal":
        trader.show_trade_journal()
    elif sys.argv[2] == "reset":
        trader.reset()
    else:
        print("  Usage:")
        print("    python main.py paper              # Show portfolio")
        print("    python main.py paper buy SYMBOL   # Buy stock")
        print("    python main.py paper buy SYMBOL 50 # Buy 50 shares")
        print("    python main.py paper sell SYMBOL  # Sell position")
        print("    python main.py paper journal      # Trade history")
        print("    python main.py paper reset        # Start fresh")


def cmd_montecarlo():
    """Run Monte Carlo simulation."""
    from src.backtest.monte_carlo import MonteCarloSimulator

    sim = MonteCarloSimulator()
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
               "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC"]
    weights = [0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.09, 0.08]

    print("\n  Running 10,000 Monte Carlo simulations (1-year horizon)...")
    result = sim.simulate_portfolio(symbols, weights, days=252, num_sims=10000)
    sim.print_report(result, title="10-STOCK PORTFOLIO")


def cmd_wfo():
    """Run walk-forward optimization."""
    from src.backtest.walk_forward import WalkForwardOptimizer

    symbol = sys.argv[2].upper() if len(sys.argv) >= 3 else "RELIANCE"
    wfo = WalkForwardOptimizer()
    summary = wfo.optimize(symbol)
    wfo.print_report(summary)


def cmd_factors():
    """Factor attribution analysis."""
    from src.backtest.factor_attribution import FactorAttributionEngine

    symbols = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
               "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC"]
    weights = [0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.09, 0.08]

    engine = FactorAttributionEngine()
    result = engine.analyze_portfolio(symbols, weights)
    engine.print_report(result)


def cmd_sectors():
    """Sector heatmap and correlation matrix."""
    from src.backtest.sector_analysis import SectorAnalyzer

    analyzer = SectorAnalyzer()
    analyzer.print_sector_heatmap()
    analyzer.print_correlation_matrix()


def cmd_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    print("\n🚀 Launching Dashboard...")
    print("Open http://localhost:8501 in your browser")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])


def cmd_terminal():
    """Launch Bloomberg-style terminal."""
    import subprocess
    print("\n⚡ Launching Trading Terminal...")
    print("Open http://localhost:8501 in your browser")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/terminal.py"])


def cmd_decide():
    """Run unified decision engine."""
    from src.strategy.decision_engine import DecisionEngine

    print("\n🧠 Running Decision Engine...")
    engine = DecisionEngine()
    decisions = engine.analyze_watchlist()
    engine.print_report(decisions)


def cmd_statval():
    """Statistical validation: IC, Fama-MacBeth, Newey-West t-stats, FDR, bootstrap Sharpe."""
    from src.research.statistical_validation import StatisticalValidator

    print("\n🔬 Running Statistical Validation (this downloads a universe, give it a minute)...")
    validator = StatisticalValidator()
    report = validator.run()
    validator.print_report(report)


def cmd_optimize():
    """Portfolio optimization: max_sharpe / min_variance / risk_parity / hrp / black_litterman."""
    from src.portfolio.optimizer import PortfolioOptimizer

    method = sys.argv[2].lower() if len(sys.argv) >= 3 else "hrp"
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
               "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC"]

    print(f"\n📐 Optimizing portfolio with method='{method}'...")
    opt = PortfolioOptimizer().load(symbols)
    if method in ("all", "compare"):
        for m in ["max_sharpe", "min_variance", "risk_parity", "hrp", "black_litterman"]:
            opt.print_report(opt.optimize(m))
    else:
        opt.print_report(opt.optimize(method))


def cmd_impact():
    """Market impact (square-root model) + Almgren-Chriss optimal execution."""
    from src.execution.market_impact import ExecutionAnalyzer

    symbol = sys.argv[2].upper() if len(sys.argv) >= 3 else "RELIANCE"
    order_value = float(sys.argv[3]) if len(sys.argv) >= 4 else 10_000_000

    print(f"\n💧 Estimating execution cost for {symbol} (Rs {order_value:,.0f})...")
    ex = ExecutionAnalyzer()
    est = ex.estimate_impact(symbol, order_value=order_value)
    ex.print_impact(est)
    if est.order_shares > 0:
        ac = ex.almgren_chriss(symbol, total_shares=est.order_shares, n_slices=10)
        ex.print_almgren(ac)


def cmd_signalstudy():
    """Screen a library of anomaly signals across a broad universe; rank by net-of-cost Sharpe."""
    from src.research.signal_study import SignalStudy

    print("\n\U0001F52C Running Signal Research Study (downloads ~50 names, give it a minute)...")
    study = SignalStudy()
    study.print_leaderboard(study.run())


def cmd_reversion():
    """Turnover study: can the mean-reversion edge survive costs after reducing turnover?"""
    from src.research.reversion_study import ReversionStudy

    print("\n\U0001F501 Running Reversion Turnover Study (rebalance x quantile sweep)...")
    study = ReversionStudy()
    _, configs, _ = study.run()
    study.print_report(configs)


def cmd_benchmark():
    """Alpha/beta of the best reversion strategy vs the Nifty 50 (^NSEI)."""
    from src.research.reversion_study import ReversionStudy
    from src.research.benchmark import BenchmarkAnalyzer

    print("\n\U0001F4C8 Building best reversion strategy, then measuring alpha vs Nifty 50...")
    study = ReversionStudy()
    best, _, returns = study.run()
    if best is None or returns is None:
        print("No tradeable configuration produced returns to benchmark.")
        return
    print(f"  Best config: {best.signal}, rebalance {best.rebalance_days}d, "
          f"top/bottom {best.quantile:.0%} (net Sharpe {best.net_sharpe:.2f})")
    BenchmarkAnalyzer().print_report(BenchmarkAnalyzer().analyze(returns))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()
    commands = {
        "rank": cmd_rank,
        "signals": cmd_signals,
        "backtest": cmd_backtest,
        "sentiment": cmd_sentiment,
        "pairs": cmd_pairs,
        "geo": cmd_geo,
        "insider": cmd_insider,
        "risk": cmd_risk,
        "telegram": cmd_telegram,
        "ml": cmd_ml,
        "regime": cmd_regime,
        "options": cmd_options,
        "events": cmd_events,
        "kelly": cmd_kelly,
        "premarket": cmd_premarket,
        "paper": cmd_paper,
        "montecarlo": cmd_montecarlo,
        "wfo": cmd_wfo,
        "factors": cmd_factors,
        "sectors": cmd_sectors,
        "full": cmd_full,
        "decide": cmd_decide,
        "statval": cmd_statval,
        "signalstudy": cmd_signalstudy,
        "reversion": cmd_reversion,
        "benchmark": cmd_benchmark,
        "optimize": cmd_optimize,
        "impact": cmd_impact,
        "dashboard": cmd_dashboard,
        "terminal": cmd_terminal,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(commands.keys())}")


if __name__ == "__main__":
    main()
