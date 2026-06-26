"""
Full Analysis Script — Rank → Signals → Detailed Single-Stock Backtest
Runs the top-ranked stock through backtesting with detailed entry/exit logs
and multi-timeframe performance (1W, 1M, 6M, 1Y, 3Y, 5Y).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.helpers import setup_logging, load_config
from src.data.fetcher import DataManager
from src.indicators.technical import compute_momentum_score
from src.ranking.ranker import StockRanker, QualityScorer

logger = setup_logging()


def run_ranking():
    """Step 1: Rank stocks by momentum + quality."""
    print("\n" + "=" * 70)
    print("  STEP 1: STOCK RANKING (Momentum + Quality)")
    print("=" * 70)

    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE",
        "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "HCLTECH",
        "NTPC", "POWERGRID", "COALINDIA", "NESTLEIND", "TECHM",
        "AXISBANK", "KOTAKBANK", "ULTRACEMCO", "ONGC", "JSWSTEEL",
    ]

    ranker = StockRanker()
    rankings = ranker.rank_quick(symbols)

    if not rankings.empty:
        print(f"\n{'Rank':>4} {'Symbol':<15} {'Composite':>10} {'Momentum':>10} {'Quality':>10}")
        print("-" * 55)
        for _, row in rankings.iterrows():
            print(
                f"{int(row.get('rank', 0)):>4} {row['symbol']:<15} "
                f"{row['composite_score']:>10.2f} {row['momentum_score']:>10.1f} "
                f"{row['quality_score']:>10.1f}"
            )
        top_stock = rankings.iloc[0]['symbol']
        top_score = rankings.iloc[0]['composite_score']
        print(f"\n  🏆 TOP RANKED STOCK: {top_stock} (Score: {top_score:.2f})")
        return top_stock, rankings
    else:
        print("  No rankings generated.")
        return None, None


def run_signals(top_stock, rankings):
    """Step 2: Generate signal for top stock."""
    print("\n" + "=" * 70)
    print("  STEP 2: TRADING SIGNAL")
    print("=" * 70)

    if rankings is None or rankings.empty:
        print("  No data.")
        return

    row = rankings.iloc[0]
    score = row['composite_score']

    if score >= 5.0:
        signal = "🟢 BUY"
    elif score <= 3.0:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"

    print(f"\n  {signal} — {top_stock}")
    print(f"  Composite Score: {score:.2f}")
    print(f"  Momentum: {row['momentum_score']:.1f}/10")
    print(f"  Quality: {row['quality_score']:.1f}/10")


def run_detailed_backtest(symbol):
    """Step 3: Detailed backtest on single stock with entries/exits."""
    print("\n" + "=" * 70)
    print(f"  STEP 3: DETAILED BACKTEST — {symbol}")
    print("=" * 70)

    dm = DataManager()
    config = load_config()
    risk_config = config.get("risk", {})

    stop_loss_pct = risk_config.get("stop_loss_pct", 8.0) / 100
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 12.0) / 100

    # Get maximum historical data (5 years)
    print(f"\n  Downloading 5-year data for {symbol}...")
    df = dm.get_stock_data(symbol, period="5y", interval="1d")

    if df.empty:
        print(f"  ERROR: No data for {symbol}")
        return

    # Normalize timezone
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(f"  Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total trading days: {len(df)}")

    # --- Multi-Timeframe Performance ---
    print("\n" + "-" * 70)
    print("  MULTI-TIMEFRAME PERFORMANCE (Buy & Hold)")
    print("-" * 70)

    now = df.index[-1]
    current_price = df['close'].iloc[-1]

    timeframes = {
        "1 Week": 5,
        "1 Month": 21,
        "6 Months": 126,
        "1 Year": 252,
        "3 Years": 756,
        "5 Years": 1260,
    }

    print(f"\n  Current Price: ₹{current_price:,.2f} ({now.date()})")
    print(f"\n  {'Timeframe':<12} {'Start Price':>12} {'Return':>10} {'Abs Gain (₹1L)':>16}")
    print("  " + "-" * 54)

    for label, days in timeframes.items():
        if len(df) > days:
            start_price = df['close'].iloc[-days]
            ret = ((current_price - start_price) / start_price) * 100
            gain_on_1l = 100000 * (current_price / start_price) - 100000
            print(f"  {label:<12} ₹{start_price:>10,.2f} {ret:>+9.2f}% ₹{gain_on_1l:>+13,.0f}")
        else:
            print(f"  {label:<12} {'(insufficient data)':<40}")

    # --- Strategy Backtest with Entries/Exits ---
    print("\n" + "-" * 70)
    print("  STRATEGY BACKTEST (Momentum-based entries with stop-loss)")
    print("-" * 70)
    print(f"  Rules: Buy when momentum > 6/10, Exit on stop-loss ({stop_loss_pct*100:.0f}%)")
    print(f"         or trailing stop ({trailing_stop_pct*100:.0f}%) or momentum < 4/10")

    initial_capital = 1000000  # ₹10 Lakh
    cash = initial_capital
    position = None  # {shares, entry_price, entry_date, peak_price}
    trades = []

    # Use weekly rebalance check
    check_days = list(range(0, len(df), 5))  # Every 5 trading days

    for i in check_days:
        if i < 60:  # Need 60 days for momentum calculation
            continue

        row = df.iloc[i]
        date = df.index[i]
        price = row['close']

        # Calculate momentum score on data up to this point
        hist = df.iloc[:i+1]
        m_score = compute_momentum_score(hist)

        if position is None:
            # Check for BUY signal
            if m_score >= 6.0:
                shares = int(cash * 0.95 / price)  # Use 95% of capital
                if shares > 0:
                    cost = shares * price
                    position = {
                        "shares": shares,
                        "entry_price": price,
                        "entry_date": date,
                        "peak_price": price,
                    }
                    cash -= cost
        else:
            # Update peak
            # Check intermediate days for stop-loss
            for j in range(max(check_days[check_days.index(i) - 1] + 1 if i > check_days[0] else i, i-5), i+1):
                if j < len(df):
                    position["peak_price"] = max(position["peak_price"], df.iloc[j]['close'])

            exit_reason = None

            # Stop-loss check
            if price <= position["entry_price"] * (1 - stop_loss_pct):
                exit_reason = "STOP-LOSS"
            # Trailing stop
            elif price <= position["peak_price"] * (1 - trailing_stop_pct):
                exit_reason = "TRAILING-STOP"
            # Momentum exit
            elif m_score < 4.0:
                exit_reason = "MOMENTUM-EXIT"

            if exit_reason:
                pnl = (price - position["entry_price"]) * position["shares"]
                pnl_pct = ((price / position["entry_price"]) - 1) * 100
                trades.append({
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "exit_date": date,
                    "exit_price": price,
                    "shares": position["shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": exit_reason,
                    "days_held": (date - position["entry_date"]).days,
                })
                cash += price * position["shares"]
                position = None

    # Close any open position at the end
    if position:
        final_price = df['close'].iloc[-1]
        final_date = df.index[-1]
        pnl = (final_price - position["entry_price"]) * position["shares"]
        pnl_pct = ((final_price / position["entry_price"]) - 1) * 100
        trades.append({
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "exit_date": final_date,
            "exit_price": final_price,
            "shares": position["shares"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": "OPEN (still holding)",
            "days_held": (final_date - position["entry_date"]).days,
        })
        cash += final_price * position["shares"]

    # --- Print Trades ---
    print(f"\n  {'#':<3} {'Entry Date':<12} {'Entry ₹':>10} {'Exit Date':<12} {'Exit ₹':>10} {'P&L %':>8} {'Days':>5} {'Reason':<15}")
    print("  " + "-" * 85)

    for idx, t in enumerate(trades, 1):
        print(
            f"  {idx:<3} {t['entry_date'].strftime('%Y-%m-%d'):<12} "
            f"₹{t['entry_price']:>9,.2f} "
            f"{t['exit_date'].strftime('%Y-%m-%d'):<12} "
            f"₹{t['exit_price']:>9,.2f} "
            f"{t['pnl_pct']:>+7.2f}% "
            f"{t['days_held']:>5} "
            f"{t['reason']:<15}"
        )

    # --- Summary ---
    print("\n" + "-" * 70)
    print("  BACKTEST SUMMARY")
    print("-" * 70)

    total_pnl = sum(t['pnl'] for t in trades)
    final_value = cash
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]

    data_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((final_value / initial_capital) ** (1 / max(data_years, 0.01)) - 1) * 100

    print(f"\n  Initial Capital:     ₹{initial_capital:>12,.0f}")
    print(f"  Final Value:         ₹{final_value:>12,.0f}")
    print(f"  Total Return:         {total_return:>+11.2f}%")
    print(f"  CAGR:                 {cagr:>+11.2f}%")
    print(f"  Total Trades:         {len(trades):>11}")
    print(f"  Winning Trades:       {len(winning):>11}")
    print(f"  Losing Trades:        {len(losing):>11}")
    print(f"  Win Rate:             {len(winning)/max(len(trades),1)*100:>10.1f}%")

    if winning:
        print(f"  Avg Win:              {np.mean([t['pnl_pct'] for t in winning]):>+10.2f}%")
    if losing:
        print(f"  Avg Loss:             {np.mean([t['pnl_pct'] for t in losing]):>+10.2f}%")

    # Buy & hold comparison
    bh_return = ((df['close'].iloc[-1] / df['close'].iloc[60]) - 1) * 100
    print(f"\n  Buy & Hold Return:    {bh_return:>+10.2f}% (same period)")
    print(f"  Strategy vs B&H:      {total_return - bh_return:>+10.2f}% {'(outperformed)' if total_return > bh_return else '(underperformed)'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "🔷" * 35)
    print("  INDIAN STOCK MARKET — FULL ANALYSIS")
    print("🔷" * 35)

    # Step 1: Rank
    top_stock, rankings = run_ranking()

    if top_stock is None:
        print("\nFailed to rank stocks. Check your internet connection.")
        sys.exit(1)

    # Step 2: Signal
    run_signals(top_stock, rankings)

    # Step 3: Detailed backtest on top stock
    run_detailed_backtest(top_stock)
