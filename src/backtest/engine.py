"""
Backtesting Engine — Test Your Strategy BEFORE Risking Real Money.

WHAT IS BACKTESTING?
====================
You take your strategy rules, apply them to HISTORICAL data, and see
what the results would have been. Like a flight simulator for trading.

WHY IS IT CRITICAL?
- A strategy that SOUNDS good might lose money in practice
- Backtesting reveals: win rate, max drawdown, Sharpe ratio, etc.
- It helps you tune the factor weights and parameters

HOW OUR BACKTEST WORKS:
1. Start from a date (e.g., Jan 2020) with ₹10 Lakh capital
2. Every week (Monday), re-rank all stocks using our composite score
3. Buy the top 15, sell any that dropped out of top 15
4. Apply position sizing (max 7% per stock)
5. Apply stop-loss (sell if stock drops 8% from entry)
6. Measure total return, Sharpe ratio, max drawdown vs Nifty 50

KEY METRICS TO EVALUATE:
- Total Return: How much money did we make?
- CAGR: Annualized return
- Sharpe Ratio: Risk-adjusted return (>1.5 is good, >2 is great)
- Max Drawdown: Largest peak-to-trough decline (lower is better)
- Win Rate: % of trades that were profitable
- vs Benchmark: Did we beat the Nifty 50?

PLATFORMS FOR BACKTESTING:
- This module (custom Python) ← We're building this
- Backtrader (Python library) ← We'll integrate this too
- Zerodha Streak (web-based, simpler but less flexible)
- TradingView Pine Script (good for technical-only strategies)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
from dataclasses import dataclass, field

from src.utils.helpers import load_config
from src.data.fetcher import DataManager
from src.indicators.technical import compute_momentum_score


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    exit_date: datetime = None
    exit_price: float = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""  # 'rebalance', 'stop_loss', 'trailing_stop'


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""
    date: datetime
    cash: float
    holdings: dict  # {symbol: {shares, entry_price, current_price, value}}
    total_value: float
    daily_return: float = 0.0


class BacktestEngine:
    """
    Custom backtesting engine for our multi-factor strategy.
    
    USAGE:
        engine = BacktestEngine()
        results = engine.run()
        engine.print_report(results)
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        bt_config = self.config.get("backtest", {})
        risk_config = self.config.get("risk", {})
        portfolio_config = self.config.get("portfolio", {})

        self.start_date = bt_config.get("start_date", "2022-01-01")
        self.end_date = bt_config.get("end_date", "2025-12-31")
        self.initial_capital = bt_config.get("initial_capital", 1000000)
        self.commission_pct = bt_config.get("commission_pct", 0.05) / 100
        self.slippage_pct = bt_config.get("slippage_pct", 0.1) / 100

        self.stop_loss_pct = risk_config.get("stop_loss_pct", 8.0) / 100
        self.trailing_stop_pct = risk_config.get("trailing_stop_pct", 12.0) / 100
        self.max_position_pct = portfolio_config.get("max_position_pct", 7.0) / 100
        self.max_stocks = portfolio_config.get("max_stocks", 15)
        self.cash_reserve_pct = portfolio_config.get("cash_reserve_pct", 5.0) / 100

        self.dm = DataManager()

    def run(
        self, symbols: list = None, rebalance_freq: str = "weekly"
    ) -> dict:
        """
        Run the backtest.
        
        Args:
            symbols: List of stock symbols to consider
            rebalance_freq: 'weekly' or 'monthly'
        
        Returns:
            dict with trades, portfolio history, and performance metrics
        """
        if symbols is None:
            # Use a smaller set for backtesting speed
            symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
                "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
                "NESTLEIND", "TECHM", "POWERGRID", "NTPC", "TATAMOTORS",
                "ONGC", "JSWSTEEL", "TATASTEEL", "COALINDIA", "CIPLA",
                "DRREDDY", "EICHERMOT", "HEROMOTOCO", "BRITANNIA", "HINDALCO",
                "APOLLOHOSP", "TATACONSUM", "DABUR", "PIDILITIND", "HAVELLS",
            ]

        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Capital: ₹{self.initial_capital:,.0f}, Stocks: {len(symbols)}")

        # 1. Download ALL historical data upfront
        logger.info("Downloading historical data...")
        all_data = {}
        for symbol in symbols:
            df = self.dm.get_stock_data(
                symbol, start=self.start_date, end=self.end_date
            )
            if not df.empty and len(df) > 50:
                # Normalize timezone-aware index to tz-naive for comparisons
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                all_data[symbol] = df

        logger.info(f"Got data for {len(all_data)}/{len(symbols)} stocks")

        if not all_data:
            logger.error("No data available for backtesting!")
            return {}

        # 1b. PRE-COMPUTE momentum scores for all stocks on all dates
        # This avoids recalculating on every rebalance (the main bottleneck)
        logger.info("Pre-computing momentum scores (vectorized)...")
        precomputed_scores = {}
        for symbol, df in all_data.items():
            try:
                close = df['close']
                if len(close) < 200:
                    precomputed_scores[symbol] = pd.Series(0.0, index=df.index)
                    continue

                # RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.inf)
                rsi = 100 - (100 / (1 + rs))

                # MACD
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal

                # Moving averages
                sma50 = close.rolling(50).mean()
                sma200 = close.rolling(200).mean()

                # Returns
                ret_3m = close.pct_change(63) * 100
                ret_6m = close.pct_change(126) * 100

                # Score computation (vectorized)
                score = pd.Series(0.0, index=df.index)
                score += ((rsi >= 40) & (rsi <= 65)).astype(float) * 3
                score += (((rsi >= 30) & (rsi < 40)) | ((rsi > 65) & (rsi <= 70))).astype(float) * 1
                score += (macd > signal).astype(float) * 1
                score += (hist > hist.shift(1)).astype(float) * 1
                score += (close > sma200).astype(float) * 1
                score += (close > sma50).astype(float) * 1
                score += (sma50 > sma200).astype(float) * 1
                score += (ret_3m > 15).astype(float) * 2
                score += ((ret_3m > 5) & (ret_3m <= 15)).astype(float) * 1
                score += (ret_6m > 20).astype(float) * 1

                # Normalize to 0-10
                precomputed_scores[symbol] = (score / 12 * 10).clip(0, 10)
            except Exception as e:
                precomputed_scores[symbol] = pd.Series(0.0, index=df.index)
                logger.debug(f"Score precompute failed for {symbol}: {e}")

        logger.info("Momentum scores pre-computed. Starting simulation...")

        # 2. Create a combined date index
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.date if hasattr(df.index, 'date') else df.index)
        trading_days = sorted(all_dates)

        # 3. Determine rebalance dates
        if rebalance_freq == "weekly":
            rebalance_dates = self._get_weekly_dates(trading_days)
        else:
            rebalance_dates = self._get_monthly_dates(trading_days)

        # 4. Run the simulation
        cash = self.initial_capital
        holdings = {}  # {symbol: {shares, entry_price, peak_price}}
        trades = []
        portfolio_history = []

        for date in trading_days:
            date_str = str(date)

            # Get current prices
            prices = {}
            for symbol, df in all_data.items():
                try:
                    if date in df.index:
                        prices[symbol] = df.loc[date, "close"]
                    elif hasattr(df.index, 'date'):
                        mask = df.index.date == date if hasattr(date, 'month') else df.index == date
                        matching = df[mask]
                        if not matching.empty:
                            prices[symbol] = matching["close"].iloc[0]
                except (KeyError, IndexError):
                    continue

            # --- Check Stop Losses ---
            symbols_to_remove = []
            for symbol, pos in holdings.items():
                if symbol not in prices:
                    continue
                current_price = prices[symbol]

                # Update peak price for trailing stop
                pos["peak_price"] = max(pos["peak_price"], current_price)

                # Fixed stop-loss
                if current_price <= pos["entry_price"] * (1 - self.stop_loss_pct):
                    symbols_to_remove.append((symbol, "stop_loss"))

                # Trailing stop-loss
                elif current_price <= pos["peak_price"] * (1 - self.trailing_stop_pct):
                    symbols_to_remove.append((symbol, "trailing_stop"))

            for symbol, reason in symbols_to_remove:
                if symbol in holdings and symbol in prices:
                    pos = holdings[symbol]
                    exit_price = prices[symbol] * (1 - self.slippage_pct)
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    commission = exit_price * pos["shares"] * self.commission_pct

                    trades.append(Trade(
                        symbol=symbol,
                        entry_date=pos["entry_date"],
                        entry_price=pos["entry_price"],
                        shares=pos["shares"],
                        exit_date=date,
                        exit_price=exit_price,
                        pnl=pnl - commission,
                        pnl_pct=((exit_price / pos["entry_price"]) - 1) * 100,
                        exit_reason=reason,
                    ))
                    cash += exit_price * pos["shares"] - commission
                    del holdings[symbol]

            # --- Rebalance ---
            if date in rebalance_dates:
                # Use pre-computed momentum scores (O(1) lookup per stock)
                rankings = []
                for symbol in all_data.keys():
                    if symbol not in prices:
                        continue
                    scores_series = precomputed_scores.get(symbol)
                    if scores_series is None:
                        continue
                    # Find the score for this date (or nearest prior date)
                    try:
                        ts = pd.Timestamp(date)
                        mask = scores_series.index <= ts
                        if mask.any():
                            m_score = scores_series[mask].iloc[-1]
                        else:
                            m_score = 0
                    except Exception:
                        m_score = 0
                    rankings.append({"symbol": symbol, "score": m_score})

                rankings.sort(key=lambda x: x["score"], reverse=True)
                target_symbols = [r["symbol"] for r in rankings[:self.max_stocks]]

                # Sell stocks no longer in top N
                for symbol in list(holdings.keys()):
                    if symbol not in target_symbols and symbol in prices:
                        pos = holdings[symbol]
                        exit_price = prices[symbol] * (1 - self.slippage_pct)
                        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                        commission = exit_price * pos["shares"] * self.commission_pct

                        trades.append(Trade(
                            symbol=symbol,
                            entry_date=pos["entry_date"],
                            entry_price=pos["entry_price"],
                            shares=pos["shares"],
                            exit_date=date,
                            exit_price=exit_price,
                            pnl=pnl - commission,
                            pnl_pct=((exit_price / pos["entry_price"]) - 1) * 100,
                            exit_reason="rebalance",
                        ))
                        cash += exit_price * pos["shares"] - commission
                        del holdings[symbol]

                # Buy new stocks
                investable = cash * (1 - self.cash_reserve_pct)
                per_stock = investable / max(
                    self.max_stocks - len(holdings), 1
                )
                max_per_stock = (cash + sum(
                    prices.get(s, 0) * h["shares"]
                    for s, h in holdings.items()
                )) * self.max_position_pct

                per_stock = min(per_stock, max_per_stock)

                for symbol in target_symbols:
                    if symbol in holdings or symbol not in prices:
                        continue
                    if cash < per_stock * 0.5:
                        break

                    price = prices[symbol] * (1 + self.slippage_pct)
                    shares = int(per_stock / price)
                    if shares <= 0:
                        continue

                    cost = price * shares
                    commission = cost * self.commission_pct

                    if cost + commission > cash:
                        continue

                    holdings[symbol] = {
                        "shares": shares,
                        "entry_price": price,
                        "entry_date": date,
                        "peak_price": price,
                    }
                    cash -= cost + commission

            # --- Record portfolio state ---
            holdings_value = sum(
                prices.get(s, pos["entry_price"]) * pos["shares"]
                for s, pos in holdings.items()
            )
            total_value = cash + holdings_value

            portfolio_history.append({
                "date": date,
                "cash": cash,
                "holdings_value": holdings_value,
                "total_value": total_value,
                "num_holdings": len(holdings),
            })

        # 5. Calculate metrics
        results = self._calculate_metrics(portfolio_history, trades)
        results["trades"] = trades
        results["portfolio_history"] = pd.DataFrame(portfolio_history)

        return results

    def _get_weekly_dates(self, dates: list) -> set:
        """Get Monday dates (or first trading day of week)."""
        weekly = set()
        current_week = None
        for d in dates:
            week = d.isocalendar()[1] if hasattr(d, 'isocalendar') else None
            if week != current_week:
                weekly.add(d)
                current_week = week
        return weekly

    def _get_monthly_dates(self, dates: list) -> set:
        """Get first trading day of each month."""
        monthly = set()
        current_month = None
        for d in dates:
            month = d.month if hasattr(d, 'month') else None
            if month != current_month:
                monthly.add(d)
                current_month = month
        return monthly

    def _calculate_metrics(self, history: list, trades: list) -> dict:
        """
        Calculate performance metrics.
        
        WHAT EACH METRIC MEANS:
        - CAGR: Compound Annual Growth Rate — your annualized return
        - Sharpe Ratio: Return per unit of risk. >1 = good, >2 = great
        - Max Drawdown: Worst peak-to-trough decline. -20% means at worst
          your ₹10L became ₹8L before recovering
        - Win Rate: % of trades that made money
        - Profit Factor: Gross profits / Gross losses. >1.5 is good
        """
        if not history:
            return {}

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        initial = df["total_value"].iloc[0]
        final = df["total_value"].iloc[-1]
        total_return = (final - initial) / initial * 100

        # CAGR
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        cagr = ((final / initial) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0

        # Daily returns for Sharpe
        df["daily_return"] = df["total_value"].pct_change()
        sharpe = (
            df["daily_return"].mean() / df["daily_return"].std() * np.sqrt(252)
            if df["daily_return"].std() > 0 else 0
        )

        # Max Drawdown
        cummax = df["total_value"].cummax()
        drawdown = (df["total_value"] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning_trades) / max(len(trades), 1) * 100

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / max(gross_loss, 1)

        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

        return {
            "initial_capital": initial,
            "final_value": round(final, 2),
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }

    @staticmethod
    def print_report(results: dict):
        """Print a formatted backtest report."""
        if not results:
            print("No results to display.")
            return

        print("\n" + "=" * 60)
        print("         BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Initial Capital:  ₹{results['initial_capital']:>12,.0f}")
        print(f"  Final Value:      ₹{results['final_value']:>12,.0f}")
        print(f"  Total Return:      {results['total_return_pct']:>11.1f}%")
        print(f"  CAGR:              {results['cagr_pct']:>11.1f}%")
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>11.2f}")
        print(f"  Max Drawdown:      {results['max_drawdown_pct']:>11.1f}%")
        print("-" * 60)
        print(f"  Total Trades:      {results['total_trades']:>11}")
        print(f"  Win Rate:          {results['win_rate_pct']:>11.1f}%")
        print(f"  Profit Factor:     {results['profit_factor']:>11.2f}")
        print(f"  Avg Win:           {results['avg_win_pct']:>11.2f}%")
        print(f"  Avg Loss:          {results['avg_loss_pct']:>11.2f}%")
        print(f"  Gross Profit:     ₹{results['gross_profit']:>12,.0f}")
        print(f"  Gross Loss:       ₹{results['gross_loss']:>12,.0f}")
        print("=" * 60)

        # Interpretation
        print("\n📊 INTERPRETATION:")
        if results["sharpe_ratio"] > 2:
            print("  ✅ Excellent risk-adjusted returns (Sharpe > 2)")
        elif results["sharpe_ratio"] > 1:
            print("  ✅ Good risk-adjusted returns (Sharpe > 1)")
        else:
            print("  ⚠️  Poor risk-adjusted returns (Sharpe < 1)")

        if results["max_drawdown_pct"] > -15:
            print("  ✅ Acceptable drawdown (< 15%)")
        elif results["max_drawdown_pct"] > -25:
            print("  ⚠️  Moderate drawdown (15-25%)")
        else:
            print("  ❌ High drawdown (> 25%) — consider tightening stops")

        if results["win_rate_pct"] > 55:
            print("  ✅ Good win rate (> 55%)")
        else:
            print("  ℹ️  Win rate < 55% — OK if avg win >> avg loss")
