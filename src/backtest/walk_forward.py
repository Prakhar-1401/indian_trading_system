"""
Walk-Forward Optimization — The ONLY valid way to backtest.

WHY REGULAR BACKTESTS LIE:
============================
If you optimize parameters on 2020-2025 data and then test on the
SAME data, of course it looks great. That's OVERFITTING.

WALK-FORWARD VALIDATION:
=========================
1. Train on Jan 2020 - Dec 2021 → Test on Jan-Mar 2022
2. Train on Jan 2020 - Mar 2022 → Test on Apr-Jun 2022
3. Train on Jan 2020 - Jun 2022 → Test on Jul-Sep 2022
... and so on.

Each test period uses ONLY past data for optimization.
This is how you'd actually trade in real-time.

WHAT WE OPTIMIZE:
- RSI thresholds (buy at 25? 30? 35?)
- MACD parameters (12/26/9 or 8/21/5?)
- ATR multiplier for stops (1.5x? 2x? 2.5x?)
- Position sizing (Kelly fraction: 25%? 50%?)
- Holding period (5 days? 10 days? 20 days?)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger
from itertools import product

from src.data.fetcher import DataManager


@dataclass
class WalkForwardResult:
    """Result of one walk-forward window."""
    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict
    in_sample_return: float  # Training period performance
    out_sample_return: float  # Testing period performance (what matters)
    num_trades: int
    win_rate: float


@dataclass
class WFOSummary:
    """Summary of entire walk-forward optimization."""
    symbol: str
    total_windows: int
    avg_oos_return: float  # Average out-of-sample return
    total_oos_return: float  # Cumulative OOS return
    avg_win_rate: float
    best_params_frequency: Dict  # Most commonly selected parameters
    degradation: float  # How much worse OOS vs IS (robustness)
    windows: List[WalkForwardResult]


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine.
    
    USAGE:
        wfo = WalkForwardOptimizer()
        summary = wfo.optimize("RELIANCE")
        wfo.print_report(summary)
    """

    def __init__(self):
        self.dm = DataManager()
        
        # Parameter search space
        self.param_grid = {
            'rsi_buy': [25, 30, 35],
            'rsi_sell': [65, 70, 75],
            'atr_stop': [1.5, 2.0, 2.5],
            'holding_days': [5, 10, 15],
        }

    def _backtest_params(self, df: pd.DataFrame, params: Dict) -> Dict:
        """
        Run a simple backtest with given parameters.
        Returns: {'return': float, 'num_trades': int, 'win_rate': float}
        """
        close = df['close'].values
        n = len(close)

        if n < 30:
            return {'return': 0, 'num_trades': 0, 'win_rate': 0}

        # Calculate indicators
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = (100 - (100 / (1 + rs))).values

        # ATR
        high = df['high'].values
        low = df['low'].values
        tr = np.maximum(high - low,
                       np.maximum(abs(high - np.roll(close, 1)),
                                  abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().values

        # Simulate trades
        trades = []
        position = None
        holding = 0

        for i in range(20, n):
            if np.isnan(rsi[i]) or np.isnan(atr[i]):
                continue

            if position is None:
                # Entry: RSI below buy threshold
                if rsi[i] < params['rsi_buy']:
                    position = {
                        'entry_price': close[i],
                        'stop': close[i] - params['atr_stop'] * atr[i],
                        'entry_idx': i,
                    }
                    holding = 0
            else:
                holding += 1
                # Exit conditions
                exit_reason = None

                if close[i] <= position['stop']:
                    exit_reason = "stop"
                elif rsi[i] > params['rsi_sell']:
                    exit_reason = "rsi_exit"
                elif holding >= params['holding_days']:
                    exit_reason = "time_exit"

                if exit_reason:
                    pnl = (close[i] / position['entry_price'] - 1) * 100
                    trades.append(pnl)
                    position = None

        if not trades:
            return {'return': 0, 'num_trades': 0, 'win_rate': 0}

        total_return = sum(trades)
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades)

        return {
            'return': round(total_return, 2),
            'num_trades': len(trades),
            'win_rate': round(win_rate, 3),
        }

    def optimize(self, symbol: str, period: str = "5y",
                train_months: int = 12, test_months: int = 3) -> WFOSummary:
        """
        Run walk-forward optimization.
        
        Args:
            symbol: Stock to optimize
            train_months: Training window in months
            test_months: Testing window in months (out-of-sample)
        """
        logger.info(f"Walk-forward optimization for {symbol}...")

        df = self.dm.get_stock_data(symbol, period=period)
        if df.empty or len(df) < 300:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Create windows
        total_days = len(df)
        train_days = train_months * 21  # ~21 trading days per month
        test_days = test_months * 21

        windows = []
        window_num = 0
        start_idx = 0

        while start_idx + train_days + test_days <= total_days:
            window_num += 1
            train_end_idx = start_idx + train_days
            test_end_idx = train_end_idx + test_days

            train_df = df.iloc[start_idx:train_end_idx]
            test_df = df.iloc[train_end_idx:test_end_idx]

            # Find best parameters on training data
            best_params = None
            best_return = -999

            # Grid search
            param_combos = list(product(
                self.param_grid['rsi_buy'],
                self.param_grid['rsi_sell'],
                self.param_grid['atr_stop'],
                self.param_grid['holding_days'],
            ))

            for rsi_buy, rsi_sell, atr_stop, holding in param_combos:
                params = {
                    'rsi_buy': rsi_buy,
                    'rsi_sell': rsi_sell,
                    'atr_stop': atr_stop,
                    'holding_days': holding,
                }
                result = self._backtest_params(train_df, params)
                if result['return'] > best_return and result['num_trades'] >= 3:
                    best_return = result['return']
                    best_params = params

            if best_params is None:
                best_params = {'rsi_buy': 30, 'rsi_sell': 70, 'atr_stop': 2.0, 'holding_days': 10}
                best_return = 0

            # Test best parameters on out-of-sample data
            oos_result = self._backtest_params(test_df, best_params)
            is_result = self._backtest_params(train_df, best_params)

            windows.append(WalkForwardResult(
                window_num=window_num,
                train_start=train_df.index[0].strftime('%Y-%m-%d'),
                train_end=train_df.index[-1].strftime('%Y-%m-%d'),
                test_start=test_df.index[0].strftime('%Y-%m-%d'),
                test_end=test_df.index[-1].strftime('%Y-%m-%d'),
                best_params=best_params,
                in_sample_return=is_result['return'],
                out_sample_return=oos_result['return'],
                num_trades=oos_result['num_trades'],
                win_rate=oos_result['win_rate'],
            ))

            # Slide forward
            start_idx += test_days

        if not windows:
            return None

        # Aggregate results
        oos_returns = [w.out_sample_return for w in windows]
        is_returns = [w.in_sample_return for w in windows]

        # Most common params
        from collections import Counter
        param_counts = Counter()
        for w in windows:
            param_counts[str(w.best_params)] += 1
        most_common_params = param_counts.most_common(1)[0][0] if param_counts else "{}"

        # Degradation = how much worse OOS is vs IS
        avg_is = np.mean(is_returns) if is_returns else 0
        avg_oos = np.mean(oos_returns) if oos_returns else 0
        degradation = ((avg_is - avg_oos) / abs(avg_is) * 100) if avg_is != 0 else 0

        return WFOSummary(
            symbol=symbol,
            total_windows=len(windows),
            avg_oos_return=round(avg_oos, 2),
            total_oos_return=round(sum(oos_returns), 2),
            avg_win_rate=round(np.mean([w.win_rate for w in windows]), 3),
            best_params_frequency=dict(param_counts.most_common(3)),
            degradation=round(degradation, 1),
            windows=windows,
        )

    def print_report(self, summary: WFOSummary):
        """Print walk-forward optimization report."""
        if summary is None:
            print("\n  ❌ Could not run WFO (insufficient data)")
            return

        print("\n" + "=" * 70)
        print(f"  🔄 WALK-FORWARD OPTIMIZATION — {summary.symbol}")
        print("=" * 70)
        print(f"  Windows: {summary.total_windows}")
        print(f"  Method: 12-month train → 3-month test (rolling)")

        # Overall performance
        print(f"\n  📊 OUT-OF-SAMPLE PERFORMANCE (What Actually Matters)")
        print(f"  " + "-" * 50)
        print(f"    Avg OOS Return per window: {summary.avg_oos_return:+.2f}%")
        print(f"    Total OOS Return: {summary.total_oos_return:+.2f}%")
        print(f"    Avg Win Rate: {summary.avg_win_rate:.1%}")

        # Robustness
        print(f"\n  🛡️ ROBUSTNESS")
        print(f"  " + "-" * 50)
        degradation_emoji = "🟢" if summary.degradation < 30 else "🟡" if summary.degradation < 60 else "🔴"
        print(f"    {degradation_emoji} IS→OOS Degradation: {summary.degradation:.1f}%")
        if summary.degradation < 30:
            print(f"    ✅ Strategy is ROBUST (performs similarly out-of-sample)")
        elif summary.degradation < 60:
            print(f"    ⚠️ Moderate overfitting. Consider simpler parameters.")
        else:
            print(f"    🔴 HIGH overfitting! Strategy may not work live.")

        # Window details
        print(f"\n  📋 WINDOW DETAILS")
        print(f"  {'#':>3} {'Test Period':<24} {'IS Ret':>7} {'OOS Ret':>8} "
              f"{'Trades':>7} {'WinRate':>8} {'Params'}")
        print(f"  " + "-" * 80)

        for w in summary.windows:
            oos_emoji = "🟢" if w.out_sample_return > 0 else "🔴"
            params_short = f"RSI:{w.best_params['rsi_buy']}/{w.best_params['rsi_sell']} ATR:{w.best_params['atr_stop']}"
            print(
                f"  {w.window_num:>3} {w.test_start} → {w.test_end} "
                f"{w.in_sample_return:>+6.1f}% {oos_emoji}{w.out_sample_return:>+6.1f}% "
                f"{w.num_trades:>7} {w.win_rate:>7.0%} {params_short}"
            )

        # Most common params
        print(f"\n  🎯 OPTIMAL PARAMETERS (most frequently selected):")
        for params_str, count in summary.best_params_frequency.items():
            print(f"    Selected {count}/{summary.total_windows} times: {params_str[:60]}")

        print("\n  KEY: If OOS ≈ IS → strategy is real. If OOS << IS → overfitting.")
        print("=" * 70)
