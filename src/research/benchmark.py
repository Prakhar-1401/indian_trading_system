"""
Benchmark Analysis — alpha and beta vs the Nifty 50.

The single number an equity-strategy recruiter asks for: does the strategy
deliver return that the market index doesn't already explain? This module
regresses a strategy's daily returns on the Nifty 50 (^NSEI):

    r_strategy_t - rf = alpha + beta * (r_nifty_t - rf) + eps_t

and reports:
    * ANNUALIZED ALPHA and its Newey-West t-stat (is the intercept real?)
    * BETA (market exposure; a market-neutral L/S book should be ~0)
    * INFORMATION RATIO (alpha / tracking error, annualized)
    * R^2 (how much of the strategy is just market beta)
    * equity curves for strategy vs Nifty

Alpha t-stats use Newey-West HAC standard errors to be robust to
autocorrelation/heteroskedasticity, consistent with the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataManager
from src.utils.helpers import setup_logging

logger = setup_logging()

TRADING_DAYS = 252
DEFAULT_RF_ANNUAL = 0.065          # ~India 10y / repo proxy


@dataclass
class BenchmarkResult:
    n_obs: int
    ann_return_strategy: float
    ann_return_benchmark: float
    ann_vol_strategy: float
    ann_vol_benchmark: float
    beta: float
    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    alpha_pvalue: float
    r_squared: float
    information_ratio: float
    tracking_error: float
    strategy_sharpe: float
    benchmark_sharpe: float
    equity_strategy: pd.Series
    equity_benchmark: pd.Series


class BenchmarkAnalyzer:
    """
    USAGE:
        ba = BenchmarkAnalyzer()
        res = ba.analyze(strategy_daily_returns)   # pandas Series indexed by date
        ba.print_report(res)
    """

    def __init__(self, data_manager: Optional[DataManager] = None,
                 rf_annual: float = DEFAULT_RF_ANNUAL, benchmark_symbol: str = "^NSEI"):
        self.dm = data_manager or DataManager()
        self.rf_daily = rf_annual / TRADING_DAYS
        self.benchmark_symbol = benchmark_symbol

    def _benchmark_returns(self, start, end) -> pd.Series:
        df = self.dm.get_stock_data(self.benchmark_symbol, start=str(start.date()), end=str(end.date()))
        if df is None or df.empty:
            raise RuntimeError(f"Could not fetch benchmark {self.benchmark_symbol}")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df["close"].astype(float).pct_change().dropna()

    def analyze(self, strategy_returns: pd.Series) -> BenchmarkResult:
        import statsmodels.api as sm

        strat = strategy_returns.dropna().sort_index()
        if len(strat) < 30:
            raise ValueError("Need at least 30 return observations.")

        bench = self._benchmark_returns(strat.index.min(), strat.index.max())
        aligned = pd.concat([strat.rename("strat"), bench.rename("bench")], axis=1).dropna()
        if len(aligned) < 30:
            raise ValueError("Too little overlap between strategy and benchmark dates.")

        rs = aligned["strat"] - self.rf_daily
        rb = aligned["bench"] - self.rf_daily

        X = sm.add_constant(rb.values)
        model = sm.OLS(rs.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(4 * (len(rs) / 100) ** (2 / 9))})
        alpha_daily = float(model.params[0])
        beta = float(model.params[1])
        alpha_tstat = float(model.tvalues[0])
        alpha_pvalue = float(model.pvalues[0])
        r_squared = float(model.rsquared)

        active = aligned["strat"] - aligned["bench"]
        tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        info_ratio = float((active.mean() * TRADING_DAYS) / tracking_error) if tracking_error > 0 else 0.0

        def _sharpe(x):
            sd = x.std(ddof=1)
            return float(((x.mean() - self.rf_daily) / sd) * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

        return BenchmarkResult(
            n_obs=len(aligned),
            ann_return_strategy=float(aligned["strat"].mean() * TRADING_DAYS),
            ann_return_benchmark=float(aligned["bench"].mean() * TRADING_DAYS),
            ann_vol_strategy=float(aligned["strat"].std(ddof=1) * np.sqrt(TRADING_DAYS)),
            ann_vol_benchmark=float(aligned["bench"].std(ddof=1) * np.sqrt(TRADING_DAYS)),
            beta=beta,
            alpha_daily=alpha_daily,
            alpha_annual=alpha_daily * TRADING_DAYS,
            alpha_tstat=alpha_tstat,
            alpha_pvalue=alpha_pvalue,
            r_squared=r_squared,
            information_ratio=info_ratio,
            tracking_error=tracking_error,
            strategy_sharpe=_sharpe(aligned["strat"]),
            benchmark_sharpe=_sharpe(aligned["bench"]),
            equity_strategy=(1 + aligned["strat"]).cumprod(),
            equity_benchmark=(1 + aligned["bench"]).cumprod(),
        )

    def print_report(self, res: BenchmarkResult) -> None:
        line = "=" * 80
        print("\n" + line)
        print(f"  BENCHMARK ANALYSIS  vs {self.benchmark_symbol}  ({res.n_obs} trading days)")
        print(line)
        print(f"  {'Metric':<28}{'Strategy':>14}{'Nifty 50':>14}")
        print("  " + "-" * 54)
        print(f"  {'Annualized return':<28}{res.ann_return_strategy*100:>13.1f}%{res.ann_return_benchmark*100:>13.1f}%")
        print(f"  {'Annualized volatility':<28}{res.ann_vol_strategy*100:>13.1f}%{res.ann_vol_benchmark*100:>13.1f}%")
        print(f"  {'Sharpe (rf-adjusted)':<28}{res.strategy_sharpe:>14.2f}{res.benchmark_sharpe:>14.2f}")
        print("  " + "-" * 54)
        print("\n  MARKET REGRESSION  (strategy_excess ~ alpha + beta * nifty_excess)")
        print("  " + "-" * 54)
        print(f"  {'Beta (market exposure)':<32}{res.beta:>10.3f}")
        print(f"  {'Alpha (annualized)':<32}{res.alpha_annual*100:>9.2f}%")
        print(f"  {'Alpha t-stat (Newey-West)':<32}{res.alpha_tstat:>10.2f}")
        print(f"  {'Alpha p-value':<32}{res.alpha_pvalue:>10.3f}")
        print(f"  {'R-squared (variance from beta)':<32}{res.r_squared*100:>9.1f}%")
        print(f"  {'Information ratio':<32}{res.information_ratio:>10.2f}")
        print(f"  {'Tracking error (annualized)':<32}{res.tracking_error*100:>9.1f}%")
        print("  " + "-" * 54)

        print("\n  VERDICT")
        print("  " + "-" * 54)
        if res.alpha_pvalue < 0.05 and res.alpha_annual > 0:
            print(f"  >> Statistically significant POSITIVE alpha of {res.alpha_annual*100:.1f}%/yr")
            print(f"     (t={res.alpha_tstat:.2f}, p={res.alpha_pvalue:.3f}) not explained by market beta.")
        elif res.alpha_annual > 0:
            print(f"  >> Positive alpha ({res.alpha_annual*100:.1f}%/yr) but NOT statistically")
            print(f"     significant (t={res.alpha_tstat:.2f}, p={res.alpha_pvalue:.3f}). Not distinguishable")
            print("     from luck at the 5% level.")
        else:
            print(f"  >> Negative alpha ({res.alpha_annual*100:.1f}%/yr, t={res.alpha_tstat:.2f}). The strategy")
            print("     does not add value over the Nifty on a risk-adjusted basis.")
        if abs(res.beta) < 0.2:
            print(f"     Beta {res.beta:.2f} is near zero: the book is effectively market-neutral.")
        print(line + "\n")
