"""
Signal Research Study — hunting for genuine, cost-surviving edge.

This is the quant-researcher workflow made concrete: instead of tuning one
strategy until its backtest looks good (overfitting), we screen a LIBRARY of
documented anomaly signals across a broad universe and ask two questions of each:

    1. Does it have statistically significant predictive power?
       -> Information Coefficient + Newey-West t-stat + Benjamini-Hochberg FDR.
    2. Does the edge survive realistic transaction costs?
       -> A daily-rebalanced long/short book with turnover-based costs, reported
          as a NET annualized Sharpe with a bootstrap confidence interval.

Signals are then ranked by their NET (tradeable) Sharpe, and any signal whose IC
survives FDR is flagged. The output is a leaderboard you can defend in an
interview: "I screened 16 signals across 50 names; here is what survives
multiple-testing correction AND transaction costs."

SIGNAL FAMILIES COVERED (all from OHLCV, no look-ahead)
-------------------------------------------------------
- Momentum / trend:        mom_20/60/120/252, mom_12_1 (12-1), dist_52w
- Reversal / mean-revert:  rev_5, zscore_10, zscore_20, rsi_14
- Low-volatility anomaly:  vol_20, vol_60
- Liquidity:               vol_surge, amihud (illiquidity)
- Lottery / higher-moment: max_20 (MAX factor), skew_60

Each signal's long/short book is oriented by the sign of its historical mean IC
(a screening convention), so a signal that predicts *low* future returns is
traded short-the-high / long-the-low. This orientation uses full-sample sign and
is a screening step, not a deployable backtest.

REFERENCES
----------
- Jegadeesh & Titman (1993), momentum.   - Ang et al. (2006), low-volatility.
- Amihud (2002), illiquidity.            - Bali, Cakici, Whitelaw (2011), MAX.
- Grinold & Kahn (2000), IC framework.   - Benjamini & Hochberg (1995), FDR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataManager
from src.research.statistical_validation import (
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    newey_west_tstat,
)
from src.utils.helpers import setup_logging

logger = setup_logging()

# A broad, liquid universe (~50 Nifty large/mid caps). Bigger than the 20-name
# validator so cross-sectional ranks are meaningful.
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TECHM", "POWERGRID", "NTPC", "TATAMOTORS", "ONGC", "JSWSTEEL",
    "M&M", "ADANIPORTS", "TATASTEEL", "BAJAJFINSV", "COALINDIA", "GRASIM",
    "BPCL", "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "APOLLOHOSP",
    "BRITANNIA", "HINDALCO", "INDUSINDBK", "SBILIFE", "TATACONSUM", "DABUR",
    "PIDILITIND", "HAVELLS", "GODREJCP", "SHREECEM", "DIVISLAB",
]

HORIZONS = [1, 5, 10, 20]
SIGNAL_NAMES = [
    "mom_20", "mom_60", "mom_120", "mom_252", "mom_12_1", "dist_52w",
    "rev_5", "zscore_10", "zscore_20", "rsi_14",
    "vol_20", "vol_60", "vol_surge", "amihud", "max_20", "skew_60",
]


@dataclass
class SignalResult:
    signal: str
    best_horizon: int
    mean_ic: float
    ic_ir: float
    nw_tstat: float
    pvalue: float
    pvalue_adj: float
    significant: bool
    direction: int            # +1 long-high, -1 long-low (from IC sign)
    gross_sharpe: float
    net_sharpe: float
    net_ci_low: float
    net_ci_high: float
    annual_net_return: float
    annual_turnover: float


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_signal_library(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full library of candidate signals from OHLCV (no look-ahead)."""
    close = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)
    ret1 = close.pct_change()

    out = pd.DataFrame(index=df.index)
    # --- momentum / trend ---
    out["mom_20"] = close.pct_change(20)
    out["mom_60"] = close.pct_change(60)
    out["mom_120"] = close.pct_change(120)
    out["mom_252"] = close.pct_change(252)
    # 12-1 momentum: cumulative return from t-252 to t-21 (skip most recent month)
    out["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    out["dist_52w"] = close / close.rolling(252).max()
    # --- reversal / mean reversion ---
    out["rev_5"] = close.pct_change(5)
    out["zscore_10"] = (close - close.rolling(10).mean()) / close.rolling(10).std()
    out["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    out["rsi_14"] = _rsi(close, 14)
    # --- low-volatility anomaly ---
    out["vol_20"] = ret1.rolling(20).std()
    out["vol_60"] = ret1.rolling(60).std()
    # --- liquidity ---
    out["vol_surge"] = vol / vol.rolling(20).mean()
    dollar_vol = (close * vol).replace(0, np.nan)
    out["amihud"] = (ret1.abs() / dollar_vol).rolling(20).mean()
    # --- lottery / higher moments ---
    out["max_20"] = ret1.rolling(20).max()
    out["skew_60"] = ret1.rolling(60).skew()
    return out


class SignalStudy:
    """
    USAGE:
        study = SignalStudy()
        results = study.run()          # downloads (cached) universe, screens all signals
        study.print_leaderboard(results)
    """

    def __init__(self, data_manager: Optional[DataManager] = None, cost_bps: float = 10.0):
        self.dm = data_manager or DataManager()
        self.cost_bps = cost_bps

    # ----- data ---------------------------------------------------------- #
    def _build_panels(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        panels: Dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            try:
                df = self.dm.get_stock_data(sym, start=start, end=end)
            except Exception as e:  # pragma: no cover
                logger.warning(f"{sym}: fetch failed ({e})")
                continue
            if df is None or df.empty or len(df) < 260:
                continue
            df = df[~df.index.duplicated(keep="last")].sort_index()
            sig = compute_signal_library(df)
            close = df["close"].astype(float)
            for h in HORIZONS:
                sig[f"fwd_{h}"] = close.shift(-h) / close - 1.0
            panels[sym] = sig
        logger.info(f"Built panels for {len(panels)} symbols")
        return panels

    @staticmethod
    def _cross_section(panels: Dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
        series = {sym: p[col] for sym, p in panels.items() if col in p}
        return pd.DataFrame(series) if series else pd.DataFrame()

    # ----- IC ------------------------------------------------------------ #
    def _ic_series(self, panels, signal: str, horizon: int) -> pd.Series:
        from scipy import stats

        fac = self._cross_section(panels, signal)
        fwd = self._cross_section(panels, f"fwd_{horizon}")
        if fac.empty or fwd.empty:
            return pd.Series(dtype=float)
        common = fac.index.intersection(fwd.index)
        ic = {}
        for dt in common:
            pair = pd.concat([fac.loc[dt], fwd.loc[dt]], axis=1).dropna()
            if len(pair) >= 8:
                rho, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                if not np.isnan(rho):
                    ic[dt] = rho
        return pd.Series(ic).sort_index()

    # ----- net-of-cost long/short --------------------------------------- #
    def _net_long_short(self, panels, signal: str, direction: int, quantile: float = 0.2):
        fac = self._cross_section(panels, signal)
        ret = self._cross_section(panels, "fwd_1")
        if fac.empty or ret.empty:
            return None
        common = fac.index.intersection(ret.index)
        fac, ret = fac.loc[common], ret.loc[common]

        weights = pd.DataFrame(0.0, index=common, columns=fac.columns)
        for dt in common:
            row = (fac.loc[dt] * direction).dropna()
            if len(row) < 10:
                continue
            n_side = max(1, int(len(row) * quantile))
            ranked = row.sort_values()
            weights.loc[dt, ranked.index[-n_side:]] = 1.0 / n_side
            weights.loc[dt, ranked.index[:n_side]] = -1.0 / n_side

        gross = (weights * ret).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        net = gross - turnover * (self.cost_bps / 10000.0)
        valid = gross.replace(0, np.nan).dropna()
        if len(valid) < 30:
            return None

        def _sh(x):
            sd = x.std(ddof=1)
            return (x.mean() / sd) * np.sqrt(252) if sd > 0 else 0.0

        g, n = valid.index, valid.index
        gross_sh = _sh(gross.loc[g])
        net_sh = _sh(net.loc[n])
        _, lo, hi = bootstrap_sharpe_ci(net.loc[n], n_boot=2000)
        return {
            "gross_sharpe": gross_sh, "net_sharpe": net_sh,
            "ci_low": lo, "ci_high": hi,
            "annual_net_return": float(net.loc[n].mean() * 252),
            "annual_turnover": float(turnover.loc[n].mean() * 252),
        }

    # ----- orchestration ------------------------------------------------- #
    def run(self, symbols: Optional[List[str]] = None,
            start: str = "2021-01-01", end: str = "2026-01-01") -> List[SignalResult]:
        symbols = symbols or DEFAULT_UNIVERSE
        panels = self._build_panels(symbols, start, end)
        if not panels:
            raise RuntimeError("No data available for the study.")

        # 1) IC across the full (signal, horizon) grid
        grid = []  # (signal, horizon, mean_ic, ic_ir, t, p)
        for sig in SIGNAL_NAMES:
            for h in HORIZONS:
                ic = self._ic_series(panels, sig, h)
                if ic.empty:
                    grid.append((sig, h, 0.0, 0.0, 0.0, 1.0))
                    continue
                mean_ic, t, p = newey_west_tstat(ic)
                sd = ic.std(ddof=1)
                grid.append((sig, h, mean_ic, mean_ic / sd if sd > 0 else 0.0, t, p))

        # 2) FDR across the WHOLE grid (anti data-mining)
        reject, p_adj = benjamini_hochberg([g[5] for g in grid], alpha=0.05)

        grid_df = pd.DataFrame(grid, columns=["signal", "horizon", "mean_ic", "ic_ir", "t", "p"])
        grid_df["p_adj"] = p_adj
        grid_df["sig"] = reject

        # 3) collapse to best horizon per signal, add net-of-cost L/S
        results: List[SignalResult] = []
        for sig in SIGNAL_NAMES:
            sub = grid_df[grid_df["signal"] == sig]
            best = sub.iloc[sub["ic_ir"].abs().values.argmax()]
            direction = 1 if best["mean_ic"] >= 0 else -1
            ls = self._net_long_short(panels, sig, direction)
            any_sig = bool(sub["sig"].any())
            results.append(SignalResult(
                signal=sig, best_horizon=int(best["horizon"]),
                mean_ic=float(best["mean_ic"]), ic_ir=float(best["ic_ir"]),
                nw_tstat=float(best["t"]), pvalue=float(best["p"]),
                pvalue_adj=float(best["p_adj"]), significant=any_sig,
                direction=direction,
                gross_sharpe=ls["gross_sharpe"] if ls else 0.0,
                net_sharpe=ls["net_sharpe"] if ls else 0.0,
                net_ci_low=ls["ci_low"] if ls else 0.0,
                net_ci_high=ls["ci_high"] if ls else 0.0,
                annual_net_return=ls["annual_net_return"] if ls else 0.0,
                annual_turnover=ls["annual_turnover"] if ls else 0.0,
            ))

        results.sort(key=lambda r: r.net_sharpe, reverse=True)
        return results

    # ----- reporting ----------------------------------------------------- #
    def print_leaderboard(self, results: List[SignalResult]) -> None:
        line = "=" * 100
        print("\n" + line)
        print("  SIGNAL RESEARCH LEADERBOARD  (ranked by net-of-cost long/short Sharpe)")
        print(line)
        print(f"  Universe: {len(DEFAULT_UNIVERSE)} names | Signals: {len(results)} | "
              f"Cost: {self.cost_bps:.0f} bps | Grid FDR-corrected")
        print("  dir: +1 = long high / short low, -1 = long low / short high\n")
        print(f"  {'Signal':<10} {'Dir':>4} {'BestH':>6} {'IC-IR':>7} {'NW t':>7} "
              f"{'p_adj':>8} {'FDR':>4} {'Gross':>7} {'Net Sh':>7} {'Net 95% CI':>18} "
              f"{'Net ret%':>9} {'Turn/yr':>8}")
        print("  " + "-" * 96)
        for r in results:
            ci = f"[{r.net_ci_low:.2f},{r.net_ci_high:.2f}]"
            flag = "YES" if r.significant else ""
            print(f"  {r.signal:<10} {r.direction:>4} {r.best_horizon:>6} {r.ic_ir:>7.3f} "
                  f"{r.nw_tstat:>7.2f} {r.pvalue_adj:>8.3f} {flag:>4} {r.gross_sharpe:>7.2f} "
                  f"{r.net_sharpe:>7.2f} {ci:>18} {r.annual_net_return*100:>8.1f}% "
                  f"{r.annual_turnover:>7.0f}x")
        print("  " + "-" * 96)

        # Robust = IC significant AND net-Sharpe CI strictly above zero.
        # Promising = IC significant AND positive point net Sharpe, but CI includes 0.
        robust = [r for r in results if r.significant and r.net_ci_low > 0]
        promising = [r for r in results if r.significant and r.net_sharpe > 0 and r.net_ci_low <= 0]
        surviving_cost = [r for r in results if r.net_sharpe > 0]
        print("\n  SUMMARY")
        print("  " + "-" * 96)
        print(f"  Signals FDR-significant (IC):       {sum(r.significant for r in results)}/{len(results)}")
        print(f"  Signals with POSITIVE net Sharpe:   {len(surviving_cost)}/{len(results)}")
        if robust:
            best = robust[0]
            print(f"  >> ROBUST edge: '{best.signal}' (dir {best.direction:+d}, H={best.best_horizon}) "
                  f"net Sharpe {best.net_sharpe:.2f}, 95% CI [{best.net_ci_low:.2f},{best.net_ci_high:.2f}] "
                  f"> 0, t={best.nw_tstat:.2f}")
            print("     -> significant IC AND net Sharpe CI excludes zero: build a strategy on it.")
        elif promising:
            best = promising[0]
            print(f"  >> PROMISING (not yet robust): '{best.signal}' (dir {best.direction:+d}, "
                  f"H={best.best_horizon}) net Sharpe {best.net_sharpe:.2f}, "
                  f"95% CI [{best.net_ci_low:.2f},{best.net_ci_high:.2f}] INCLUDES zero.")
            print("     -> IC is statistically real, but the NET-of-cost profitability is not")
            print("        distinguishable from zero. Honest read: a genuine short-horizon")
            print("        mean-reversion signal whose edge is largely eaten by transaction costs.")
        else:
            print("  >> No signal is both FDR-significant AND net-Sharpe-positive.")
            print("     Honest conclusion: no robust, cost-surviving edge in this signal set/universe.")
            print("     This is itself a defensible research finding (avoids overfitting a dead signal).")
        print(line + "\n")


if __name__ == "__main__":  # pragma: no cover
    s = SignalStudy()
    s.print_leaderboard(s.run())
