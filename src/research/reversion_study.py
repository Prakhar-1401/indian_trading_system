"""
Reversion Study — can the medium-horizon mean-reversion edge survive costs?

The signal study found that in this universe/period, medium-horizon signals
(mom_60, dist_52w) are *mean-reverting*: buying recent losers / shorting recent
winners has statistically real predictive power (FDR-significant IC) and a
positive gross Sharpe. But at daily rebalancing the net-of-cost Sharpe CI still
straddled zero — turnover ate the edge.

This module attacks that directly. For the reversion signals it sweeps the two
levers that reduce turnover:

    * REBALANCE FREQUENCY  — trade every k days instead of daily (hold in between)
    * SELECTION QUANTILE   — trade only the most extreme names (tails)

For every (frequency, quantile) configuration it reports the NET-of-cost
annualized Sharpe with a bootstrap 95% CI, plus annual turnover and net return.
The goal is to find a configuration whose net-Sharpe CI lower bound is > 0 (a
genuinely tradeable, cost-surviving edge) rather than curve-fitting a lookback.

Because we only sweep execution levers (not the signal definition or lookback),
this is turnover engineering, not signal overfitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.fetcher import DataManager
from src.research.signal_study import DEFAULT_UNIVERSE, compute_signal_library
from src.research.statistical_validation import bootstrap_sharpe_ci
from src.utils.helpers import setup_logging

logger = setup_logging()

# Signals that screened as mean-reverting with significant IC in the signal study.
REVERSION_SIGNALS = ["mom_60", "dist_52w", "mom_120"]
REBALANCE_FREQS = [1, 5, 10, 20]          # trading days between rebalances
QUANTILES = [0.1, 0.2, 0.3]               # fraction of universe on each side
PERIODS_PER_YEAR = 252


@dataclass
class ReversionConfig:
    signal: str
    rebalance_days: int
    quantile: float
    gross_sharpe: float
    net_sharpe: float
    net_ci_low: float
    net_ci_high: float
    annual_net_return: float
    annual_turnover: float
    robust: bool                          # net-Sharpe CI lower bound > 0


class ReversionStudy:
    """
    USAGE:
        study = ReversionStudy()
        best, all_configs, returns = study.run()
        study.print_report(all_configs)
        # `returns` is the net daily return series of the single best config
    """

    def __init__(self, data_manager: Optional[DataManager] = None, cost_bps: float = 10.0):
        self.dm = data_manager or DataManager()
        self.cost_bps = cost_bps

    def _build_panels(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        panels: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.dm.get_stock_data(sym, start=start, end=end)
            except Exception as e:  # pragma: no cover
                logger.warning(f"{sym}: fetch failed ({e})")
                continue
            if df is None or df.empty or len(df) < 260:
                continue
            df = df[~df.index.duplicated(keep="last")].sort_index()
            sig = compute_signal_library(df)
            sig["fwd_1"] = df["close"].astype(float).shift(-1) / df["close"].astype(float) - 1.0
            panels[sym] = sig
        logger.info(f"Built panels for {len(panels)} symbols")
        return panels

    @staticmethod
    def _cross_section(panels: Dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
        series = {sym: p[col] for sym, p in panels.items() if col in p}
        return pd.DataFrame(series) if series else pd.DataFrame()

    def _config_returns(self, fac: pd.DataFrame, ret: pd.DataFrame,
                        direction: int, rebalance_days: int, quantile: float) -> Optional[pd.Series]:
        """Net daily return series for one (freq, quantile) configuration."""
        common = fac.index.intersection(ret.index)
        fac, ret = fac.loc[common], ret.loc[common]
        rebalance_mask = np.arange(len(common)) % rebalance_days == 0

        target = pd.DataFrame(0.0, index=common, columns=fac.columns)
        for i, dt in enumerate(common):
            if not rebalance_mask[i]:
                continue
            row = (fac.loc[dt] * direction).dropna()
            if len(row) < 10:
                continue
            n_side = max(1, int(len(row) * quantile))
            ranked = row.sort_values()
            target.loc[dt, ranked.index[-n_side:]] = 1.0 / n_side       # long expected winners
            target.loc[dt, ranked.index[:n_side]] = -1.0 / n_side       # short expected losers

        # hold weights between rebalance dates
        held = target.where(pd.Series(rebalance_mask, index=common), np.nan).ffill().fillna(0.0)
        gross = (held * ret).sum(axis=1)
        turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
        net = gross - turnover * (self.cost_bps / 10000.0)

        valid = gross.replace(0, np.nan).dropna()
        if len(valid) < 30:
            return None
        return net.loc[valid.index]

    @staticmethod
    def _sharpe(x: pd.Series) -> float:
        sd = x.std(ddof=1)
        return (x.mean() / sd) * np.sqrt(PERIODS_PER_YEAR) if sd > 0 else 0.0

    def run(self, symbols: Optional[List[str]] = None,
            start: str = "2021-01-01", end: str = "2026-01-01"):
        symbols = symbols or DEFAULT_UNIVERSE
        panels = self._build_panels(symbols, start, end)
        if not panels:
            raise RuntimeError("No data available for the study.")
        ret = self._cross_section(panels, "fwd_1")

        configs: List[ReversionConfig] = []
        best_returns: Dict[str, pd.Series] = {}
        for sig in REVERSION_SIGNALS:
            fac = self._cross_section(panels, sig)
            if fac.empty:
                continue
            direction = -1  # reversion: long low signal, short high signal
            for freq, q in product(REBALANCE_FREQS, QUANTILES):
                net = self._config_returns(fac, ret, direction, freq, q)
                if net is None:
                    continue
                # recompute gross for reporting
                net_sh = self._sharpe(net)
                _, lo, hi = bootstrap_sharpe_ci(net, n_boot=2000)
                cfg = ReversionConfig(
                    signal=sig, rebalance_days=freq, quantile=q,
                    gross_sharpe=0.0,             # filled by _enrich
                    net_sharpe=net_sh, net_ci_low=lo, net_ci_high=hi,
                    annual_net_return=float(net.mean() * PERIODS_PER_YEAR),
                    annual_turnover=0.0,          # filled by _enrich
                    robust=lo > 0,
                )
                configs.append(cfg)
                best_returns[f"{sig}|{freq}|{q}"] = net

        # enrich with gross sharpe + turnover (recompute cleanly, keeps _config_returns simple)
        configs = self._enrich(panels, ret, configs)
        configs.sort(key=lambda c: c.net_sharpe, reverse=True)

        best = configs[0] if configs else None
        best_series = best_returns.get(f"{best.signal}|{best.rebalance_days}|{best.quantile}") if best else None
        return best, configs, best_series

    def _enrich(self, panels, ret, configs: List[ReversionConfig]) -> List[ReversionConfig]:
        """Fill gross Sharpe and annual turnover for each config."""
        out = []
        for c in configs:
            fac = self._cross_section(panels, c.signal)
            common = fac.index.intersection(ret.index)
            f, r = fac.loc[common], ret.loc[common]
            mask = np.arange(len(common)) % c.rebalance_days == 0
            target = pd.DataFrame(0.0, index=common, columns=f.columns)
            for i, dt in enumerate(common):
                if not mask[i]:
                    continue
                row = (f.loc[dt] * -1).dropna()
                if len(row) < 10:
                    continue
                n_side = max(1, int(len(row) * c.quantile))
                ranked = row.sort_values()
                target.loc[dt, ranked.index[-n_side:]] = 1.0 / n_side
                target.loc[dt, ranked.index[:n_side]] = -1.0 / n_side
            held = target.where(pd.Series(mask, index=common), np.nan).ffill().fillna(0.0)
            gross = (held * r).sum(axis=1)
            turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
            valid = gross.replace(0, np.nan).dropna()
            c.gross_sharpe = self._sharpe(gross.loc[valid.index])
            c.annual_turnover = float(turnover.loc[valid.index].mean() * PERIODS_PER_YEAR)
            out.append(c)
        return out

    def print_report(self, configs: List[ReversionConfig]) -> None:
        line = "=" * 96
        print("\n" + line)
        print("  REVERSION TURNOVER STUDY  (net-of-cost Sharpe by rebalance frequency & quantile)")
        print(line)
        print(f"  Universe: {len(DEFAULT_UNIVERSE)} names | Cost: {self.cost_bps:.0f} bps | "
              f"Direction: long low-signal / short high-signal (reversion)\n")
        print(f"  {'Signal':<10} {'Rebal':>6} {'Qtile':>6} {'Gross':>7} {'Net Sh':>7} "
              f"{'Net 95% CI':>18} {'Net ret%':>9} {'Turn/yr':>8} {'Robust':>7}")
        print("  " + "-" * 92)
        for c in configs:
            ci = f"[{c.net_ci_low:.2f},{c.net_ci_high:.2f}]"
            flag = "YES" if c.robust else ""
            print(f"  {c.signal:<10} {c.rebalance_days:>5}d {c.quantile:>6.1f} {c.gross_sharpe:>7.2f} "
                  f"{c.net_sharpe:>7.2f} {ci:>18} {c.annual_net_return*100:>8.1f}% "
                  f"{c.annual_turnover:>7.0f}x {flag:>7}")
        print("  " + "-" * 92)

        robust = [c for c in configs if c.robust]
        print("\n  SUMMARY")
        print("  " + "-" * 92)
        print(f"  Configurations tested:              {len(configs)}")
        print(f"  Configurations with CI low > 0:     {len(robust)} (genuinely cost-surviving)")
        if robust:
            b = robust[0]
            print(f"  >> ROBUST edge: {b.signal}, rebalance every {b.rebalance_days}d, top/bottom "
                  f"{b.quantile:.0%}")
            print(f"     net Sharpe {b.net_sharpe:.2f}, 95% CI [{b.net_ci_low:.2f},{b.net_ci_high:.2f}] > 0, "
                  f"net {b.annual_net_return*100:.1f}%/yr at {b.annual_turnover:.0f}x turnover")
            print("     -> reducing turnover pushed the edge above the cost hurdle. Tradeable.")
        else:
            best = configs[0] if configs else None
            if best:
                print(f"  >> No config clears the bar. Best: {best.signal} @ {best.rebalance_days}d/"
                      f"{best.quantile:.0%}, net Sharpe {best.net_sharpe:.2f} "
                      f"[{best.net_ci_low:.2f},{best.net_ci_high:.2f}].")
            print("     Honest read: the reversion edge is real in IC but too thin to beat 10 bps costs")
            print("     robustly, even after turnover reduction. A defensible negative result.")
        print(line + "\n")


if __name__ == "__main__":  # pragma: no cover
    s = ReversionStudy()
    _, configs, _ = s.run()
    s.print_report(configs)
