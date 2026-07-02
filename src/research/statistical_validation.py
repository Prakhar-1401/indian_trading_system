"""
Statistical Validation — proving signals work with academic rigor.

This is the module that separates a "retail backtester" from a quant researcher.
Instead of asking "what was the return?", it asks "is the alpha statistically
real, or is it luck / data-mining?".

WHAT IT COMPUTES
----------------
1. Information Coefficient (IC) + IC-IR
   - IC_t = cross-sectional Spearman rank correlation between a factor's value
     across stocks at time t and the stocks' forward returns.
   - IC-IR = mean(IC) / std(IC)  -> risk-adjusted signal quality (like a Sharpe
     for the signal itself). |IC-IR| > 0.5 is genuinely good.

2. Newey-West (HAC) adjusted t-statistics
   - The IC time series is autocorrelated (overlapping forward windows), so a
     naive t-stat overstates significance. Newey-West HAC standard errors fix
     this. This is the standard practice in published factor research.

3. Benjamini-Hochberg (FDR) multiple-testing correction
   - When you test many (factor, horizon) pairs you WILL find spurious winners.
     BH controls the false discovery rate so a "significant" result means
     something. This is the antidote to data-mining.

4. Fama-MacBeth cross-sectional regression
   - For each date, regress forward returns on factor exposures; average the
     slopes over time and test them with NW errors. This is THE canonical test
     for whether a factor earns a premium.

5. Turnover-adjusted (net) Sharpe
   - Build the factor's long/short portfolio, subtract realistic transaction
     costs scaled by turnover, and report gross vs net Sharpe. Alpha that dies
     after costs is not alpha.

6. Bootstrap confidence interval on the Sharpe ratio
   - "My Sharpe is 1.2" is meaningless without error bars. We resample returns
     to produce "Sharpe = 1.2, 95% CI [0.4, 1.9]".

REFERENCES
----------
- Fama & MacBeth (1973), Journal of Political Economy.
- Newey & West (1987), Econometrica.
- Benjamini & Hochberg (1995), JRSS-B.
- Grinold & Kahn, "Active Portfolio Management" (IC / IC-IR framework).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:  # statsmodels powers HAC errors + FDR; guarded so import never hard-crashes
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False

from src.data.fetcher import DataManager
from src.utils.helpers import setup_logging

logger = setup_logging()

# Default broad-but-liquid universe (Nifty large caps). Cross-sectional tests
# need a reasonable number of names per date to be meaningful.
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "BAJFINANCE", "SUNPHARMA", "TITAN", "MARUTI", "WIPRO", "NTPC",
    "POWERGRID", "COALINDIA", "CIPLA", "ULTRACEMCO", "KOTAKBANK",
]

DEFAULT_HORIZONS = [1, 5, 10, 20]


# --------------------------------------------------------------------------- #
#  Result containers
# --------------------------------------------------------------------------- #
@dataclass
class ICResult:
    factor: str
    horizon: int
    mean_ic: float
    ic_ir: float
    nw_tstat: float
    pvalue: float
    pvalue_adj: float = np.nan
    significant: bool = False
    n_periods: int = 0


@dataclass
class FamaMacBethResult:
    factor: str
    horizon: int
    mean_premium: float       # average per-period slope (in return units)
    nw_tstat: float
    pvalue: float
    n_periods: int


@dataclass
class TurnoverSharpeResult:
    factor: str
    gross_sharpe: float
    net_sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    annual_gross_return: float
    annual_net_return: float
    avg_daily_turnover: float
    annual_turnover: float
    n_days: int


@dataclass
class ValidationReport:
    universe: List[str]
    start: str
    end: str
    horizons: List[int]
    cost_bps: float
    ic_results: List[ICResult] = field(default_factory=list)
    fm_results: List[FamaMacBethResult] = field(default_factory=list)
    turnover_result: Optional[TurnoverSharpeResult] = None


# --------------------------------------------------------------------------- #
#  Statistical primitives (pure functions — easy to unit test)
# --------------------------------------------------------------------------- #
def newey_west_tstat(series: pd.Series, lags: Optional[int] = None) -> tuple[float, float, float]:
    """
    Test H0: mean(series) == 0 using Newey-West (HAC) standard errors.

    Returns (mean, t_stat, p_value). Falls back to an iid t-test if statsmodels
    is unavailable.
    """
    x = pd.Series(series).dropna().astype(float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float(x.mean()) if n else 0.0, 0.0, 1.0

    mean = float(x.mean())

    if not _HAS_SM:
        se = x.std(ddof=1) / np.sqrt(n)
        t = mean / se if se > 0 else 0.0
        # two-sided normal approx
        from scipy import stats
        p = 2 * (1 - stats.norm.cdf(abs(t)))
        return mean, t, float(p)

    if lags is None:
        # Standard rule of thumb: floor(4 * (n/100)^(2/9))
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(lags, 1)

    y = x.values
    X = np.ones((n, 1))
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(model.tvalues[0])
    p = float(model.pvalues[0])
    return mean, t, p


def benjamini_hochberg(pvalues: List[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (reject_flags, adjusted_pvalues).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.array([])

    if _HAS_SM:
        reject, p_adj, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
        return reject, p_adj

    # Manual BH fallback
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out <= alpha, out


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 5000,
    periods_per_year: int = 252,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval on the annualized Sharpe ratio.

    Returns (point_sharpe, ci_low, ci_high).
    """
    r = pd.Series(returns).dropna().astype(float).values
    n = len(r)
    if n < 10:
        return 0.0, 0.0, 0.0

    def _sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        if sd == 0:
            return 0.0
        return (x.mean() / sd) * np.sqrt(periods_per_year)

    point = _sharpe(r)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = r[rng.integers(0, n, size=n)]
        boots[i] = _sharpe(sample)

    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return float(point), lo, hi


# --------------------------------------------------------------------------- #
#  Factor construction
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a small panel of standard cross-sectional factors from OHLCV.

    Returns a DataFrame indexed like `df` with one column per factor. All
    factors are computed using ONLY information available up to time t (no
    look-ahead), so they can be legitimately paired with forward returns.
    """
    close = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(index=df.index, dtype=float)

    out = pd.DataFrame(index=df.index)
    out["mom_20"] = close.pct_change(20)
    out["mom_60"] = close.pct_change(60)
    out["rsi_14"] = _rsi(close, 14)
    out["vol_ratio"] = vol / vol.rolling(20).mean()
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out["zscore_20"] = (close - ma20) / sd20
    return out


FACTOR_NAMES = ["mom_20", "mom_60", "rsi_14", "vol_ratio", "zscore_20"]


# --------------------------------------------------------------------------- #
#  Main engine
# --------------------------------------------------------------------------- #
class StatisticalValidator:
    """
    Run the full statistical validation suite over a universe of stocks.

    USAGE:
        v = StatisticalValidator()
        report = v.run(horizons=[1, 5, 10, 20])
        v.print_report(report)
    """

    def __init__(self, data_manager: Optional[DataManager] = None):
        self.dm = data_manager or DataManager()

    # ----- data assembly ------------------------------------------------- #
    def _build_panels(
        self, symbols: List[str], start: str, end: str, horizons: List[int]
    ) -> Dict[str, pd.DataFrame]:
        """Per-symbol DataFrame holding factors + forward returns for each horizon."""
        panels: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.dm.get_stock_data(sym, start=start, end=end)
            except Exception as e:  # pragma: no cover - network dependent
                logger.warning(f"{sym}: fetch failed ({e})")
                continue
            if df is None or df.empty or len(df) < 80:
                continue

            df = df[~df.index.duplicated(keep="last")].sort_index()
            factors = compute_factors(df)
            close = df["close"].astype(float)
            for h in horizons:
                factors[f"fwd_{h}"] = close.shift(-h) / close - 1.0
            panels[sym] = factors
        return panels

    @staticmethod
    def _cross_section(panels: Dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
        """Wide frame: rows = dates, columns = symbols, values = `col`."""
        series = {sym: p[col] for sym, p in panels.items() if col in p}
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series)

    # ----- 1. IC analysis ------------------------------------------------ #
    def information_coefficient(
        self, panels: Dict[str, pd.DataFrame], factor: str, horizon: int
    ) -> tuple[pd.Series, ICResult]:
        from scipy import stats

        fac_cs = self._cross_section(panels, factor)
        fwd_cs = self._cross_section(panels, f"fwd_{horizon}")
        if fac_cs.empty or fwd_cs.empty:
            return pd.Series(dtype=float), ICResult(factor, horizon, 0, 0, 0, 1, np.nan, False, 0)

        common = fac_cs.index.intersection(fwd_cs.index)
        ic_series = {}
        for dt in common:
            f = fac_cs.loc[dt]
            r = fwd_cs.loc[dt]
            pair = pd.concat([f, r], axis=1).dropna()
            if len(pair) >= 5:  # need a real cross-section
                rho, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                if not np.isnan(rho):
                    ic_series[dt] = rho
        ic = pd.Series(ic_series).sort_index()

        if ic.empty:
            return ic, ICResult(factor, horizon, 0, 0, 0, 1, np.nan, False, 0)

        mean_ic, t, p = newey_west_tstat(ic)
        std_ic = ic.std(ddof=1)
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        return ic, ICResult(factor, horizon, mean_ic, ic_ir, t, p, np.nan, False, len(ic))

    # ----- 2. Fama-MacBeth ---------------------------------------------- #
    def fama_macbeth(
        self, panels: Dict[str, pd.DataFrame], factor: str, horizon: int
    ) -> FamaMacBethResult:
        """
        Per-period cross-sectional regression of forward returns on a
        z-scored factor exposure; average the slopes (the 'factor premium')
        and test with Newey-West errors.
        """
        fac_cs = self._cross_section(panels, factor)
        fwd_cs = self._cross_section(panels, f"fwd_{horizon}")
        if fac_cs.empty or fwd_cs.empty:
            return FamaMacBethResult(factor, horizon, 0, 0, 1, 0)

        common = fac_cs.index.intersection(fwd_cs.index)
        slopes = {}
        for dt in common:
            f = fac_cs.loc[dt]
            r = fwd_cs.loc[dt]
            pair = pd.concat([f, r], axis=1).dropna()
            if len(pair) < 5:
                continue
            x = pair.iloc[:, 0].values.astype(float)
            y = pair.iloc[:, 1].values.astype(float)
            # cross-sectionally standardize the exposure so slope = premium per 1 std
            sd = x.std(ddof=1)
            if sd == 0:
                continue
            x = (x - x.mean()) / sd
            # OLS slope via covariance (avoids statsmodels overhead per period)
            beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
            slopes[dt] = beta
        s = pd.Series(slopes).sort_index()
        if s.empty:
            return FamaMacBethResult(factor, horizon, 0, 0, 1, 0)
        mean, t, p = newey_west_tstat(s)
        return FamaMacBethResult(factor, horizon, mean, t, p, len(s))

    # ----- 3. Turnover-adjusted Sharpe ---------------------------------- #
    def turnover_adjusted_sharpe(
        self,
        panels: Dict[str, pd.DataFrame],
        factor: str,
        cost_bps: float = 10.0,
        quantile: float = 0.3,
    ) -> TurnoverSharpeResult:
        """
        Build a daily long/short portfolio that is long the top `quantile` of
        stocks by `factor` and short the bottom `quantile`, rebalanced daily.
        Report gross vs net (turnover-cost-adjusted) Sharpe.
        """
        fac_cs = self._cross_section(panels, factor)
        ret_cs = self._cross_section(panels, "fwd_1")  # 1-day forward = next-day return
        if fac_cs.empty or ret_cs.empty:
            return TurnoverSharpeResult(factor, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        common = fac_cs.index.intersection(ret_cs.index)
        fac_cs = fac_cs.loc[common]
        ret_cs = ret_cs.loc[common]

        weights = pd.DataFrame(0.0, index=common, columns=fac_cs.columns)
        for dt in common:
            row = fac_cs.loc[dt].dropna()
            if len(row) < 6:
                continue
            n_side = max(1, int(len(row) * quantile))
            ranked = row.sort_values()
            shorts = ranked.index[:n_side]
            longs = ranked.index[-n_side:]
            weights.loc[dt, longs] = 1.0 / n_side
            weights.loc[dt, shorts] = -1.0 / n_side

        # gross daily P&L = sum(weight * next-day return)
        gross = (weights * ret_cs).sum(axis=1)
        # turnover = sum |w_t - w_{t-1}| ; cost charged on traded notional
        turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
        cost = turnover * (cost_bps / 10000.0)
        net = gross - cost

        valid = gross.replace(0, np.nan).dropna()
        if len(valid) < 10:
            return TurnoverSharpeResult(factor, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        def _sharpe(x: pd.Series) -> float:
            sd = x.std(ddof=1)
            return (x.mean() / sd) * np.sqrt(252) if sd > 0 else 0.0

        gross_sharpe = _sharpe(gross.loc[valid.index])
        net_sharpe = _sharpe(net.loc[valid.index])
        _, ci_lo, ci_hi = bootstrap_sharpe_ci(net.loc[valid.index])

        avg_turn = float(turnover.loc[valid.index].mean())
        return TurnoverSharpeResult(
            factor=factor,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            sharpe_ci_low=ci_lo,
            sharpe_ci_high=ci_hi,
            annual_gross_return=float(gross.loc[valid.index].mean() * 252),
            annual_net_return=float(net.loc[valid.index].mean() * 252),
            avg_daily_turnover=avg_turn,
            annual_turnover=avg_turn * 252,
            n_days=len(valid),
        )

    # ----- orchestration ------------------------------------------------- #
    def run(
        self,
        symbols: Optional[List[str]] = None,
        start: str = "2022-01-01",
        end: str = "2026-01-01",
        horizons: Optional[List[int]] = None,
        cost_bps: float = 10.0,
        sharpe_factor: str = "mom_20",
    ) -> ValidationReport:
        symbols = symbols or DEFAULT_UNIVERSE
        horizons = horizons or DEFAULT_HORIZONS

        logger.info(f"Building panels for {len(symbols)} symbols {start}->{end}")
        panels = self._build_panels(symbols, start, end, horizons)
        if not panels:
            raise RuntimeError("No data available to validate.")

        report = ValidationReport(
            universe=list(panels.keys()), start=start, end=end,
            horizons=horizons, cost_bps=cost_bps,
        )

        # IC + Fama-MacBeth for every factor/horizon
        for factor in FACTOR_NAMES:
            for h in horizons:
                _, ic_res = self.information_coefficient(panels, factor, h)
                report.ic_results.append(ic_res)
                report.fm_results.append(self.fama_macbeth(panels, factor, h))

        # FDR correction across all IC p-values
        pvals = [r.pvalue for r in report.ic_results]
        reject, p_adj = benjamini_hochberg(pvals, alpha=0.05)
        for r, rej, pa in zip(report.ic_results, reject, p_adj):
            r.pvalue_adj = float(pa)
            r.significant = bool(rej)

        # turnover-adjusted Sharpe on the headline factor
        report.turnover_result = self.turnover_adjusted_sharpe(panels, sharpe_factor, cost_bps)
        return report

    # ----- reporting ----------------------------------------------------- #
    def print_report(self, report: ValidationReport) -> None:
        line = "=" * 78
        print("\n" + line)
        print("  STATISTICAL VALIDATION REPORT")
        print(line)
        print(f"  Universe symbols: {len(report.universe)} | "
              f"Period: {report.start} -> {report.end}")
        print(f"  Horizons tested: {report.horizons} | Trading cost: {report.cost_bps:.0f} bps")

        print("\n  IC / IC-IR / SIGNIFICANCE (Newey-West + Benjamini-Hochberg FDR)")
        print("  " + "-" * 74)
        print(f"  {'Factor':<10} {'H':>3} {'Mean IC':>9} {'IC-IR':>8} "
              f"{'NW t':>8} {'p':>10} {'p_adj':>10} {'Sig':>5}")
        for r in report.ic_results:
            print(f"  {r.factor:<10} {r.horizon:>3} {r.mean_ic:>9.5f} {r.ic_ir:>8.3f} "
                  f"{r.nw_tstat:>8.3f} {r.pvalue:>10.6f} {r.pvalue_adj:>10.6f} "
                  f"{'YES' if r.significant else 'NO':>5}")

        print("\n  FAMA-MACBETH FACTOR PREMIA (slope per 1-std exposure, NW t-stat)")
        print("  " + "-" * 74)
        print(f"  {'Factor':<10} {'H':>3} {'Premium':>11} {'NW t':>8} {'p':>10} {'N':>6}")
        for r in report.fm_results:
            star = "*" if r.pvalue < 0.05 else " "
            print(f"  {r.factor:<10} {r.horizon:>3} {r.mean_premium:>11.6f} "
                  f"{r.nw_tstat:>8.3f} {r.pvalue:>10.6f}{star} {r.n_periods:>6}")

        t = report.turnover_result
        if t is not None:
            print("\n  TURNOVER-ADJUSTED PERFORMANCE (daily long/short)")
            print("  " + "-" * 74)
            print(f"  Factor: {t.factor}")
            print(f"  Gross Sharpe: {t.gross_sharpe:>7.3f}")
            print(f"  Net Sharpe:   {t.net_sharpe:>7.3f}   "
                  f"(95% bootstrap CI [{t.sharpe_ci_low:.3f}, {t.sharpe_ci_high:.3f}])")
            print(f"  Annual gross return: {t.annual_gross_return*100:>7.2f}%")
            print(f"  Annual net return:   {t.annual_net_return*100:>7.2f}%")
            print(f"  Avg daily turnover:  {t.avg_daily_turnover:>7.4f}  "
                  f"(~{t.annual_turnover:.1f}x / yr)")
            print(f"  Days evaluated:      {t.n_days}")

        n_sig = sum(r.significant for r in report.ic_results)
        best = max(report.ic_results, key=lambda r: abs(r.ic_ir)) if report.ic_results else None
        print("\n  SUMMARY")
        print("  " + "-" * 74)
        print(f"  Tests run: {len(report.ic_results)}")
        print(f"  Significant after FDR: {n_sig}")
        if best:
            print(f"  Best |IC-IR|: {best.factor} @ H={best.horizon} (IC-IR={best.ic_ir:.3f})")
        print(line + "\n")


if __name__ == "__main__":  # pragma: no cover
    v = StatisticalValidator()
    rep = v.run()
    v.print_report(rep)
