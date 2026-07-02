"""
Portfolio Construction — turning a basket of stocks into an optimal allocation.

Most retail systems equal-weight or eyeball position sizes. Quant desks SOLVE
for weights. This module implements the four allocation schemes an interviewer
expects you to know, from textbook to cutting-edge:

1. Mean-Variance Optimization (Markowitz, 1952)
   - Max-Sharpe (tangency) and Min-Variance portfolios on the efficient frontier.
   - The foundation of modern portfolio theory; also famous for being unstable
     (garbage-in expected returns -> garbage-out weights), which motivates 2-4.

2. Risk Parity (Equal Risk Contribution)
   - Each asset contributes the SAME amount of risk to the portfolio. No expected
     returns needed -> far more robust. This is the strategy that built Bridgewater
     and is core to BlackRock's risk-based products.

3. Hierarchical Risk Parity (HRP, Lopez de Prado, 2016)
   - Clusters assets by a distance metric, then allocates top-down via recursive
     bisection. Avoids inverting an ill-conditioned covariance matrix entirely,
     so it out-of-samples MVO. Signals you read modern quant research.

4. Black-Litterman (1992)
   - Starts from the market-implied equilibrium returns (reverse optimization) and
     Bayesian-updates them with the manager's VIEWS. Produces stable, intuitive
     weights. The model Goldman Sachs Asset Management actually built.

REFERENCES
----------
- Markowitz (1952), "Portfolio Selection", Journal of Finance.
- Maillard, Roncalli, Teiletche (2010), risk-parity construction.
- Lopez de Prado (2016), "Building Diversified Portfolios that Outperform OOS".
- Black & Litterman (1992), Financial Analysts Journal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from src.data.fetcher import DataManager
from src.utils.helpers import setup_logging

logger = setup_logging()

TRADING_DAYS = 252


@dataclass
class OptimizationResult:
    method: str
    weights: pd.Series
    expected_return: float       # annualized
    volatility: float            # annualized
    sharpe: float
    risk_contributions: pd.Series
    diversification_ratio: float


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _annualize_return(mean_daily: pd.Series) -> pd.Series:
    return mean_daily * TRADING_DAYS


def _annualize_cov(cov_daily: pd.DataFrame) -> pd.DataFrame:
    return cov_daily * TRADING_DAYS


def portfolio_stats(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0
) -> tuple[float, float, float]:
    """Return (annual_return, annual_vol, sharpe) for given weights."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Percentage risk contribution of each asset (sums to 1)."""
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol == 0:
        return np.full_like(weights, 1.0 / len(weights))
    marginal = cov @ weights
    contrib = weights * marginal / port_vol
    return contrib / contrib.sum()


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Weighted avg vol / portfolio vol. Higher = better diversified."""
    vols = np.sqrt(np.diag(cov))
    weighted_vol = float(weights @ vols)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    return weighted_vol / port_vol if port_vol > 0 else 1.0


# --------------------------------------------------------------------------- #
#  Optimizer
# --------------------------------------------------------------------------- #
class PortfolioOptimizer:
    """
    USAGE:
        opt = PortfolioOptimizer()
        opt.load(["RELIANCE", "TCS", "HDFCBANK", ...])
        res = opt.optimize("hrp")          # or max_sharpe / min_variance / risk_parity / black_litterman
        opt.print_report(res)
    """

    def __init__(self, data_manager: Optional[DataManager] = None, rf_annual: float = 0.065):
        self.dm = data_manager or DataManager()
        self.rf_annual = rf_annual           # Indian risk-free ~6.5% (T-bill)
        self.rf_daily = rf_annual / TRADING_DAYS
        self.symbols: List[str] = []
        self.returns: Optional[pd.DataFrame] = None
        self.mu: Optional[pd.Series] = None          # annualized mean returns
        self.cov: Optional[pd.DataFrame] = None      # annualized covariance

    # ----- data ---------------------------------------------------------- #
    def load(self, symbols: List[str], period: str = "2y") -> "PortfolioOptimizer":
        prices = {}
        for sym in symbols:
            try:
                df = self.dm.get_stock_data(sym, period=period)
            except Exception as e:  # pragma: no cover
                logger.warning(f"{sym}: {e}")
                continue
            if df is not None and not df.empty and len(df) > 60:
                prices[sym] = df["close"].astype(float)
        if len(prices) < 2:
            raise RuntimeError("Need at least 2 valid symbols to optimize.")

        px = pd.DataFrame(prices).sort_index().dropna(how="all").ffill().dropna()
        rets = px.pct_change().dropna()
        self.symbols = list(rets.columns)
        self.returns = rets
        self.mu = _annualize_return(rets.mean())
        self.cov = _annualize_cov(rets.cov())
        return self

    # ----- 1. Mean-Variance --------------------------------------------- #
    def _mean_variance(self, objective: str) -> np.ndarray:
        n = len(self.symbols)
        mu = self.mu.values
        cov = self.cov.values
        rf = self.rf_annual
        x0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        if objective == "max_sharpe":
            def neg_sharpe(w):
                r, v, _ = portfolio_stats(w, mu, cov, rf)
                return -(r - rf) / v if v > 0 else 1e6
            obj = neg_sharpe
        else:  # min_variance
            def variance(w):
                return w @ cov @ w
            obj = variance

        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-10})
        w = res.x if res.success else x0
        w = np.clip(w, 0, None)
        return w / w.sum()

    # ----- 2. Risk Parity ------------------------------------------------ #
    def _risk_parity(self) -> np.ndarray:
        n = len(self.symbols)
        cov = self.cov.values
        target = np.full(n, 1.0 / n)

        def objective(w):
            rc = risk_contributions(w, cov)
            return np.sum((rc - target) ** 2)

        x0 = np.full(n, 1.0 / n)
        bounds = [(1e-4, 1.0)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-12})
        w = res.x if res.success else x0
        w = np.clip(w, 0, None)
        return w / w.sum()

    # ----- 3. Hierarchical Risk Parity ---------------------------------- #
    def _hrp(self) -> np.ndarray:
        cov = self.cov
        corr = self.returns.corr()
        # distance metric d = sqrt(0.5 * (1 - corr))  (Lopez de Prado)
        dist = np.sqrt(0.5 * (1 - corr).clip(lower=0))
        condensed = squareform(dist.values, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = self._quasi_diag(link, len(corr))
        ordered = corr.index[sort_ix].tolist()
        weights = self._recursive_bisection(cov, ordered)
        return weights.reindex(self.symbols).values

    @staticmethod
    def _quasi_diag(link: np.ndarray, n: int) -> List[int]:
        """Return the leaf order that places similar assets adjacently."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        while sort_ix.max() >= n:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= n]
            i = df0.index
            j = df0.values - n
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _recursive_bisection(self, cov: pd.DataFrame, ordered: List[str]) -> pd.Series:
        """Canonical Lopez de Prado top-down recursive bisection."""
        w = pd.Series(1.0, index=ordered)
        clusters = [ordered]
        while clusters:
            # bisect every cluster with length > 1
            clusters = [
                c[j:k]
                for c in clusters
                for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]
            # process the resulting halves in left/right pairs
            for i in range(0, len(clusters), 2):
                left = clusters[i]
                right = clusters[i + 1]
                var_left = self._cluster_var(cov, left)
                var_right = self._cluster_var(cov, right)
                alpha = 1 - var_left / (var_left + var_right)
                w[left] *= alpha
                w[right] *= (1 - alpha)
        return w / w.sum()

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
        sub = cov.loc[items, items].values
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    # ----- 4. Black-Litterman ------------------------------------------- #
    def _black_litterman(
        self, views: Optional[Dict[str, float]] = None, tau: float = 0.05
    ) -> np.ndarray:
        """
        Reverse-optimize market-cap-proxy weights to equilibrium returns, then
        blend in the manager's views. With no views, returns the market prior.
        """
        cov = self.cov.values
        n = len(self.symbols)
        # Use equal-weight market prior (we lack market caps here) and derive a
        # risk-aversion coefficient from the prior's Sharpe.
        w_mkt = np.full(n, 1.0 / n)
        r, v, _ = portfolio_stats(w_mkt, self.mu.values, cov, self.rf_annual)
        delta = (r - self.rf_annual) / (v ** 2) if v > 0 else 2.5
        pi = delta * cov @ w_mkt                      # equilibrium excess returns

        if not views:
            posterior = pi
        else:
            # Build picking matrix P and view vector Q (absolute views on annual ret)
            P = np.zeros((len(views), n))
            Q = np.zeros(len(views))
            idx = {s: i for i, s in enumerate(self.symbols)}
            for k, (sym, q) in enumerate(views.items()):
                if sym in idx:
                    P[k, idx[sym]] = 1.0
                    Q[k] = q - self.rf_annual         # excess-return view
            omega = np.diag(np.diag(P @ (tau * cov) @ P.T)) + 1e-8 * np.eye(len(views))
            tau_cov = tau * cov
            inv = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P)
            posterior = inv @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(omega) @ Q)

        # optimal weights given posterior returns: w ∝ delta^-1 * cov^-1 * posterior
        w = np.linalg.solve(delta * cov, posterior)
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = w_mkt
        return w / w.sum()

    # ----- dispatch ------------------------------------------------------ #
    def optimize(
        self, method: str = "hrp", views: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        if self.returns is None:
            raise RuntimeError("Call load() before optimize().")

        method = method.lower().replace("-", "_")
        if method in ("max_sharpe", "maxsharpe", "sharpe"):
            w = self._mean_variance("max_sharpe"); label = "Max-Sharpe (Markowitz)"
        elif method in ("min_variance", "minvar", "minvariance"):
            w = self._mean_variance("min_variance"); label = "Min-Variance (Markowitz)"
        elif method in ("risk_parity", "riskparity", "rp"):
            w = self._risk_parity(); label = "Risk Parity (Equal Risk Contribution)"
        elif method == "hrp":
            w = self._hrp(); label = "Hierarchical Risk Parity (HRP)"
        elif method in ("black_litterman", "blacklitterman", "bl"):
            w = self._black_litterman(views); label = "Black-Litterman"
        else:
            raise ValueError(f"Unknown method '{method}'.")

        weights = pd.Series(w, index=self.symbols)
        mu = self.mu.values
        cov = self.cov.values
        ret, vol, sharpe = portfolio_stats(w, mu, cov, self.rf_annual)
        rc = pd.Series(risk_contributions(w, cov), index=self.symbols)
        dr = diversification_ratio(w, cov)
        return OptimizationResult(label, weights, ret, vol, sharpe, rc, dr)

    # ----- reporting ----------------------------------------------------- #
    def print_report(self, res: OptimizationResult) -> None:
        line = "=" * 70
        print("\n" + line)
        print(f"  PORTFOLIO OPTIMIZATION — {res.method}")
        print(line)
        print(f"  Expected return (ann): {res.expected_return*100:>7.2f}%")
        print(f"  Volatility (ann):      {res.volatility*100:>7.2f}%")
        print(f"  Sharpe ratio:          {res.sharpe:>7.3f}   (rf={self.rf_annual*100:.1f}%)")
        print(f"  Diversification ratio: {res.diversification_ratio:>7.3f}")
        print("\n  {:<14} {:>10} {:>14}".format("Asset", "Weight", "Risk Contrib"))
        print("  " + "-" * 40)
        order = res.weights.sort_values(ascending=False)
        for sym in order.index:
            print(f"  {sym:<14} {res.weights[sym]*100:>9.2f}% "
                  f"{res.risk_contributions[sym]*100:>13.2f}%")
        print(f"  {'TOTAL':<14} {res.weights.sum()*100:>9.2f}% "
              f"{res.risk_contributions.sum()*100:>13.2f}%")
        print(line + "\n")


if __name__ == "__main__":  # pragma: no cover
    syms = ["RELIANCE", "TCS", "HDFCBANK", "SBIN", "BHARTIARTL",
            "SUNPHARMA", "COALINDIA", "NTPC", "BAJFINANCE", "ITC"]
    opt = PortfolioOptimizer().load(syms)
    for m in ["max_sharpe", "min_variance", "risk_parity", "hrp", "black_litterman"]:
        opt.print_report(opt.optimize(m))
