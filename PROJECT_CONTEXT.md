# PROJECT CONTEXT — Indian Equities Research & Trading System

> **Purpose of this file:** A complete A-to-Z handoff. If you are an AI or a developer
> reading this cold, this document gives you the full mental model of what this project
> is, why it exists, how it is built, what has been proven (and disproven), and where it
> should go next. Read this first before touching code.

- **Repository:** https://github.com/Prakhar-1401/indian_trading_system
- **Language/stack:** Python 3.13 (works on 3.11+), pandas/numpy/scipy/statsmodels/scikit-learn, Streamlit dashboard, yfinance + NSE data.
- **Domain:** Indian equities (NSE), Nifty 500 universe.
- **Author's goal:** A research-first algorithmic trading system rigorous enough to
  showcase quant-researcher skills (the kind expected at systematic funds), not a
  get-rich bot. **Rigor is the deliverable, not inflated returns.**

---

## 1. The core philosophy (READ THIS)

This project deliberately optimizes for **honest, defensible research** over pretty
backtests. Three principles run through everything:

1. **Signal-first, not strategy-first.** Instead of tuning one strategy until its
   backtest looks good (the classic overfitting trap), we screen *libraries* of signals
   and ask: is the predictive power statistically real, and does it survive costs?
2. **Statistical significance before profitability claims.** Every signal is judged with
   Information Coefficient (IC), Newey-West (HAC) t-stats, and Benjamini-Hochberg
   false-discovery-rate (FDR) correction across *all* tests — so we don't fool ourselves
   with multiple-testing noise.
3. **Net-of-cost or it doesn't count.** A signal's Sharpe is reported *after* transaction
   costs, with a bootstrap confidence interval. If the CI includes zero, we say so.

**The honest headline result:** In the tested Nifty universe (2021–2026), momentum does
**not** work — the market is mildly **mean-reverting** at medium horizons. The best
mean-reversion book is market-neutral (beta ~0.10) with ~12% raw alpha, but that alpha is
**not yet statistically significant** (t≈1.18) and its net-of-cost Sharpe CI sits right at
the zero boundary. This "here's what's real, here's what isn't" story is the point.

---

## 2. High-level architecture

```
main.py                     # single CLI entry point; dispatches ~25 commands
scheduler.py                # scheduled runs
config/strategy.yaml        # ALL strategy parameters (factor weights, thresholds)
src/
  data/         fetcher.py            # DataManager: yfinance/NSE download + CSV cache
  indicators/   technical.py          # RSI, MACD, moving averages, etc.
  ranking/      ranker.py             # multi-factor stock ranking (the composite strategy)
  strategy/     executor.py, decision_engine.py, kelly_sizing.py, ml_ensemble.py,
                regime_detector.py, pairs_trading.py, options_analyzer.py,
                event_driven.py, risk_manager.py, paper_trader.py, rebalancer.py,
                premarket_report.py
  backtest/     engine.py, walk_forward.py, monte_carlo.py,
                factor_attribution.py, sector_analysis.py
  sentiment/    news_analyzer.py, geopolitical.py
  smart_money/  tracker.py, insider_tracker.py
  research/     statistical_validation.py, signal_study.py,
                reversion_study.py, benchmark.py         # <-- the quant research layer
  portfolio/    optimizer.py          # mean-variance / risk-parity / HRP / Black-Litterman
  execution/    market_impact.py      # square-root impact + Almgren-Chriss
  dashboard/    app.py, terminal.py, screener.py, ... (Streamlit UI)
  utils/        helpers.py, telegram_alerts.py
tests/          pytest suite (28 tests) for the research/portfolio/execution layer
notebooks/      research_note.ipynb   # 2-page LaTeX research note with charts
.github/workflows/ci.yml              # CI: pytest on py3.11 + 3.12
```

There are two conceptual halves:
- **The "product" half** (original): a multi-factor ranking system + a full toolbox of
  trading features (ML, regimes, options, pairs, sentiment, Kelly sizing, paper trading,
  dashboard). This is broad and feature-rich.
- **The "research" half** (`src/research`, `src/portfolio`, `src/execution` + tests + CI +
  notebook): the academically rigorous layer added to prove/disprove edge. **This is the
  interview-grade part and the most important to understand.**

---

## 3. The data layer (`src/data/fetcher.py`)

`DataManager` is the single interface to market data. Key facts a future editor MUST know:

- `get_stock_data(symbol, period="2y", interval="1d", start=None, end=None, source="auto")`
  returns a pandas DataFrame with **lowercase** columns (`open, high, low, close, volume`)
  and a datetime index.
- **Auto-appends `.NS`** for NSE symbols (e.g. `RELIANCE` -> `RELIANCE.NS`), but **not** for
  index symbols starting with `^` (e.g. `^NSEI` = Nifty 50). This is why benchmark code can
  fetch the index directly.
- **Caches to CSV** under `data/`. Cache keys are **date-range-specific** (a bug where
  windows shared cache slices was fixed — critical for trustworthy walk-forward tests).
- `get_universe()` / `get_nifty500_symbols()` scrapes the NSE Nifty 500 CSV, and **falls
  back to a hardcoded ~50 large-cap list** if the scrape fails. The research modules use a
  self-contained 50-name universe to avoid scrape dependency.
- `get_bulk_data(symbols, period)` returns a dict of DataFrames.

---

## 4. The research layer — the heart of the project

This is the sequence a future AI should understand as **one connected research arc**:
*screen broadly → prove significance → engineer costs → benchmark alpha.*

### 4.1 `statistical_validation.py` — prove signals with academic rigor
Pure, unit-tested primitives (reused everywhere else):
- `newey_west_tstat(series)` → (mean, t-stat, p) using HAC lags = floor(4·(n/100)^(2/9)).
- `benjamini_hochberg(pvalues, alpha=0.05)` → FDR-corrected reject flags + adjusted p-values.
- `bootstrap_sharpe_ci(returns, n_boot=5000, ...)` → (point, lo, hi) Sharpe CI.
- `compute_factors(df)` → basic factor panel (mom_20, mom_60, rsi_14, vol_ratio, zscore_20).

`StatisticalValidator` computes per-signal **Information Coefficient** (Spearman rank corr,
cross-sectional, per date), **Fama-MacBeth** cross-sectional regressions, and
**turnover-adjusted (net-of-cost) Sharpe**.
**Finding:** Of 20 (signal, horizon) hypotheses on 20 large-caps, only `zscore_20 @ 1-day`
survives FDR (mean IC −0.0266, NW t=−3.24, p_adj=0.024) — a short-horizon mean-reversion
effect. Naive momentum long/short: net Sharpe −1.21, −17.3%/yr at ~165x turnover.

### 4.2 `signal_study.py` — the broad signal leaderboard
`compute_signal_library(df)` builds **16 documented anomaly signals** from OHLCV (no
look-ahead): momentum (mom_20/60/120/252, mom_12_1, dist_52w), reversal (rev_5, zscore_10/20,
rsi_14), low-vol (vol_20/60), liquidity (vol_surge, amihud illiquidity), lottery/skew
(max_20, skew_60). `SignalStudy.run()` computes IC + IC-IR + NW t-stat per (signal, horizon),
FDR-corrects across the **whole grid**, then builds a daily net-of-cost long/short book
(oriented by sign of mean IC) with bootstrap CI, and ranks into a leaderboard.
**Finding (50 names, 2021–2026):** 8/16 signals have real IC; **every momentum signal has
negative IC** (market is mean-reverting here); only 2/16 have positive net Sharpe. Best daily
book `mom_60` (dir −1): net Sharpe 0.62, CI [−0.35, 1.52] (includes zero).

### 4.3 `reversion_study.py` — can the edge survive costs?
Takes the FDR-significant reversion signals (`mom_60`, `dist_52w`, `mom_120`) and sweeps two
**turnover levers**: rebalance frequency {1,5,10,20 days} × selection quantile {0.1,0.2,0.3}.
This is **turnover engineering, not lookback tuning**, so it does not overfit the signal.
Reports net Sharpe + bootstrap CI + turnover per config.
**Finding:** Turnover reduction lifts the best net Sharpe from 0.62 → **0.92**
(`dist_52w`, 5-day rebalance, top/bottom 10%), net **18.7%/yr** at 79x turnover — but the
95% CI [−0.10, 1.98] still marginally includes zero. Real, near-tradeable, right at the cost
boundary. **0 of 36 configs clear CI>0** — a defensible negative/borderline result.

### 4.4 `benchmark.py` — alpha vs the Nifty 50
`BenchmarkAnalyzer.analyze(strategy_returns)` regresses strategy excess returns on Nifty 50
excess returns (`r = alpha + beta·r_mkt`), with Newey-West t-stats. Reports annualized alpha,
beta, R², information ratio, tracking error, and equity curves.
**Finding (best reversion book, 977 days):** beta **0.10** (market-neutral), R²=0.5%, alpha
**11.6%/yr** but NW t=1.18, p=0.24 — positive yet **not significant at 5%**.

---

## 5. Portfolio & execution layers

### `portfolio/optimizer.py`
`PortfolioOptimizer` supports **Mean-Variance** (max-Sharpe, min-variance, via SLSQP),
**Risk Parity** (equal risk contribution), **Hierarchical Risk Parity** (HRP, López de Prado:
linkage → quasi-diagonalization → recursive bisection), and **Black-Litterman**
(equilibrium prior + views). Validated: risk parity gives exactly equal risk contributions;
BL with no views returns the equal-weight prior.

### `execution/market_impact.py`
`ExecutionAnalyzer` implements the **square-root market-impact model**
(`impact_bps = c·σ·√(Q/ADV)·1e4`), **implementation shortfall** (Perold decomposition), and
the **Almgren-Chriss** optimal execution trajectory (closed-form via sinh). Example: a
Rs 10M RELIANCE order ≈ 24 bps total cost; Almgren-Chriss front-loads ~38% into the first slice.

---

## 6. The "product" toolbox (original feature set)

Reached via `main.py` commands. These are broad and functional but are **not** the rigorous
core; treat their outputs as exploratory unless validated by the research layer:
- `ranking/ranker.py` — multi-factor composite score (weights in `config/strategy.yaml`:
  momentum 0.40, quality 0.25, sentiment 0.20, smart_money 0.15).
- `strategy/` — ML ensemble, regime detection, pairs trading, options analysis, event-driven,
  Kelly position sizing, risk manager, paper trader, pre-market report, rebalancer,
  decision engine.
- `backtest/` — engine, walk-forward (overfitting detection), Monte Carlo (tail risk),
  factor attribution, sector analysis.
- `sentiment/` — news + geopolitical sentiment. `smart_money/` — insider/FII tracking.
- `dashboard/` — Streamlit app + terminal UI, screener, watchlist, performance views.

---

## 7. CLI reference (`python main.py <command>`)

Research layer (the important ones):
```
statval       # IC, Fama-MacBeth, Newey-West t-stats, FDR, bootstrap Sharpe (20 names)
signalstudy   # 16-signal anomaly leaderboard, net-of-cost ranking (50 names)
reversion     # turnover sweep (rebalance x quantile) on mean-reversion signals
benchmark     # alpha/beta of the best reversion book vs Nifty 50 (^NSEI)
optimize <m>  # portfolio optimization: max_sharpe|min_variance|risk_parity|hrp|black_litterman|all
impact <SYM> <value>   # market impact + Almgren-Chriss execution
```
Product layer: `rank, signals, backtest, sentiment, pairs, geo, insider, risk, telegram, ml,
regime, options, events, kelly, premarket, paper, montecarlo, wfo, factors, sectors, full,
decide, dashboard, terminal`.

---

## 8. Validation methodology (the 4-layer stack)

1. **Historical backtest** — baseline viability.
2. **Walk-forward** — overfitting detection. (ULTRACEMCO WFO showed 102.9% IS→OOS
   degradation = severe overfitting of a naive momentum rule — a key cautionary result.)
3. **Monte Carlo** — tail risk / distribution realism.
4. **Statistical validation + signal study** — significance & net-of-cost edge (the layer
   that actually decides whether a signal is real).

---

## 9. Honest results summary (as of July 2026)

| Study | Key number | Verdict |
|---|---|---|
| Momentum backtest | CAGR −0.9%, Sharpe −0.01, MaxDD −13.7% | momentum has no edge here |
| WFO (ULTRACEMCO) | IS→OOS degradation 102.9% | severe overfitting warning |
| statval (20 names) | only `zscore_20@1d` survives FDR | most edges are noise |
| signalstudy (50 names) | 8/16 real IC, momentum IC negative | market is mean-reverting |
| reversion sweep | best net Sharpe 0.92, CI [−0.10, 1.98] | near-tradeable, at cost boundary |
| benchmark | alpha 11.6%/yr, beta 0.10, t=1.18 | market-neutral, not yet significant |

**Bottom line:** No deployable, statistically-significant, cost-surviving edge has been
proven yet. The value delivered is the *rigorous machinery* that can tell real edge from
noise — and an honest map of what is and isn't real in this universe.

---

## 10. How to run (environment + gotchas)

```powershell
# from the project root
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# IMPORTANT on Windows: many commands print emoji; set UTF-8 to avoid cp1252 crashes
$env:PYTHONIOENCODING='utf-8'
python main.py signalstudy
pytest -q                      # 28 tests
```
- Python venv path used in development: `venv/Scripts/python.exe` (Python 3.13).
- Scientific stack required: statsmodels, scipy, scikit-learn, pandas, numpy, matplotlib,
  ipykernel, pytest (all in `requirements.txt`).
- First research run downloads ~50 symbols (~1 min); subsequent runs use the CSV cache.

---

## 11. Development workflow (dual-directory — important for future edits)

Two directories exist on the author's machine:
- **DEV:** `c:\Users\Z00587UT\indian-trading-system` — NO git; all code is written/tested here.
- **PUBLISH:** `c:\Users\Z00587UT\indian_trading_system` — the git repo (branch `main`, pushed
  to GitHub). New/changed files are copied DEV → PUBLISH, then committed and pushed.

```powershell
Copy-Item -Force "<dev>\src\research\<file>.py" "<publish>\src\research\<file>.py"
cd <publish>; git add -A; git commit -m "..."; git push origin main
```
(A future AI working only in one checkout can ignore this and just use normal git.)

---

## 12. Known limitations / not-yet-done

- **No proven positive OOS edge** across symbols/regimes (by design — honesty over hype).
- **Priority 3 "novel alpha" not built** (needs paid/tick data): FinBERT earnings NLP, SEBI
  SAST insider-filing parsing, order-flow imbalance, cross-asset signals.
- **No production data store** (PostgreSQL/TimescaleDB) — everything is CSV-cached.
- **No broker-connected live execution** with operational controls/audit logs.
- Product-layer features (ML, options, pairs, sentiment) are functional but **not
  validated** by the research layer; treat as exploratory.

---

## 13. Recommended next steps (in priority order)

1. **Composite reversion signal** — combine `mom_60` + `dist_52w` (and possibly `zscore_20`)
   into one z-scored ensemble to diversify noise; re-run reversion + benchmark to see if the
   net-Sharpe CI and alpha t-stat clear significance. (Most promising, low effort.)
2. **Extend the sample / expand universe** — more data lengthens t-stats; test Nifty 500
   (not just 50) to see if the reversion effect strengthens with more cross-section.
3. **Regime-conditioning** — test whether the reversion edge concentrates in high-volatility
   or specific regimes (via `regime_detector.py`).
4. **Novel alpha (Priority 3)** — only if paid/tick data becomes available.
5. **Infra hardening** — TimescaleDB store, then a strict long-duration forward paper test.

**Guiding rule for whoever continues this:** never chase a prettier backtest by tuning
lookbacks. Add signal diversity, more data, or better cost/turnover engineering — and always
report net-of-cost results with confidence intervals. If something isn't significant, say so.
