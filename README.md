# Indian Trading System

A modular, research-first algorithmic trading system for Indian equities (NSE) that combines:
- multi-factor ranking
- machine learning confidence scoring
- Kelly-criterion position sizing
- market regime and event filters
- backtesting + walk-forward validation + Monte Carlo stress testing
- **statistical signal validation** (Information Coefficient, Fama-MacBeth, Newey-West t-stats, FDR correction, bootstrap Sharpe CIs)
- **portfolio construction** (Mean-Variance, Risk Parity, Hierarchical Risk Parity, Black-Litterman)
- **transaction-cost modelling** (square-root market impact, implementation shortfall, Almgren-Chriss execution)
- live paper trading workflow
- unit tests + CI (pytest + GitHub Actions) and a reproducible research note (Jupyter + LaTeX)

This repository is designed to be honest about what is validated, what is exploratory, and how results were produced.

## 1) Why this project was built

Most retail systems fail for one of two reasons:
1. They rely on one indicator and overfit quickly.
2. They skip robust validation and go directly from idea to capital.

This project was built from scratch to solve both:
- combine multiple independent edges (technical, quality, sentiment, smart money)
- force every strategy decision through validation layers used by professionals

## 2) Strategy philosophy

The core philosophy is evidence stacking.

Instead of saying "RSI is bullish, therefore buy", the system asks:
- Does momentum support the trade?
- Is the underlying company quality acceptable?
- Is market/news sentiment supportive?
- Is smart money flow aligned?
- Does ML agree with historical pattern probability?
- Is the market regime favorable?
- Is sizing mathematically justified (Kelly edge)?

If enough independent modules align, only then a high-conviction trade is considered.

## 3) End-to-end architecture

### Data layer (`src/data/fetcher.py`)
- Yahoo Finance based OHLCV and fundamentals
- caching for speed
- DataManager abstraction to keep modules decoupled

### Indicators (`src/indicators/technical.py`)
- RSI, MACD, moving averages, volatility, volume confirmation
- momentum score normalized to 0-10

### Multi-factor ranker (`src/ranking/ranker.py`)
Composite score:

`Composite = 0.40*Momentum + 0.25*Quality + 0.20*Sentiment + 0.15*SmartMoney`

Used for stock ranking and rotational selection.

### Decision engine (`src/strategy/decision_engine.py`)
Unified 0-100 score from:
- signal score
- ML confidence
- Kelly edge
- regime bonus/penalty
- event safety
- geopolitical safety

Verdict:
- TAKE
- WATCH
- SKIP

### ML ensemble (`src/strategy/ml_ensemble.py`)
- feature engineering (49+ features in current build)
- Random Forest + Gradient Boosting
- confidence-based BUY/HOLD/SELL
- agreement scoring between models

### Kelly sizing (`src/strategy/kelly_sizing.py`)
- computes edge from win rate and avg win/loss
- supports fractional Kelly for risk control

### Backtesting (`src/backtest/engine.py`)
- weekly rebalance simulation
- stop-loss and trailing logic
- transaction costs + slippage
- metrics: CAGR, Sharpe, drawdown, win rate, PF

### Walk-forward optimization (`src/backtest/walk_forward.py`)
- train window -> test window rolling validation
- reports out-of-sample return and degradation

### Monte Carlo (`src/backtest/monte_carlo.py`)
- distributional stress testing
- probability of profit, VaR, CVaR, drawdown distribution

### Dashboard (`src/dashboard/terminal.py`)
- integrated trading terminal
- ranking, analysis, paper trading, performance views

### Statistical validation (`src/research/statistical_validation.py`)
- Information Coefficient (IC) and IC-IR per signal/horizon
- Newey-West (HAC) adjusted t-statistics on the IC time series
- Benjamini-Hochberg false-discovery-rate correction across all tests
- Fama-MacBeth cross-sectional regression for factor premia
- turnover-adjusted (net-of-cost) Sharpe with a bootstrap confidence interval

### Signal research study (`src/research/signal_study.py`)
- library of 16 documented anomaly signals (momentum, reversal, low-vol, liquidity, lottery/skew)
- Information Coefficient + IC-IR + Newey-West t-stat per signal
- Benjamini-Hochberg FDR across the whole (signal x horizon) grid (anti data-mining)
- net-of-cost long/short Sharpe with bootstrap CI, ranked into a leaderboard

### Reversion turnover study (`src/research/reversion_study.py`)
- takes the FDR-significant mean-reversion signals and sweeps rebalance frequency x quantile
- reports net-of-cost Sharpe + bootstrap CI for each configuration
- turnover engineering (not lookback tuning) to test whether the edge can clear the cost hurdle

### Benchmark analysis (`src/research/benchmark.py`)
- regresses strategy returns on the Nifty 50 (`^NSEI`): `r = alpha + beta * r_mkt`
- annualized alpha with Newey-West t-stat, beta, R-squared, information ratio

### Portfolio construction (`src/portfolio/optimizer.py`)
- Mean-Variance (max-Sharpe and min-variance)
- Risk Parity (equal risk contribution)
- Hierarchical Risk Parity (HRP, Lopez de Prado)
- Black-Litterman (equilibrium prior + views)

### Execution quality (`src/execution/market_impact.py`)
- square-root market-impact model: `impact = c * sigma * sqrt(Q/ADV)`
- implementation shortfall (Perold) decomposition
- Almgren-Chriss optimal execution trajectory

### Research note (`notebooks/research_note.ipynb`)
- reproducible 2-page quant research note with LaTeX, tables, and charts
- ties the statistical results, portfolio construction, and cost model together

## 4) Validation methodology (how real traders validate)

The project follows a 4-layer validation stack:

1. Historical backtest (baseline viability)
2. Walk-forward (overfitting detection)
3. Monte Carlo (tail risk and distribution realism)
4. Live paper trading (execution realism)

Cross-platform reconciliation (TradingView strategy replication) is used to verify engine consistency.

## 5) Real results captured in this repository

The numbers below come from executed runs in this project (not synthetic examples).

### A) Multi-run backtest matrix (6 runs)
Source: `logs/validation_backtests_2026-06-22.csv` (generated during validation batch)

- Avg CAGR: 10.815%
- Avg Sharpe: 1.085
- Avg Max Drawdown: -10.343%
- Best run: CAGR 19.84%, Sharpe 1.94, MaxDD -10.49%
- Worst run: CAGR 0.58%, Sharpe 0.11, MaxDD -12.59%

Interpretation:
- strategy performance is regime-sensitive
- older windows stronger than latest window

### B) Walk-forward optimization (5 symbols)
Source: `logs/validation_wfo_2026-06-22.csv`

- Avg total out-of-sample return: -7.404%
- Avg degradation: 106.1%

Interpretation:
- significant overfitting risk in current parameterization
- out-of-sample robustness must improve before real capital deployment

### C) ML ensemble
Source: `logs/validation_ml_2026-06-22.csv`

- Avg RF accuracy: 63.234%
- Avg GB accuracy: 66.667%
- Current signal split: 0 BUY, 3 HOLD, 7 SELL

Interpretation:
- directional model has predictive signal above random baseline
- but confidence is currently defensive/risk-off for sampled symbols

### D) Monte Carlo (10,000 simulations, 1-year horizon)
Source: `montecarlo_results_2026-06-22.txt`

- Mean return: +17.46%
- Median return: +16.45%
- P(Profit > 0): 85.9%
- 95% VaR: -8.01%
- CVaR: -12.90%

Interpretation:
- favorable expected distribution under model assumptions
- still requires robust OOS verification due to WFO degradation warning

### E) Statistical signal validation (20 large-caps, 2022-2026)
Source: `python main.py statval`

Of 20 (signal, horizon) hypotheses tested, only **one survives** Newey-West +
Benjamini-Hochberg correction:

- `zscore_20 @ 1-day`: mean IC -0.0266, Newey-West t = -3.24, adjusted p = 0.024 (significant)
- a short-horizon **mean-reversion** effect; trend/momentum signals were not significant

Turnover-adjusted momentum long/short:
- Gross Sharpe -0.05, **Net Sharpe -1.21** (95% bootstrap CI [-2.17, -0.18])
- Annual net return -17.3% at ~165x annual turnover

Interpretation:
- disciplined result: most apparent edges are noise after correcting for multiple testing
- a naive momentum long/short does not survive realistic transaction costs

### F) Signal research leaderboard (50 large/mid-caps, 2021-2026)
Source: `python main.py signalstudy`

Screening 16 anomaly signals x 4 horizons (FDR-corrected across the full grid):
- **8/16 signals** have statistically real IC after Newey-West + Benjamini-Hochberg
- every momentum signal has *negative* IC: in this universe/period the effect is **mean-reversion**, not trend
- only **2/16** have a positive net-of-cost long/short Sharpe (`mom_60`, `dist_52w`)
- best daily book (`mom_60`, dir -1): net Sharpe 0.62, 95% CI [-0.35, 1.52] (includes zero)

### G) Reversion turnover study
Source: `python main.py reversion`

Sweeping rebalance frequency x selection quantile on the reversion signals:
- turnover reduction lifts the best net Sharpe from 0.62 -> **0.92** (`dist_52w`, 5-day rebalance, top/bottom 10%)
- net return rises to **18.7%/yr** at 79x turnover, but the 95% CI [-0.10, 1.98] still marginally includes zero
- honest read: a genuine, near-tradeable reversion edge whose robustness is right at the cost boundary

### H) Alpha vs Nifty 50
Source: `python main.py benchmark`

Regressing the best reversion book on the Nifty 50 (977 trading days):
- **beta 0.10** (effectively market-neutral), R-squared 0.5%
- **alpha 11.6%/yr**, Newey-West t = 1.18, p = 0.24 (positive but not significant at 5%)
- verdict: uncorrelated to the market with promising raw alpha, but not yet distinguishable from luck

## 6) Important engineering correction made

A validation-critical bug was fixed:
- historical cache keys previously did not differentiate explicit `start/end` windows
- this could contaminate multi-window backtests with reused cache slices
- fixed in `src/data/fetcher.py` by using date-range-specific cache keys

This ensures windowed validation is now trustworthy.

## 7) What is validated vs what is not

### Validated now
- executable pipeline from data -> signals -> backtest -> paper trade
- multi-run robustness stats and stress-testing outputs
- walk-forward diagnostics including degradation

### Not yet fully production-validated
- stable positive out-of-sample edge across symbols/regimes
- broker-connected live execution with operational controls
- long-duration forward test with strict risk limits and audit logs

## 8) Local setup

## Prerequisites
- Python 3.10+
- Windows/macOS/Linux

## Install
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Run key modules
```bash
python main.py backtest
python main.py wfo
python main.py ml
python main.py montecarlo
python main.py statval            # statistical signal validation (IC, Fama-MacBeth, t-stats)
python main.py signalstudy        # screen 16 anomaly signals, net-of-cost leaderboard
python main.py reversion          # turnover sweep on the mean-reversion signals
python main.py benchmark          # alpha/beta of the best reversion book vs Nifty 50
python main.py optimize hrp       # portfolio optimization (max_sharpe/min_variance/risk_parity/hrp/black_litterman)
python main.py impact RELIANCE 10000000   # market impact + Almgren-Chriss execution
python main.py terminal
```

## Run tests
```bash
pytest -q
```

## Run dashboard terminal
```bash
python -m streamlit run src/dashboard/terminal.py --server.port 8502
```

## 9) Recommended professional workflow

Daily:
1. refresh data
2. run decision engine + risk sizing
3. take only high-conviction paper trades
4. journal entries/exits and slippage

Weekly:
1. run validation batch
2. compare rolling CAGR/Sharpe/drawdown drift
3. track WFO degradation trend
4. update feature set/filters only if justified

Monthly:
1. reconcile with TradingView strategy outputs
2. review regime-wise performance segmentation
3. decide whether strategy parameters are promotable

## 10) TradingView cross-platform validation (step-by-step)

1. Create a TradingView `strategy` script (not indicator).
2. Implement same entry/exit logic used in project baseline (RSI/MACD/MA/stop rules).
3. Set equivalent assumptions:
   - commission: 0.05%
   - slippage: 0.1%
   - comparable capital and timeframe
4. Run identical windows used in validation matrix.
5. Export trade list and compare with Python engine:
   - total return
   - drawdown
   - trade count
   - win rate
6. If drift is large, inspect implementation mismatches (bar close semantics, stop execution model, corporate action handling).

## 11) Risk disclaimer

This repository is for research and education. It is not financial advice.
All strategy outputs can fail in unseen market regimes.
Do not deploy real capital without long forward testing, controls, and independent verification.

## 12) Contribution and extension roadmap

High-impact next steps:
- unify ML + ranker + decision-engine into one fully backtestable event loop
- add regime-conditioned parameter sets
- improve OOS robustness and reduce WFO degradation
- add brokerage adapter with dry-run and production safety checks
- build experiment tracking for reproducibility

## 13) Project structure

```text
config/
notebooks/
  research_note.ipynb
src/
  backtest/
  data/
  dashboard/
  execution/
  indicators/
  portfolio/
  ranking/
  research/
  sentiment/
  smart_money/
  strategy/
  utils/
tests/
.github/workflows/ci.yml
main.py
pytest.ini
requirements.txt
```

## 14) Resume/interview positioning (truthful)

Built a modular Indian equities quant platform from scratch with multi-factor ranking, ML confidence modeling, Kelly sizing, walk-forward validation, Monte Carlo stress testing, and live paper-trading workflow. Implemented robust validation pipeline, identified and fixed a cache-window contamination issue, and produced reproducible performance diagnostics across regimes.
