# Indian Trading System

A modular, research-first algorithmic trading system for Indian equities (NSE) that combines:
- multi-factor ranking
- machine learning confidence scoring
- Kelly-criterion position sizing
- market regime and event filters
- backtesting + walk-forward validation + Monte Carlo stress testing
- live paper trading workflow

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
python main.py terminal
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
src/
  backtest/
  data/
  dashboard/
  indicators/
  ranking/
  sentiment/
  smart_money/
  strategy/
  utils/
main.py
requirements.txt
```

## 14) Resume/interview positioning (truthful)

Built a modular Indian equities quant platform from scratch with multi-factor ranking, ML confidence modeling, Kelly sizing, walk-forward validation, Monte Carlo stress testing, and live paper-trading workflow. Implemented robust validation pipeline, identified and fixed a cache-window contamination issue, and produced reproducible performance diagnostics across regimes.
