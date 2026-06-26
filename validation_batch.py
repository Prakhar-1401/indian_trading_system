import json
from datetime import datetime
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardOptimizer
from src.strategy.ml_ensemble import MLEnsembleModel

UNIVERSE_A = ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN","BHARTIARTL","ITC","LT","BAJFINANCE"]
UNIVERSE_B = ["SUNPHARMA","TITAN","MARUTI","WIPRO","NTPC","POWERGRID","COALINDIA","CIPLA","ULTRACEMCO","KOTAKBANK"]

WINDOWS = [
    ("2022-01-01", "2024-01-01"),
    ("2023-01-01", "2025-01-01"),
    ("2024-06-01", "2026-06-01"),
]

backtest_rows = []

for u_name, symbols in [("A", UNIVERSE_A), ("B", UNIVERSE_B)]:
    for start, end in WINDOWS:
        cfg = {
            "backtest": {
                "start_date": start,
                "end_date": end,
                "initial_capital": 1000000,
                "commission_pct": 0.05,
                "slippage_pct": 0.1,
            },
            "risk": {
                "stop_loss_pct": 8.0,
                "trailing_stop_pct": 12.0,
            },
            "portfolio": {
                "max_position_pct": 7.0,
                "max_stocks": 10,
                "cash_reserve_pct": 5.0,
            },
        }
        engine = BacktestEngine(config=cfg)
        res = engine.run(symbols=symbols, rebalance_freq="weekly")
        if not res:
            continue
        backtest_rows.append({
            "universe": u_name,
            "start": start,
            "end": end,
            "total_return_pct": res.get("total_return_pct"),
            "cagr_pct": res.get("cagr_pct"),
            "sharpe_ratio": res.get("sharpe_ratio"),
            "max_drawdown_pct": res.get("max_drawdown_pct"),
            "win_rate_pct": res.get("win_rate_pct"),
            "profit_factor": res.get("profit_factor"),
            "total_trades": res.get("total_trades"),
        })

wfo_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
wfo_rows = []
wfo = WalkForwardOptimizer()
for s in wfo_symbols:
    summary = wfo.optimize(s)
    if summary is None:
        continue
    wfo_rows.append({
        "symbol": s,
        "windows": summary.total_windows,
        "avg_oos_return": summary.avg_oos_return,
        "total_oos_return": summary.total_oos_return,
        "avg_win_rate": summary.avg_win_rate,
        "degradation": summary.degradation,
    })

ml_symbols = ["RELIANCE","TCS","HDFCBANK","INFY","SBIN","BHARTIARTL","SUNPHARMA","COALINDIA","NTPC","BAJFINANCE"]
ml = MLEnsembleModel()
signals = ml.predict_batch(ml_symbols)
ml_rows = []
for s in signals:
    m = ml.models.get(s.symbol, {})
    ml_rows.append({
        "symbol": s.symbol,
        "action": s.action,
        "confidence": s.confidence,
        "agreement": s.agreement,
        "rf_accuracy": round(m.get("rf_accuracy", 0.0) * 100, 2),
        "gb_accuracy": round(m.get("gb_accuracy", 0.0) * 100, 2),
        "train_samples": m.get("train_samples", 0),
    })

summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "backtest": {
        "runs": len(backtest_rows),
        "avg_cagr_pct": round(pd.DataFrame(backtest_rows)["cagr_pct"].mean(), 3) if backtest_rows else None,
        "avg_sharpe": round(pd.DataFrame(backtest_rows)["sharpe_ratio"].mean(), 3) if backtest_rows else None,
        "avg_max_drawdown_pct": round(pd.DataFrame(backtest_rows)["max_drawdown_pct"].mean(), 3) if backtest_rows else None,
        "best_run": max(backtest_rows, key=lambda x: x["cagr_pct"]) if backtest_rows else None,
        "worst_run": min(backtest_rows, key=lambda x: x["cagr_pct"]) if backtest_rows else None,
    },
    "wfo": {
        "runs": len(wfo_rows),
        "avg_total_oos_return": round(pd.DataFrame(wfo_rows)["total_oos_return"].mean(), 3) if wfo_rows else None,
        "avg_degradation": round(pd.DataFrame(wfo_rows)["degradation"].mean(), 3) if wfo_rows else None,
    },
    "ml": {
        "symbols": len(ml_rows),
        "avg_rf_accuracy": round(pd.DataFrame(ml_rows)["rf_accuracy"].mean(), 3) if ml_rows else None,
        "avg_gb_accuracy": round(pd.DataFrame(ml_rows)["gb_accuracy"].mean(), 3) if ml_rows else None,
        "sell_count": sum(1 for r in ml_rows if r["action"] == "SELL"),
        "buy_count": sum(1 for r in ml_rows if r["action"] == "BUY"),
        "hold_count": sum(1 for r in ml_rows if r["action"] == "HOLD"),
    },
}

out_json = "logs/validation_matrix_2026-06-22.json"
out_bt = "logs/validation_backtests_2026-06-22.csv"
out_wfo = "logs/validation_wfo_2026-06-22.csv"
out_ml = "logs/validation_ml_2026-06-22.csv"

with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "backtest_rows": backtest_rows,
        "wfo_rows": wfo_rows,
        "ml_rows": ml_rows,
    }, f, indent=2)

pd.DataFrame(backtest_rows).to_csv(out_bt, index=False)
pd.DataFrame(wfo_rows).to_csv(out_wfo, index=False)
pd.DataFrame(ml_rows).to_csv(out_ml, index=False)

print("VALIDATION_DONE")
print(json.dumps(summary, indent=2))
print(f"FILES: {out_json}, {out_bt}, {out_wfo}, {out_ml}")
