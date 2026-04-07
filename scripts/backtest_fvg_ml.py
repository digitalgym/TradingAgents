"""Backtest FVG Rebalance ML vs Rule-Based."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TRANSFORMERS_NO_TORCH"] = "1"

from tradingagents.quant_strats.trainer import WalkForwardTrainer
from tradingagents.quant_strats.strategies.fvg_rebalance import FVGRebalanceStrategy
from tradingagents.quant_strats.strategies.fvg_rebalance_ml import FVGRebalanceMLStrategy
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()

def load_data(symbol, tf, bars=5000):
    tf_map = {"H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(tf, mt5.TIMEFRAME_D1), 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return df

trainer = WalkForwardTrainer()

combos = [
    ('XAUUSD', 'D1'),
    ('XAGUSD', 'D1'),
    ('BTCUSD', 'D1'),
    ('XRPUSD', 'H4'),
    ('XAUUSD', 'H4'),
]

print("FVG REBALANCE: RULE-BASED vs ML (XGBoost)", flush=True)
print("=" * 100, flush=True)
print(f"{'Pair':<10s} {'TF':>4s} | {'--- Rule-Based ---':^30s} | {'--- XGBoost ML ---':^30s}", flush=True)
print(f"{'':10s} {'':>4s} | {'PF':>6s} {'Sharpe':>8s} {'Trades':>7s} | {'PF':>6s} {'Sharpe':>8s} {'Trades':>7s} {'Better':>7s}", flush=True)
print("-" * 100, flush=True)

for symbol, tf in combos:
    df = load_data(symbol, tf, 5000)
    if df is None or len(df) < 300:
        print(f'{symbol:<10s} {tf:>4s} | insufficient data', flush=True)
        continue

    # Rule-based
    rb_strategy = FVGRebalanceStrategy()
    rb_result = trainer.train_and_evaluate(rb_strategy, df, symbol, tf)

    # ML
    ml_strategy = FVGRebalanceMLStrategy()
    ml_result = trainer.train_and_evaluate(ml_strategy, df, symbol, tf)

    rb_pf = rb_result.profit_factor
    rb_sh = rb_result.sharpe
    rb_tr = rb_result.total_trades

    ml_pf = ml_result.profit_factor
    ml_sh = ml_result.sharpe
    ml_tr = ml_result.total_trades

    better = "ML" if ml_sh > rb_sh else "RULES"

    print(f'{symbol:<10s} {tf:>4s} | {rb_pf:>6.2f} {rb_sh:>8.3f} {rb_tr:>7d} | {ml_pf:>6.2f} {ml_sh:>8.3f} {ml_tr:>7d} {better:>7s}', flush=True)

mt5.shutdown()
print("\nDone.", flush=True)
