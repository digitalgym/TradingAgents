"""Optimize all top strategies with Optuna."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TRANSFORMERS_NO_TORCH"] = "1"

from tradingagents.quant_strats.trainer import WalkForwardTrainer
from tradingagents.quant_strats.strategies.crypto_breakout import CryptoBreakoutStrategy
from tradingagents.quant_strats.strategies.crypto_trend_follow import CryptoTrendFollowStrategy
from tradingagents.quant_strats.strategies.smc_zones import SMCZonesStrategy
from tradingagents.quant_strats.strategies.breakout import BreakoutStrategy
from tradingagents.quant_strats.config import RiskDefaults
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
mt5.initialize()

def load_data(symbol, tf, bars=5000):
    tf_map = {"H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1, "H1": mt5.TIMEFRAME_H1}
    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(tf, mt5.TIMEFRAME_D1), 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return df

trainer = WalkForwardTrainer()

strat_classes = {
    'crypto_breakout': CryptoBreakoutStrategy,
    'crypto_trend': CryptoTrendFollowStrategy,
    'smc_zones': SMCZonesStrategy,
    'breakout': BreakoutStrategy,
}

combos = [
    ('crypto_breakout', 'XAUUSD', 'D1'),
    ('crypto_breakout', 'XAGUSD', 'D1'),
    ('crypto_breakout', 'BTCUSD', 'D1'),
    ('crypto_breakout', 'XRPUSD', 'H4'),
    ('crypto_trend', 'XAUUSD', 'D1'),
    ('crypto_trend', 'XAGUSD', 'D1'),
    ('smc_zones', 'XAUUSD', 'D1'),
    ('smc_zones', 'XAGUSD', 'D1'),
    ('breakout', 'XAUUSD', 'D1'),
    ('breakout', 'XAGUSD', 'D1'),
]

print("MULTI-STRATEGY OPTUNA OPTIMIZATION (40 trials each)", flush=True)
print("=" * 115, flush=True)

for strat_name, symbol, tf in combos:
    cls = strat_classes[strat_name]
    df = load_data(symbol, tf, 5000)
    if df is None or len(df) < 300:
        print(f'{strat_name} {symbol} {tf}: insufficient data', flush=True)
        continue

    def make_objective(cls_, df_, symbol_, tf_):
        def objective(trial):
            sl = trial.suggest_float("sl_atr_mult", 1.0, 4.0, step=0.25)
            tp = trial.suggest_float("tp_atr_mult", 2.0, 10.0, step=0.5)
            threshold = trial.suggest_float("signal_threshold", 0.30, 0.80, step=0.05)
            max_hold = trial.suggest_int("max_hold_bars", 10, 50, step=5)
            if tp / sl < 1.5:
                return -10.0
            risk = RiskDefaults(sl_atr_mult=sl, tp_atr_mult=tp, signal_threshold=threshold, max_hold_bars=max_hold)
            strategy = cls_(risk=risk)
            try:
                result = trainer.train_and_evaluate(strategy, df_, symbol_, tf_)
                if result.total_trades < 15:
                    return -10.0
                return result.sharpe
            except Exception:
                return -10.0
        return objective

    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(cls, df, symbol, tf), n_trials=40, show_progress_bar=False)

    best = study.best_params
    risk = RiskDefaults(
        sl_atr_mult=best["sl_atr_mult"], tp_atr_mult=best["tp_atr_mult"],
        signal_threshold=best["signal_threshold"], max_hold_bars=best["max_hold_bars"],
    )
    strategy = cls(risk=risk)
    result = trainer.train_and_evaluate(strategy, df, symbol, tf)

    wr = result.win_rate
    if wr > 1: wr = wr / result.total_trades * 100 if result.total_trades else 0
    else: wr *= 100
    rr = best["tp_atr_mult"] / best["sl_atr_mult"]

    print(f'{strat_name:20s} {symbol} {tf}: SL={best["sl_atr_mult"]:.2f}x TP={best["tp_atr_mult"]:.1f}x '
          f'thr={best["signal_threshold"]:.2f} hold={best["max_hold_bars"]} R:R={rr:.1f} | '
          f'WR={wr:.1f}% PF={result.profit_factor:.2f} Sharpe={result.sharpe:.3f} Trades={result.total_trades}',
          flush=True)

mt5.shutdown()
print("\nDone.", flush=True)
