"""
SMC Ruthless Simplification Test

Tests combinations of SMC rules from simplest to most complex,
measuring which additions actually improve performance.

Philosophy: "Start by coding just 2-3 core rules before layering more."

Test matrix:
  Layer 0: BOS bias only (directional filter)
  Layer 1: + OB/FVG retest (entry at zone)
  Layer 2: + Volume filter (confirmation)
  Layer 3: + Kill zone session
  Layer 4: + OTE overlap
  Layer 5: + Liquidity sweep
  Layer 6: + Displacement strength
  Layer 7: + CHOCH confirmation
  Full: All features (current fvg_rebalance)

Each layer adds ONE rule. We measure if the addition helps or hurts.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TRANSFORMERS_NO_TORCH"] = "1"

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from tradingagents.quant_strats.trainer import WalkForwardTrainer
from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures
from tradingagents.quant_strats.config import RiskDefaults

mt5.initialize()
trainer = WalkForwardTrainer()


def load_data(symbol, tf, bars=5000):
    tf_map = {"H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(tf, mt5.TIMEFRAME_D1), 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return df


class SimplifiedSMCFeatures(CompositeFeatures):
    def __init__(self, windows=None):
        super().__init__(
            providers=[TechnicalFeatures(windows=windows), SMCFeatures(windows=windows)],
            windows=windows,
        )


class LayeredSMCStrategy(BaseStrategy):
    """Configurable SMC strategy with togglable layers."""

    def __init__(self, layers=None, risk=None, windows=None):
        super().__init__(risk=risk, windows=windows)
        self.layers = layers or set()
        self._name = "smc_layer_" + "_".join(sorted(self.layers)) if self.layers else "smc_base"

    @property
    def name(self):
        return self._name

    def get_feature_set(self):
        return SimplifiedSMCFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # === LAYER 0: BOS bias (directional filter) ===
        bos_bull = row.get("bos_bullish_recent", 0) or 0
        bos_bear = row.get("bos_bearish_recent", 0) or 0
        if bos_bull == 0 and bos_bear == 0:
            return 0.0  # No structure = no trade
        direction = 1.0 if bos_bull > bos_bear else -1.0
        score += 0.30

        # === LAYER 1: OB/FVG retest (zone proximity) ===
        if "zone_retest" in self.layers:
            fvg_dist = abs(row.get("nearest_fvg_dist_atr", 5) or 5)
            ob_dist = abs(row.get("nearest_ob_dist_atr", 5) or 5)
            zone_dist = min(fvg_dist, ob_dist)
            if zone_dist > 2.0:
                return 0.0  # No zone nearby
            score += 0.25 * max(0, 1.0 - zone_dist / 2.0)

        # === LAYER 2: Volume filter ===
        if "volume" in self.layers:
            vol = row.get("volume_ratio", 1.0) or 1.0
            if vol > 1.2:
                score += 0.10
            elif vol < 0.5:
                score *= 0.5  # Low volume = weak setup

        # === LAYER 3: Kill zone session ===
        if "session" in self.layers:
            in_kill = row.get("in_kill_zone", 0) or 0
            if in_kill > 0:
                score *= 1.10
            else:
                in_london = row.get("in_london_session", 0) or 0
                in_ny = row.get("in_ny_session", 0) or 0
                if in_london == 0 and in_ny == 0:
                    score *= 0.7  # Off-session penalty

        # === LAYER 4: OTE overlap ===
        if "ote" in self.layers:
            in_ote = row.get("in_ote_zone", 0) or 0
            if in_ote > 0:
                score *= 1.15

        # === LAYER 5: Liquidity sweep ===
        if "sweep" in self.layers:
            has_sweep = row.get("has_strong_sweep", 0) or 0
            if has_sweep > 0:
                score *= 1.15

        # === LAYER 6: Displacement strength ===
        if "displacement" in self.layers:
            bos_disp = row.get("bos_displacement_strength", 0) or 0
            if bos_disp > 1.5:
                score *= 1.10
            elif 0 < bos_disp < 0.8:
                score *= 0.7

        # === LAYER 7: CHOCH confirmation ===
        if "choch" in self.layers:
            choch = row.get("choch_detected", 0) or 0
            if choch:
                score *= 1.20  # Strong bonus
            # No penalty for missing CHOCH — it's a bonus, not requirement

        return min(1.0, score) * direction

    @property
    def default_risk(self):
        return RiskDefaults(
            sl_atr_mult=2.5, tp_atr_mult=7.5,
            signal_threshold=0.55, max_hold_bars=25,
        )


# Define test configurations — each adds one layer
layer_configs = [
    ("L0: BOS only", set()),
    ("L1: + Zone retest", {"zone_retest"}),
    ("L2: + Volume", {"zone_retest", "volume"}),
    ("L3: + Session", {"zone_retest", "volume", "session"}),
    ("L4: + OTE", {"zone_retest", "volume", "session", "ote"}),
    ("L5: + Sweep", {"zone_retest", "volume", "session", "ote", "sweep"}),
    ("L6: + Displacement", {"zone_retest", "volume", "session", "ote", "sweep", "displacement"}),
    ("L7: + CHOCH (full)", {"zone_retest", "volume", "session", "ote", "sweep", "displacement", "choch"}),
]

# Also test specific minimal combos
minimal_configs = [
    ("Min A: BOS + Zone", {"zone_retest"}),
    ("Min B: BOS + Zone + CHOCH", {"zone_retest", "choch"}),
    ("Min C: BOS + Zone + Sweep", {"zone_retest", "sweep"}),
    ("Min D: BOS + Zone + Session", {"zone_retest", "session"}),
    ("Min E: BOS + Zone + OTE + CHOCH", {"zone_retest", "ote", "choch"}),
    ("Min F: BOS + Zone + Sweep + CHOCH", {"zone_retest", "sweep", "choch"}),
]

pairs = [("XAUUSD", "D1"), ("XAGUSD", "D1"), ("BTCUSD", "D1")]

risk = RiskDefaults(sl_atr_mult=2.5, tp_atr_mult=7.5, signal_threshold=0.55, max_hold_bars=25)

print("SMC RUTHLESS SIMPLIFICATION TEST", flush=True)
print("=" * 110, flush=True)

for symbol, tf in pairs:
    df = load_data(symbol, tf, 5000)
    if df is None or len(df) < 300:
        print(f"{symbol} {tf}: insufficient data", flush=True)
        continue

    # Pre-compute features once (expensive)
    print(f"\n{'=' * 110}", flush=True)
    print(f"{symbol} {tf} — Computing features...", flush=True)
    feature_set = SimplifiedSMCFeatures()
    features = feature_set.compute(df)
    print(f"Features computed: {len(features)} rows, {len(features.columns)} cols", flush=True)

    print(f"\n{'Config':<35s} {'PF':>6s} {'Sharpe':>8s} {'Trades':>7s} {'WR%':>6s} {'Delta':>8s}", flush=True)
    print("-" * 80, flush=True)

    prev_sharpe = 0

    # Layered tests
    print("--- Incremental Layers ---", flush=True)
    for name, layers in layer_configs:
        strategy = LayeredSMCStrategy(layers=layers, risk=risk)
        strategy._name = name.replace(" ", "_").replace(":", "").replace("+", "")
        result = trainer.train_and_evaluate(strategy, df, symbol, tf, precomputed_features=features)
        wr = result.win_rate
        if wr > 1: wr = wr / result.total_trades * 100 if result.total_trades else 0
        else: wr *= 100
        delta = result.sharpe - prev_sharpe
        delta_str = f"{delta:+.3f}" if prev_sharpe > 0 else "base"
        print(f"{name:<35s} {result.profit_factor:>6.2f} {result.sharpe:>8.3f} {result.total_trades:>7d} {wr:>5.1f}% {delta_str:>8s}", flush=True)
        prev_sharpe = result.sharpe

    # Minimal combo tests
    print("\n--- Minimal Combos ---", flush=True)
    for name, layers in minimal_configs:
        strategy = LayeredSMCStrategy(layers=layers, risk=risk)
        strategy._name = name.replace(" ", "_").replace(":", "").replace("+", "")
        result = trainer.train_and_evaluate(strategy, df, symbol, tf, precomputed_features=features)
        wr = result.win_rate
        if wr > 1: wr = wr / result.total_trades * 100 if result.total_trades else 0
        else: wr *= 100
        print(f"{name:<35s} {result.profit_factor:>6.2f} {result.sharpe:>8.3f} {result.total_trades:>7d} {wr:>5.1f}%", flush=True)

mt5.shutdown()
print("\nDone.", flush=True)
