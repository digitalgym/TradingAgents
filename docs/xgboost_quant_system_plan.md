# XGBoost Quant Strategy System — Implementation Plan

## What This Is

A library of XGBoost-based trading strategies that learn from historical data, deployed alongside the existing LLM-based quant pipelines. A pair scanner finds opportunities, a strategy selector picks the best model(s) for each pair, and a parameter tuner optimises per-pair at deployment time.

This does NOT replace existing quants. It's a new pipeline type (`XGBOOST`) that sits alongside `SMC_QUANT`, `VOLUME_PROFILE`, etc.

---

## What Already Exists (Reuse, Don't Rebuild)

### Data Layer — READY
- `tradingagents/dataflows/mt5_data.py` → `get_mt5_data(symbol, start, end, timeframe)` returns OHLCV
- Supports all timeframes (M1→MN1), all symbols, commodity aliases
- `get_mt5_current_price()` for live bid/ask

### Indicator Layer — PARTIALLY READY (need vectorisation)
- `tradingagents/indicators/smart_money.py` → `SmartMoneyAnalyzer.analyze_full_smc(df)` returns:
  - Order Blocks: `strength` (0-1), `top/bottom/midpoint`, `mitigated`, `type`
  - FVGs: `size`, `fill_percentage`, `remaining_size`
  - Confluence: `total_score` (0-100), `bullish_factors`, `bearish_factors`, `bias`
  - Premium/Discount: `position_pct` (0-100), `zone`
  - OTE zones, liquidity zones, BOS/CHoCH events
- `tradingagents/indicators/volume_profile.py` → `VolumeProfileAnalyzer.calculate_volume_profile(df)` returns:
  - `poc`, `value_area_high`, `value_area_low`, `poc_volume_pct`
  - `high_volume_nodes`, `low_volume_nodes` with price/volume
- `tradingagents/indicators/regime.py` → `RegimeDetector.get_full_regime()` returns:
  - `market_regime`, `volatility_regime`, `expansion_regime`
  - `squeeze_strength` (0-100), `breakout_ready` (bool)
- `web/backend/main.py:6015-6077` → RSI, MACD, ADX, ATR, BB, EMA calculations
  - Currently returns single latest values — need to vectorise into arrays for training

### Backtesting Layer — PARTIALLY READY
- `tradingagents/automation/auto_tuner.py`:
  - `_simulate_exit(direction, entry, sl, tp, high, low, close, entry_bar, max_hold)` → walk-forward exit sim
  - `_compute_stats(trades, strategy, timeframe, params)` → Sharpe, PF, WR, total PnL
  - `_compute_atr(high, low, close, period)` → vectorised ATR (Wilder's method)
  - Existing backtest functions for range, breakout, SMC, VP strategies

### Learning Layer — READY
- `tradingagents/learning/reward.py` → `RewardCalculator.calculate_reward()`
- `tradingagents/learning/pattern_analyzer.py` → `PatternAnalyzer.analyze_patterns()`
- `tradingagents/learning/trade_similarity.py` → `TradeSimilaritySearch`

---

## New Dependencies

Add to `pyproject.toml`:
```toml
xgboost = ">=2.0.0"
scikit-learn = ">=1.4.0"
optuna = ">=3.5.0"        # For hyperparameter tuning
```

---

## Architecture

```
tradingagents/
├── xgb_quant/                          # NEW — entire XGBoost strategy system
│   ├── __init__.py
│   ├── config.py                       # Strategy configs, feature windows, thresholds
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseFeatureSet abstract class
│   │   ├── technical.py                # Vectorised RSI, MACD, ADX, ATR, BB, EMA
│   │   ├── smc.py                      # SMC features (wraps existing SmartMoneyAnalyzer)
│   │   ├── volume_profile.py           # VP features (wraps existing VolumeProfileAnalyzer)
│   │   ├── regime.py                   # Regime features (wraps existing RegimeDetector)
│   │   └── composite.py               # Combines feature sets for a strategy
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseStrategy abstract class
│   │   ├── trend_following.py          # EMA crossovers, ADX, momentum, RSI
│   │   ├── mean_reversion.py           # BB%, Z-score, RSI extremes, Stochastic
│   │   ├── breakout.py                 # Donchian, squeeze, volume spike
│   │   ├── smc_zones.py               # OB/FVG strength, confluence, premium/discount
│   │   ├── volume_profile_strat.py     # POC proximity, value area position
│   │   └── regime_classifier.py        # Regime detection (used by selector, not traded directly)
│   ├── models/                         # Saved .json model files
│   │   ├── trend_following/
│   │   │   ├── XAUUSD_D1.json
│   │   │   └── GBPJPY_H4.json
│   │   ├── mean_reversion/
│   │   ├── breakout/
│   │   ├── smc_zones/
│   │   └── volume_profile/
│   ├── backtest.py                     # Walk-forward backtester (uses existing _simulate_exit)
│   ├── trainer.py                      # Train/retrain models per strategy per pair
│   ├── predictor.py                    # Live inference — get signal from trained model
│   ├── ensemble.py                     # Confluence/voting across strategies
│   ├── strategy_selector.py            # Score & rank strategies per pair
│   ├── parameter_tuner.py              # Optuna-based hyperparameter optimisation
│   └── scanner.py                      # Pair momentum scanner (from earlier discussion)
```

---

## Strategy Library

### Phase 1: Core Strategies (Build First)

#### 1. Trend Following
```
Hypothesis: Captures sustained directional moves
Features:
  - EMA crossovers (short/mid/long periods)
  - ADX (trend strength, +DI/-DI directional)
  - RSI (momentum confirmation, not just overbought/oversold)
  - Rate of Change (momentum)
  - ATR % (volatility-normalised movement)
  - Volume trend ratio (is volume confirming the move?)
  - Price position relative to EMA20/EMA50

Label: Next bar direction (1=up, 0=down)
Best for: XAUUSD, GBPJPY, EURJPY — trending, volatile pairs
```

#### 2. Mean Reversion
```
Hypothesis: Fades overextended price moves back to the mean
Features:
  - Bollinger Band % position (where in the bands)
  - Bollinger Band width (volatility state)
  - Z-score (standard deviations from mean)
  - RSI extremes (overbought/oversold)
  - Stochastic %K/%D
  - Close vs rolling mean distance
  - ATR % (is the extension unusual?)

Label: Next bar direction (1=up, 0=down)
Best for: EURGBP, AUDNZD, USDCHF — range-bound, low-volatility pairs
```

#### 3. Breakout
```
Hypothesis: Detects volatility compression and enters on expansion
Features:
  - Donchian channel position/width
  - Bar range vs ATR (is current bar expanding?)
  - ATR ratio (current vs 20-period avg — squeeze detection)
  - Volume spike (volume > 1.5x average)
  - Momentum (rate of change)
  - RSI (confirming directional bias)
  - Bollinger Band width (squeeze confirmation)
  - RegimeDetector squeeze_strength (0-100)
  - RegimeDetector breakout_ready (bool)

Label: Next bar direction (1=up, 0=down)
Best for: Pairs exiting consolidation, pre-news setups
```

### Phase 2: SMC-Enhanced Strategies (Leverage Existing Infrastructure)

#### 4. SMC Zone Strategy
```
Hypothesis: XGBoost learns which SMC zones actually hold and which fail
Features (from existing SmartMoneyAnalyzer):
  - Nearest bullish OB strength (0-1)
  - Nearest bearish OB strength (0-1)
  - Distance to nearest OB (in ATR units)
  - Nearest FVG fill % (0-100)
  - FVG remaining size (normalised)
  - Confluence total_score (0-100)
  - Confluence bullish_factors count
  - Confluence bearish_factors count
  - Premium/discount position_pct (0-100)
  - BOS count (recent, directional)
  - CHoCH detected (bool)
  - Liquidity sweep detected (bool)
  - Zone time decay factor (0-1)
  - With-trend alignment (bool → 1/0)
  - Higher TF aligned (bool → 1/0)

Label: Next bar direction (1=up, 0=down)
Why this matters: Your current SMC quant uses LLM to interpret zones.
XGBoost can learn from HUNDREDS of zone interactions which ones actually work,
without the cost/latency of an LLM call.
```

#### 5. Volume Profile Strategy
```
Hypothesis: Price reacts predictably at VP levels
Features (from existing VolumeProfileAnalyzer):
  - Distance to POC (in ATR units)
  - Distance to VAH (in ATR units)
  - Distance to VAL (in ATR units)
  - Price position relative to value area (above/inside/below → encoded)
  - POC volume % (how dominant is the POC)
  - Nearest HVN distance
  - Nearest LVN distance
  - Value area width (normalised)
  - Volume profile developing vs fixed

Label: Next bar direction (1=up, 0=down)
Best for: BTCUSD, indices — volume-heavy instruments
```

### Phase 3: Expanded Library (After Core Proves Out)

| Strategy | Hypothesis | Key Features |
|----------|-----------|--------------|
| Session Edge | Different sessions have directional bias | Hour of day, session (Asian/London/NY), open gaps |
| Candlestick Patterns | Pattern recognition without hard-coded rules | Body/wick ratios, inside bar, engulfing, pin bar features |
| Volatility Regime | Predict vol expansion/contraction | ATR ratios, BB width changes, historical vol percentile |
| Multi-Factor Composite | Everything combined | Top features from all strategies |
| SL/TP Optimiser | Predict optimal exit levels | Regression variant (predict magnitude, not just direction) |

---

## Feature Engineering Pipeline

### Base Feature Set (shared across all strategies)

```python
class BaseFeatureSet:
    """Vectorised features from OHLCV data — one row per candle"""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: OHLCV DataFrame (columns: open, high, low, close, volume)
        Output: DataFrame with feature columns, same index as input

        Common features added to ALL strategies:
          - atr_14: ATR(14) — reuse existing _compute_atr() from auto_tuner.py
          - atr_pct: ATR / close (volatility normalised)
          - rsi_14: RSI(14)
          - adx_14: ADX(14)
          - plus_di_14: +DI(14)
          - minus_di_14: -DI(14)
          - macd_hist: MACD histogram
          - bb_pct: Bollinger Band % position ((close - lower) / (upper - lower))
          - bb_width: Bollinger Band width (normalised)
          - ema_20: EMA(20) / close (normalised)
          - ema_50: EMA(50) / close (normalised)
          - volume_ratio: volume / 20-bar avg volume
          - returns_1: 1-bar return
          - returns_5: 5-bar return
          - returns_20: 20-bar return
          - hour: hour of day (for session-aware strategies)
          - day_of_week: 0-4

        All features normalised relative to price or ATR so they're
        comparable across symbols (XAUUSD ~3000 vs EURUSD ~1.08).
        """
```

### Strategy-Specific Feature Sets

Each strategy adds its own features ON TOP of the base set:

```python
class TrendFollowingFeatures(BaseFeatureSet):
    """Adds trend-specific features"""
    # EMA crossover signals, momentum, directional movement

class SMCZoneFeatures(BaseFeatureSet):
    """Adds SMC zone features by calling SmartMoneyAnalyzer"""
    # OB strengths, FVG fill%, confluence scores, premium/discount

class VolumeProfileFeatures(BaseFeatureSet):
    """Adds VP features by calling VolumeProfileAnalyzer"""
    # POC distance, VAH/VAL distance, value area position
```

### Vectorisation Approach

The existing indicators in `main.py` compute single values. We need arrays:

```python
def vectorise_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Return RSI for every row, not just the latest"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))
```

For SMC features, the challenge is that `analyze_full_smc()` analyses the full DataFrame and returns current state. For training data, we need to call it on each 200-bar window:

```python
def compute_smc_features_vectorised(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    For each bar i (from window onwards):
      - Run analyze_full_smc(df[i-window:i])
      - Extract numerical features
      - Store as row i in output DataFrame

    This is the expensive step — ~1 second per bar.
    For 1000 bars of D1 data = ~16 minutes. Acceptable for training.
    Cache results to disk.
    """
```

### Feature Caching

Training features are expensive to compute (especially SMC). Cache them:

```
tradingagents/xgb_quant/feature_cache/
├── XAUUSD_D1_technical.parquet    # Fast — recalculate weekly
├── XAUUSD_D1_smc.parquet         # Slow — cache aggressively
├── XAUUSD_D1_vp.parquet          # Medium — cache
├── GBPJPY_H4_technical.parquet
└── ...
```

Use parquet format — fast read/write, typed columns, small on disk.

---

## Label Engineering

### Primary Label: Next-Bar Direction

```python
def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    1 if next bar closes higher than current bar close
    0 if next bar closes lower or equal

    Note: We shift by -1 (look ahead 1 bar) during TRAINING only.
    During live prediction we never see the future.
    """
    return (df['close'].shift(-1) > df['close']).astype(int)
```

### Enhanced Labels (Phase 2)

Instead of just direction, predict whether a trade with specific SL/TP would WIN:

```python
def create_trade_labels(
    df: pd.DataFrame,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    max_hold: int = 20
) -> pd.Series:
    """
    For each bar, simulate a BUY trade:
      entry = close[i]
      sl = close[i] - atr[i] * sl_atr_mult
      tp = close[i] + atr[i] * tp_atr_mult
    Walk forward: did TP or SL get hit first within max_hold bars?

    1 = TP hit (win)
    0 = SL hit or timeout (loss)

    Uses existing _simulate_exit() from auto_tuner.py.
    """
```

This is a better label because it accounts for your actual trade mechanics, not just direction.

---

## Model Training

### Walk-Forward Training (No Look-Ahead)

```python
class WalkForwardTrainer:
    """
    Train window: 500 bars
    Test window: 100 bars
    Step: 100 bars (non-overlapping test windows)

    Fold 1: Train[0:500]    Test[500:600]
    Fold 2: Train[100:600]  Test[600:700]
    Fold 3: Train[200:700]  Test[700:800]
    ...

    For each fold:
      1. Compute features on train window
      2. Train XGBoost on train features + labels
      3. Predict on test window
      4. Record predictions + outcomes for evaluation
    """

    def train_and_evaluate(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        train_window: int = 500,
        test_window: int = 100,
    ) -> BacktestResult:
        # Load full OHLCV
        df = load_data(symbol, timeframe, lookback_bars=2000)

        # Compute features once (cached)
        features_df = strategy.compute_features(df)
        labels = create_trade_labels(df)

        all_predictions = []

        for fold_start in range(0, len(df) - train_window - test_window, test_window):
            train_end = fold_start + train_window
            test_end = train_end + test_window

            X_train = features_df.iloc[fold_start:train_end]
            y_train = labels.iloc[fold_start:train_end]
            X_test = features_df.iloc[train_end:test_end]
            y_test = labels.iloc[train_end:test_end]

            # Drop NaN rows (from indicator warmup)
            mask = X_train.notna().all(axis=1) & y_train.notna()
            X_train, y_train = X_train[mask], y_train[mask]

            # Train
            model = xgb.XGBClassifier(
                **strategy.default_params,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            # Predict probabilities
            probs = model.predict_proba(X_test)[:, 1]  # P(up)

            for i, (idx, prob) in enumerate(zip(X_test.index, probs)):
                all_predictions.append({
                    'bar_index': idx,
                    'prob_up': prob,
                    'actual': y_test.iloc[i] if i < len(y_test) else None,
                    'fold': fold_start
                })

        # Evaluate using existing _compute_stats() logic
        return self._evaluate_predictions(all_predictions, df, strategy)
```

### XGBoost Default Parameters (Per Strategy)

```python
STRATEGY_DEFAULTS = {
    "trend_following": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "min_child_weight": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "mean_reversion": {
        "max_depth": 3,         # Shallower — simpler patterns
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.7,
        "min_child_weight": 10, # More conservative
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,       # More regularisation
        "reg_lambda": 2.0,
    },
    "breakout": {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "min_child_weight": 3,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "smc_zones": {
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.8,
        "min_child_weight": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 1.5,
    },
    "volume_profile": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "min_child_weight": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
}
```

### Model Saving

```python
# Save as XGBoost native JSON (portable, no pickle security issues)
model.save_model(f"tradingagents/xgb_quant/models/{strategy}/{symbol}_{timeframe}.json")

# Load
model = xgb.XGBClassifier()
model.load_model(f"tradingagents/xgb_quant/models/{strategy}/{symbol}_{timeframe}.json")
```

---

## Ensemble / Confluence Logic

### Voting System

```python
class StrategyEnsemble:
    """Combines predictions from multiple strategies"""

    def __init__(self, strategies: List[BaseStrategy], min_agree: int = 2, min_prob: float = 0.60):
        self.strategies = strategies
        self.min_agree = min_agree
        self.min_prob = min_prob

    def predict(self, df: pd.DataFrame) -> Signal:
        """
        1. Each strategy predicts P(up) for latest bar
        2. Count how many predict UP (prob > min_prob) vs DOWN (prob < 1-min_prob)
        3. If >= min_agree strategies agree on direction, fire signal
        4. If no consensus, return HOLD

        Returns:
          Signal(direction="BUY"|"SELL"|"HOLD", confidence=avg_prob, strategies_agreed=[...])
        """
        votes_up = []
        votes_down = []

        for strategy in self.strategies:
            features = strategy.compute_features(df)
            prob_up = strategy.model.predict_proba(features.iloc[[-1]])[:, 1][0]

            if prob_up >= self.min_prob:
                votes_up.append((strategy.name, prob_up))
            elif prob_up <= (1 - self.min_prob):
                votes_down.append((strategy.name, 1 - prob_up))

        if len(votes_up) >= self.min_agree:
            avg_conf = np.mean([v[1] for v in votes_up])
            return Signal("BUY", avg_conf, [v[0] for v in votes_up])

        if len(votes_down) >= self.min_agree:
            avg_conf = np.mean([v[1] for v in votes_down])
            return Signal("SELL", avg_conf, [v[0] for v in votes_down])

        return Signal("HOLD", 0.0, [])
```

### Weighted Voting (Phase 2)

Instead of equal votes, weight by each strategy's historical accuracy on this pair:

```python
# Strategy selector provides weights based on backtest performance
weights = {
    "trend_following": 0.45,   # Best performer on XAUUSD
    "smc_zones": 0.35,         # Good complement
    "breakout": 0.20,          # Weaker but adds diversity
}

weighted_prob = sum(prob[s] * weights[s] for s in strategies) / sum(weights.values())
```

---

## Pair Scanner (From Earlier Discussion)

Scans watchlist for pairs on the move. No changes from the earlier plan — pure price data, no ML needed:

```python
class PairScanner:
    """
    Scans 16+ pairs every 4 hours:
      - ATR expansion (current vs 20-period avg)
      - ADX strength
      - Directional move % over recent bars
      - Structure break (new swing high/low)
      - EMA alignment
      - Volume confirmation
      - Spread cost ratio (spread / ATR)

    Returns ranked shortlist with direction bias.
    Disqualifies: choppy (ADX<20), expensive (spread>15% ATR), already positioned.
    """
```

### Watchlist
Majors: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
Crosses: EURJPY, GBPJPY, EURGBP, AUDNZD, CADJPY, EURAUD
Metals: XAUUSD, XAGUSD
Crypto: BTCUSD, ETHUSD

---

## Strategy Selector

### How It Works

Given a pair from the scanner, score each strategy in the library and pick the best combination:

```python
class StrategySelector:
    """
    Inputs:
      - symbol: which pair
      - direction: LONG or SHORT (from scanner)
      - current_regime: trending/ranging (from RegimeDetector)
      - current_volatility: low/normal/high/extreme

    Scoring per strategy:
      1. Walk-forward backtest metrics for this pair (Sharpe, PF, WR) — 40%
      2. Regime suitability (trend_following scores high in trending) — 25%
      3. Recent performance (last 30 days weighted 2x vs last 180 days) — 20%
      4. Signal correlation with other selected strategies (want low) — 15%

    Output:
      - Ranked list of strategies
      - Recommended ensemble (top 3 with lowest correlation)
      - Per-strategy confidence
    """

    def select(self, symbol: str, direction: str, regime: str, volatility: str) -> SelectionResult:
        scores = {}

        for strategy_name, strategy in self.library.items():
            # 1. Backtest performance on this pair
            bt = self.get_backtest_result(strategy_name, symbol)
            if bt is None:
                bt_score = 0.3  # Cold start default
            else:
                bt_score = (
                    bt.win_rate * 0.4 +
                    min(bt.profit_factor / 2, 1.0) * 0.3 +
                    min(bt.sharpe / 2, 1.0) * 0.3
                )

            # 2. Regime suitability
            regime_score = REGIME_SUITABILITY[strategy_name].get(regime, 0.5)

            # 3. Recency weighting
            recent = self.get_recent_performance(strategy_name, symbol, days=30)
            older = self.get_recent_performance(strategy_name, symbol, days=180)
            recency_score = recent * 0.67 + older * 0.33 if recent else bt_score

            # 4. Correlation (computed after initial ranking)
            # Deferred to ensemble selection step

            scores[strategy_name] = bt_score * 0.40 + regime_score * 0.25 + recency_score * 0.20

        # Rank and select top 3-5 with minimum correlation
        return self._select_ensemble(scores, symbol)


# Regime suitability scores (human knowledge as prior)
REGIME_SUITABILITY = {
    "trend_following":  {"trending-up": 0.9, "trending-down": 0.9, "ranging": 0.2},
    "mean_reversion":   {"trending-up": 0.2, "trending-down": 0.2, "ranging": 0.9},
    "breakout":         {"trending-up": 0.6, "trending-down": 0.6, "ranging": 0.7},
    "smc_zones":        {"trending-up": 0.8, "trending-down": 0.8, "ranging": 0.5},
    "volume_profile":   {"trending-up": 0.5, "trending-down": 0.5, "ranging": 0.8},
}
```

---

## Parameter Tuner

### Per-Pair Optimisation with Optuna

```python
class ParameterTuner:
    """
    At deployment time, fine-tune XGBoost hyperparameters + feature windows
    for a specific pair using Optuna Bayesian optimisation.

    Runs walk-forward backtest as the objective function.
    Optimises for Sharpe ratio (balances return and risk).
    """

    def tune(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        n_trials: int = 50,
        train_bars: int = 1000,
    ) -> TuneResult:

        def objective(trial):
            # XGBoost hyperparameters
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            }

            # Feature window sizes (strategy-specific)
            feature_params = strategy.suggest_feature_params(trial)
            # e.g. SHORT_WIN: 5-20, MID_WIN: 15-50, LONG_WIN: 50-200

            # Risk parameters
            risk_params = {
                "sl_atr_mult": trial.suggest_float("sl_atr_mult", 1.0, 3.0),
                "tp_atr_mult": trial.suggest_float("tp_atr_mult", 1.5, 4.0),
                "signal_threshold": trial.suggest_float("signal_threshold", 0.55, 0.75),
            }

            # Walk-forward backtest with these params
            result = self.trainer.train_and_evaluate(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                xgb_params=params,
                feature_params=feature_params,
                risk_params=risk_params,
            )

            # Optimise for Sharpe (or composite metric)
            return result.sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return TuneResult(
            best_params=study.best_params,
            best_sharpe=study.best_value,
            study=study,
        )
```

### Retuning Schedule

- **Initial**: Full tune (50 trials) when first deploying on a pair
- **Monthly**: Retune with updated data (30 trials, warm-started from previous best)
- **On regime change**: If regime detector flags a shift, trigger retune

---

## Integration into Existing Automation

### New Pipeline Type

```python
# In quant_automation.py PipelineType enum:
XGBOOST = "xgboost"
XGBOOST_ENSEMBLE = "xgboost_ensemble"
```

### Analysis Endpoint

```python
# New endpoint in web/backend/main.py:
@app.post("/api/analysis/xgboost")
async def xgboost_analysis(request: AnalysisRequest):
    """
    1. Load OHLCV for symbol/timeframe
    2. Load trained models for the requested strategy (or ensemble)
    3. Compute features
    4. Predict
    5. Return signal with confidence, same AnalysisCycleResult format

    No LLM call. Pure XGBoost inference. Sub-second response time.
    """
```

### Quant Automation Dispatch

```python
# In quant_automation.py main analysis loop:
elif pipeline == PipelineType.XGBOOST:
    result = await self._run_xgboost_analysis(symbol)
elif pipeline == PipelineType.XGBOOST_ENSEMBLE:
    result = await self._run_xgboost_ensemble_analysis(symbol)
```

### Scanner Automation (New Class)

```python
class ScannerAutomation:
    """
    Separate from QuantAutomation. Runs on a schedule:

    1. Scanner scans watchlist → shortlisted pairs with direction
    2. For top candidate:
       a. Strategy selector picks best ensemble for that pair
       b. Load/train models if not cached
       c. Run ensemble prediction
       d. If signal passes confidence threshold → execute trade
    3. Sleep until next scan interval
    """
```

---

## Execution & Risk Management

Uses existing infrastructure — nothing changes:

```
XGBoost signal (BUY/SELL, confidence, entry/SL/TP)
    │
    ▼
Existing _execute_trade() in quant_automation.py
    ├── Confidence check (min_confidence)
    ├── Position count limits
    ├── Guardrails (circuit breaker)
    ├── Price validation (live market price)
    ├── SL/TP side validation
    ├── Position sizing (lot step, volume limits)
    └── MT5 order execution
```

SL/TP calculated from XGBoost risk parameters:
- **Stop Loss**: `entry ± (ATR × sl_atr_mult)` — direction-aware
- **Take Profit**: `entry ± (ATR × tp_atr_mult)` — direction-aware
- **Trailing Stop**: Enabled, uses existing `trailing_stop_atr_multiplier` from config

---

## Performance Matrix

Every trained model produces a `BacktestResult`. Stored as JSON:

```
tradingagents/xgb_quant/results/
├── performance_matrix.json     # Summary: pair × strategy → metrics
├── XAUUSD/
│   ├── trend_following_D1.json
│   ├── mean_reversion_D1.json
│   ├── breakout_D1.json
│   ├── smc_zones_D1.json
│   └── volume_profile_D1.json
├── GBPJPY/
│   └── ...
└── EURUSD/
    └── ...
```

```json
// performance_matrix.json
{
  "XAUUSD": {
    "trend_following": {"win_rate": 0.58, "profit_factor": 1.9, "sharpe": 1.2, "trades": 87},
    "mean_reversion": {"win_rate": 0.45, "profit_factor": 0.8, "sharpe": -0.3, "trades": 62},
    "smc_zones":      {"win_rate": 0.55, "profit_factor": 1.5, "sharpe": 0.9, "trades": 74}
  },
  "EURGBP": {
    "trend_following": {"win_rate": 0.42, "profit_factor": 0.7, "sharpe": -0.5, "trades": 55},
    "mean_reversion": {"win_rate": 0.61, "profit_factor": 2.1, "sharpe": 1.5, "trades": 48}
  }
}
```

This is the table that the strategy selector reads to make decisions.

---

## Build Order

### Step 1: Feature Pipeline
- `features/technical.py` — vectorise RSI, MACD, ADX, ATR, BB, EMA
- `features/base.py` — BaseFeatureSet with common features
- `features/smc.py` — wrap SmartMoneyAnalyzer for vectorised SMC features
- `features/volume_profile.py` — wrap VolumeProfileAnalyzer

**Test**: Generate feature DataFrame for XAUUSD D1, verify no NaN leaks, verify no look-ahead.

### Step 2: Core Strategies + Trainer
- `strategies/base.py` — BaseStrategy abstract class
- `strategies/trend_following.py` — first strategy
- `strategies/mean_reversion.py`
- `strategies/breakout.py`
- `trainer.py` — walk-forward train + evaluate
- `backtest.py` — uses existing `_simulate_exit()`

**Test**: Train trend_following on XAUUSD D1. Walk-forward backtest. Print Sharpe, WR, PF.

### Step 3: Ensemble + Live Predictor
- `ensemble.py` — voting/weighted combination
- `predictor.py` — load models, compute features, predict
- New endpoint `/api/analysis/xgboost`
- Wire into `quant_automation.py` as new PipelineType

**Test**: Run live prediction on XAUUSD. Verify signal matches backtest logic.

### Step 4: Scanner
- `scanner.py` — pair momentum scoring
- Watchlist config
- Integration as `ScannerAutomation` or loop within existing automation

**Test**: Scan all 16 pairs. Verify scoring. Check spread filter works.

### Step 5: Strategy Selector + Parameter Tuner
- `strategy_selector.py` — rank strategies per pair
- `parameter_tuner.py` — Optuna integration
- Performance matrix generation

**Test**: Select best strategy for GBPJPY. Tune parameters. Compare tuned vs default.

### Step 6: SMC + VP Strategies
- `strategies/smc_zones.py` — XGBoost on SMC features
- `strategies/volume_profile_strat.py` — XGBoost on VP features

**Test**: Compare XGBoost SMC strategy vs existing LLM-based SMC Quant on same data.

### Step 7: Web UI
- Scanner dashboard (watchlist heatmap, momentum scores)
- Performance matrix table
- Strategy selector results
- Model training status / retrain trigger
- Feature importance charts

---

## What This Gives You vs Current System

| | Current LLM Quants | XGBoost Strategies |
|---|---|---|
| **Cost per analysis** | ~$0.02-0.10 (LLM call) | ~$0.00 (local inference) |
| **Latency** | 5-30 seconds | <100ms |
| **Learns from data** | Via memory/reflection (qualitative) | Via training (quantitative) |
| **Adapts to pair** | Same prompt, different data | Different model per pair |
| **Backtestable** | Expensive (need LLM per bar) | Fast (train once, walk-forward) |
| **Explainable** | LLM rationale text | Feature importance scores |
| **Handles 16 pairs** | 16 × $0.05 per cycle = expensive | 16 × 100ms = 1.6 seconds |

They complement each other:
- **LLM quants** understand context, news, narrative — good for novel situations
- **XGBoost** finds statistical patterns in historical data — good for repeatable setups

The scanner + selector can choose EITHER an LLM quant OR an XGBoost strategy for a given pair, based on which has the better track record.
