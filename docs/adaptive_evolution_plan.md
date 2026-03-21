# Adaptive Evolution System — Implementation Plan

## The Problem

You're trading 2 symbols (XAUUSD, BTCUSD) with 10 automation instances across 6 pipeline types. 130 decisions, 122 closed, 6 wins, 28 losses — the rest closed without clear outcome tracking. Pipeline names are inconsistent across decisions making it hard to know what's actually working. The system doesn't know which strategy fits which pair, doesn't scan for opportunities beyond what's configured, and can't adapt its own parameters.

Forex can be won, but it requires a system that finds the right opportunities, applies the right strategy, and evolves.

## The Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE EVOLUTION SYSTEM                     │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │  SCANNER  │───▶│  MATCHER  │───▶│ EXECUTOR │───▶│ EVOLVER  │ │
│  │           │    │           │    │          │    │          │ │
│  │ Find pairs│    │ Pick best │    │ Run quant│    │ Breed    │ │
│  │ on the    │    │ strategy  │    │ pipeline │    │ winners, │ │
│  │ move      │    │ for pair  │    │ + trade  │    │ kill     │ │
│  │           │    │           │    │          │    │ losers   │ │
│  └──────────┘    └───────────┘    └──────────┘    └──────────┘ │
│       ▲                                                │        │
│       └────────────────────────────────────────────────┘        │
│                     feedback loop                               │
└─────────────────────────────────────────────────────────────────┘
```

Four components, built in order, each useful on its own:

1. **Backtester** — replay history through strategies, build the performance matrix
2. **Scanner** — find pairs on the move from a watchlist
3. **Matcher** — pick the best quant for a given pair based on data
4. **Evolver** — genetic algorithm that breeds strategy parameter sets

---

## Phase 1: Simplified Backtester

### Why First

Everything depends on having performance data per pair per strategy. You have 130 decisions but pipeline names are fragmented (`btc_smc_lmm`, `quant_automation_quant`, `web_ui`, etc). The backtester gives you clean, comparable data across all pairs and strategies.

### What It Does

Walks forward through historical candles and simulates what each pipeline would have done:

```
For each pair in watchlist:
  For each pipeline (smc_quant, volume_profile, breakout, range, smc_mtf, rule_based):
    For each candle window (sliding 200-candle blocks):
      1. Run SMC analysis on the candle data (rule-based, no LLM)
      2. Run technical indicators (ATR, RSI, ADX, MACD, BB)
      3. Apply strategy rules to generate signal (BUY/SELL/HOLD)
      4. If signal: record entry, SL, TP from strategy rules
      5. Walk forward to find outcome (TP hit, SL hit, or timeout)
      6. Record result in same format as trade_decisions
```

### Two Tiers

**Tier 1: Rule-Based Backtest (fast, free, build first)**

No LLM calls. Each pipeline becomes a set of rules:

```python
class StrategyRules:
    """Rule-based approximation of a pipeline's logic"""

    def evaluate(self, market_data: MarketSnapshot) -> Optional[Signal]:
        """
        market_data contains:
          - ohlcv: 200 candles
          - smc: OrderBlocks, FVGs, BOS/CHoCH, liquidity zones
          - indicators: ATR, RSI, ADX, MACD, BB, EMA20/50
          - volume_profile: POC, VAH, VAL, HVN, LVN
          - regime: trending-up/down/ranging, volatility level

        Returns Signal(direction, entry, sl, tp, confidence) or None
        """
```

Example rule sets per pipeline:

| Pipeline | Core Rule | Entry | SL | TP |
|----------|-----------|-------|----|----|
| smc_quant | OB or FVG zone + trend alignment + confluence >= 2 | Zone edge | Below zone (BUY) / above zone (SELL) | Next liquidity target |
| volume_profile | Price at VAL/VAH/POC + volume confirmation | VP level | Beyond VP level + ATR buffer | Opposite VP level |
| breakout | BOS confirmed + ADX > 25 + volume spike | Break level + buffer | Below structure | 2x ATR or next resistance |
| range | RSI oversold/overbought + price at BB band + ADX < 25 | BB band | Beyond recent swing | Opposite BB band |
| smc_mtf | HTF trend + LTF entry zone + multi-TF confluence | LTF zone within HTF bias | Below LTF zone | HTF target |
| rule_based | RSI + MACD + BB crossover + trend filter | At signal candle close | ATR-based | ATR-based with RR target |

**Tier 2: LLM-Validated Backtest (slow, expensive, for top candidates only)**

Take the top 10% of rule-based performers. Run them through the actual LLM pipeline on a sample of their trades. Validates that the LLM would have agreed with the rule-based signal. This is a calibration check, not a full backtest.

### Data Source

MT5 `copy_rates_from_pos` already works for historical data. Pull 6-12 months of H1/H4/D1 candles per pair:

```python
# Already available in mt5_data.py
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
# count=4380 for 6 months of H1 candles
# count=1095 for 6 months of H4 candles
# count=130 for 6 months of D1 candles
```

### Output: Performance Matrix

```python
@dataclass
class BacktestResult:
    symbol: str
    pipeline: str
    timeframe: str
    strategy_params: dict          # The parameter set used
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_rr: float
    profit_factor: float           # gross_profit / gross_loss
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_duration_candles: int
    best_regime: str               # Which regime this strategy does best in
    worst_regime: str              # Which regime to avoid
    fitness_score: float           # Composite score for ranking
```

Stored as JSON in `examples/backtest_results/` — one file per pair/pipeline/params combo.

### New Files

```
tradingagents/evolution/
├── __init__.py
├── backtester.py          # Walk-forward backtester engine
├── strategy_rules.py      # Rule-based strategy approximations
├── market_snapshot.py     # Data container for analysis inputs
└── backtest_results.py    # Result storage and querying
```

### Key Design Decisions

- **Walk-forward, not look-ahead**: The backtester only sees candles up to the current point. No future data leaks.
- **Realistic execution**: Add 1 candle delay for entry (signal on close, enter on next open). Account for spread.
- **Timeout**: If neither SL nor TP hit within 50 candles (configurable), close at market — records as timeout.
- **Same outcome analysis**: Uses existing `analyze_trade_outcome()` logic for consistent comparison with live trades.

---

## Phase 2: Pair Scanner

### What It Does

Scans a watchlist of 15-25 forex pairs on a schedule. Scores each pair on movement/momentum. Returns a ranked shortlist.

### Scoring Model

All from price data — no LLM needed:

```python
@dataclass
class PairScore:
    symbol: str
    direction: str              # "LONG" or "SHORT"
    momentum_score: float       # 0-100 composite

    # Components
    atr_expansion: float        # Current ATR / 20-period avg ATR (>1.2 = expanding)
    adx_strength: float         # ADX value (>25 = trending)
    directional_move_pct: float # % move in one direction over last N candles
    structure_break: bool       # Broke recent swing high/low
    ema_alignment: bool         # EMA20 > EMA50 (bullish) or EMA20 < EMA50 (bearish)
    volume_confirmation: bool   # Volume above average during move
    spread_cost_ratio: float    # Spread / ATR (<0.1 = affordable to trade)

    # Filters (disqualifiers)
    is_choppy: bool             # ADX < 20 and no clear direction
    spread_too_wide: bool       # Spread > 15% of ATR
    already_has_position: bool  # We already hold this pair
```

### Scoring Formula

```python
def calculate_momentum_score(data: PairScore) -> float:
    score = 0

    # ATR expansion (0-25 points)
    # Expanding volatility = pair is waking up
    if data.atr_expansion > 1.5:
        score += 25
    elif data.atr_expansion > 1.2:
        score += 15
    elif data.atr_expansion > 1.0:
        score += 5

    # ADX trend strength (0-25 points)
    if data.adx_strength > 40:
        score += 25
    elif data.adx_strength > 30:
        score += 20
    elif data.adx_strength > 25:
        score += 15
    elif data.adx_strength > 20:
        score += 5

    # Directional move (0-20 points)
    # How much has price moved in one direction recently
    if abs(data.directional_move_pct) > 1.5:
        score += 20
    elif abs(data.directional_move_pct) > 1.0:
        score += 15
    elif abs(data.directional_move_pct) > 0.5:
        score += 10

    # Structure break (0-15 points)
    if data.structure_break:
        score += 15

    # EMA alignment (0-10 points)
    if data.ema_alignment:
        score += 10

    # Volume confirmation (0-5 points)
    if data.volume_confirmation:
        score += 5

    return score  # 0-100
```

### Disqualifiers

A pair is removed from the shortlist if ANY of these are true:
- `spread_cost_ratio > 0.15` — too expensive to trade
- `is_choppy` — no clear direction, ADX < 20
- `already_has_position` — already trading this pair
- `momentum_score < 40` — not enough conviction

### Watchlist

Start with these (available on most MT5 brokers):

**Majors** (tight spreads, high liquidity):
EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD

**Crosses** (wider spreads but bigger moves):
EURJPY, GBPJPY, EURGBP, AUDNZD, CADJPY, EURAUD

**Metals** (existing):
XAUUSD, XAGUSD

**Crypto** (24/7):
BTCUSD, ETHUSD

**Total**: ~16 pairs. Scanner runs in <5 seconds (just MT5 data pulls, no LLM).

### Schedule

- **Every 4 hours** on H4 timeframe (swing trading focus)
- Or **every 1 hour** on H1 for more active trading
- Configurable per-instance

### Output

```python
@dataclass
class ScanResult:
    timestamp: str
    watchlist_size: int
    shortlist: List[PairScore]    # Top 3-5 ranked by momentum_score
    disqualified: List[str]       # Pairs that failed filters
    best_candidate: PairScore     # #1 pick
```

### New Files

```
tradingagents/evolution/
├── scanner.py              # Pair scanner
├── watchlist.py            # Watchlist management
```

### Integration

The scanner is a new loop inside quant_automation, or a standalone automation type:

```python
class ScannerAutomation:
    """Scans watchlist, picks best pair, delegates to best quant"""

    async def _scan_loop(self):
        while self.running:
            # 1. Scan all pairs
            scan_result = self.scanner.scan(self.watchlist, timeframe="H4")

            # 2. Pick best candidate
            if scan_result.best_candidate and scan_result.best_candidate.momentum_score >= 50:
                pair = scan_result.best_candidate

                # 3. Find best strategy for this pair (Phase 3)
                best_pipeline = self.matcher.get_best_pipeline(pair.symbol, pair.direction)

                # 4. Run that pipeline's analysis
                result = await self._run_analysis(pair.symbol, best_pipeline, bias=pair.direction)

                # 5. Execute if confirmed
                if result.signal != "HOLD" and result.confidence >= self.config.min_confidence:
                    await self._execute_trade(result)

            await asyncio.sleep(self.config.scan_interval_seconds)
```

---

## Phase 3: Strategy Matcher

### What It Does

Given a pair and direction, returns the best pipeline to use — based on the performance matrix from backtesting + live trading.

### Data Sources (priority order)

1. **Live trade data** — from `examples/trade_decisions/` (highest trust, but sparse)
2. **Backtest data** — from Phase 1 (lower trust, but comprehensive)

Live data weight increases over time as more trades accumulate:

```python
def get_data_weight(live_count: int) -> tuple[float, float]:
    """Returns (live_weight, backtest_weight)"""
    if live_count >= 30:
        return (0.8, 0.2)    # Enough live data to mostly trust it
    elif live_count >= 15:
        return (0.6, 0.4)    # Decent sample, blend
    elif live_count >= 5:
        return (0.4, 0.6)    # Still lean on backtest
    else:
        return (0.1, 0.9)    # Almost entirely backtest
```

### Matching Logic

```python
class StrategyMatcher:

    def get_best_pipeline(
        self,
        symbol: str,
        direction: str,
        current_regime: str = None,
        current_volatility: str = None
    ) -> MatchResult:
        """
        Returns best pipeline for this symbol + context.

        MatchResult contains:
          - pipeline: PipelineType
          - confidence: float (how confident we are in this recommendation)
          - reasoning: str
          - performance: BacktestResult (the data behind the pick)
          - fallback: PipelineType (second choice if primary unavailable)
        """

        candidates = []

        for pipeline in PipelineType:
            # Get blended performance
            live_perf = self._get_live_performance(symbol, pipeline)
            backtest_perf = self._get_backtest_performance(symbol, pipeline)

            live_w, bt_w = get_data_weight(live_perf.total_trades)

            blended_wr = live_perf.win_rate * live_w + backtest_perf.win_rate * bt_w
            blended_rr = live_perf.avg_rr * live_w + backtest_perf.avg_rr * bt_w
            blended_pf = live_perf.profit_factor * live_w + backtest_perf.profit_factor * bt_w

            # Regime filter: penalize if current regime is this strategy's worst
            regime_penalty = 1.0
            if current_regime and backtest_perf.worst_regime == current_regime:
                regime_penalty = 0.7
            elif current_regime and backtest_perf.best_regime == current_regime:
                regime_penalty = 1.2  # bonus

            # Composite score
            score = (
                blended_wr * 0.35 +           # Win rate matters most
                min(blended_rr / 3.0, 1.0) * 0.25 +  # RR capped at 3
                min(blended_pf / 2.0, 1.0) * 0.20 +  # Profit factor capped at 2
                (1 - backtest_perf.max_drawdown_pct) * 0.20  # Lower DD = better
            ) * regime_penalty

            candidates.append((pipeline, score, blended_wr, blended_rr))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        fallback = candidates[1] if len(candidates) > 1 else None

        return MatchResult(
            pipeline=best[0],
            confidence=best[1],
            win_rate=best[2],
            avg_rr=best[3],
            fallback=fallback[0] if fallback else None
        )
```

### Cold Start Defaults

When no data exists for a pair (brand new to the watchlist):

```python
PAIR_CHARACTERISTIC_DEFAULTS = {
    # Trending volatile pairs → SMC works well (structure-based)
    "XAUUSD": PipelineType.SMC_QUANT,
    "GBPJPY": PipelineType.SMC_QUANT,
    "EURJPY": PipelineType.SMC_QUANT,

    # Volume-heavy pairs → VP gives edge
    "BTCUSD": PipelineType.VOLUME_PROFILE,
    "ETHUSD": PipelineType.VOLUME_PROFILE,

    # Range-bound pairs → Range strategy
    "EURGBP": PipelineType.RANGE_QUANT,
    "AUDNZD": PipelineType.RANGE_QUANT,
    "USDCHF": PipelineType.RANGE_QUANT,

    # High-momentum pairs → Breakout
    "GBPUSD": PipelineType.BREAKOUT_QUANT,

    # Unknown → SMC+VP confirmation (safest, dual agreement)
    "_default": PipelineType.SMC_QUANT,  # or SMC_VP_CONFIRMATION when built
}
```

These are just starting points. The matcher overwrites them as data accumulates.

### New Files

```
tradingagents/evolution/
├── matcher.py              # Strategy matcher
├── performance_matrix.py   # Aggregates live + backtest data
```

---

## Phase 4: Genetic Evolver

### What It Does

Takes strategy parameters and evolves them. Instead of one fixed `min_confidence=0.65` for all pairs, the evolver discovers that GBPJPY works best with `min_confidence=0.55` (catch more moves) while EURGBP needs `min_confidence=0.75` (be very selective in ranges).

### Strategy Genome

A strategy is NOT code — it's a parameter dictionary:

```python
@dataclass
class StrategyGenome:
    """A complete, evolvable strategy configuration"""

    # Identity
    genome_id: str                      # Unique ID
    parent_ids: List[str]               # Parents (for lineage tracking)
    generation: int                     # Which generation
    created_at: str

    # Pipeline selection gene
    pipeline: str                       # "smc_quant", "volume_profile", etc.
    timeframe: str                      # "H1", "H4", "D1"

    # Entry filter genes
    min_confidence: float               # 0.5 - 0.85 (step 0.05)
    min_adx: float                      # 15 - 40 (trend strength filter)
    min_atr_expansion: float            # 0.8 - 2.0 (volatility filter)
    require_trend_alignment: bool       # Must trade with higher TF trend
    min_confluence_factors: int         # 1 - 4 (how many factors must align)

    # SMC-specific genes (only active for SMC pipelines)
    min_zone_strength: float            # 0.3 - 0.9 (zone quality filter)
    ob_volume_multiplier: float         # 1.0 - 2.0 (OB detection sensitivity)
    fvg_min_size_atr: float             # 0.2 - 0.6 (FVG size filter)
    confluence_ob_weight: int           # 15 - 40 (importance of OBs)
    confluence_fvg_weight: int          # 10 - 30 (importance of FVGs)
    confluence_liquidity_weight: int    # 15 - 35 (importance of liquidity)
    regime_trending_boost: float        # 1.0 - 1.5 (trend alignment boost)
    regime_counter_penalty: float       # 0.5 - 0.9 (counter-trend penalty)
    time_decay_fresh_threshold: int     # 10 - 30 (candles before zone ages)

    # VP-specific genes (only active for VP pipelines)
    vp_value_area_pct: float            # 0.60 - 0.80
    vp_hvn_threshold: float             # 1.2 - 2.0
    vp_lvn_threshold: float             # 0.3 - 0.7

    # Risk management genes
    sl_method: str                      # "zone_edge", "atr_multiple", "structure"
    sl_atr_multiplier: float            # 1.0 - 3.0
    tp_method: str                      # "rr_ratio", "next_zone", "atr_multiple"
    tp_rr_target: float                 # 1.5 - 4.0
    trailing_stop_atr_mult: float       # 1.0 - 3.0
    move_to_breakeven_atr_mult: float   # 1.0 - 2.5
    max_risk_per_trade_pct: float       # 0.5 - 2.0

    # Session filter genes
    allowed_sessions: List[str]         # ["asian", "london", "newyork"]

    # Fitness (calculated, not evolved)
    fitness: Optional[FitnessScore] = None
    symbol_fitness: Dict[str, FitnessScore] = field(default_factory=dict)


@dataclass
class FitnessScore:
    """Multi-factor fitness evaluation"""
    win_rate: float
    avg_rr: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    composite: float                    # The single number used for ranking

    @staticmethod
    def calculate(results: List[BacktestResult]) -> 'FitnessScore':
        """
        composite = (
            win_rate * 0.25 +
            min(avg_rr / 3, 1.0) * 0.20 +
            min(profit_factor / 2, 1.0) * 0.20 +
            sharpe_ratio_normalized * 0.15 +
            (1 - max_drawdown_pct) * 0.15 +
            min(total_trades / 50, 1.0) * 0.05  # Prefer strategies that trade enough
        )
        """
```

### Gene Ranges & Constraints

```python
GENE_RANGES = {
    "min_confidence":           (0.50, 0.85, 0.05),   # (min, max, step)
    "min_adx":                  (15, 40, 5),
    "min_atr_expansion":        (0.8, 2.0, 0.1),
    "require_trend_alignment":  [True, False],          # Categorical
    "min_confluence_factors":   (1, 4, 1),
    "min_zone_strength":        (0.3, 0.9, 0.1),
    "ob_volume_multiplier":     (1.0, 2.0, 0.1),
    "fvg_min_size_atr":         (0.2, 0.6, 0.05),
    "sl_method":                ["zone_edge", "atr_multiple", "structure"],
    "sl_atr_multiplier":        (1.0, 3.0, 0.25),
    "tp_method":                ["rr_ratio", "next_zone", "atr_multiple"],
    "tp_rr_target":             (1.5, 4.0, 0.25),
    "trailing_stop_atr_mult":   (1.0, 3.0, 0.25),
    "max_risk_per_trade_pct":   (0.5, 2.0, 0.25),
    "allowed_sessions":         [["london", "newyork"], ["london"], ["newyork"],
                                 ["asian", "london", "newyork"]],
}
```

### Genetic Operations

**Population**: 50 genomes per generation (configurable).

**Selection (top 20%)**:
```python
def select(population: List[StrategyGenome], top_pct: float = 0.20) -> List[StrategyGenome]:
    """Tournament selection — top 20% survive"""
    sorted_pop = sorted(population, key=lambda g: g.fitness.composite, reverse=True)
    n_survivors = max(2, int(len(sorted_pop) * top_pct))
    return sorted_pop[:n_survivors]
```

**Crossover (uniform)**:
```python
def crossover(parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
    """For each gene, randomly pick from parent A or B"""
    child = StrategyGenome(
        genome_id=generate_id(),
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
        generation=max(parent_a.generation, parent_b.generation) + 1,
    )

    for gene_name in GENE_RANGES:
        # 50/50 chance of inheriting from either parent
        source = random.choice([parent_a, parent_b])
        setattr(child, gene_name, getattr(source, gene_name))

    return child
```

**Mutation (per-gene)**:
```python
def mutate(genome: StrategyGenome, mutation_rate: float = 0.15) -> StrategyGenome:
    """Each gene has mutation_rate chance of being nudged"""
    mutated = copy.deepcopy(genome)

    for gene_name, gene_range in GENE_RANGES.items():
        if random.random() > mutation_rate:
            continue

        if isinstance(gene_range, list):
            # Categorical: random choice
            setattr(mutated, gene_name, random.choice(gene_range))
        else:
            min_val, max_val, step = gene_range
            current = getattr(mutated, gene_name)
            # Nudge by ±1-2 steps
            nudge = random.choice([-2, -1, 1, 2]) * step
            new_val = max(min_val, min(max_val, current + nudge))
            setattr(mutated, gene_name, round(new_val, 4))

    return mutated
```

**Elitism**: Top 2 genomes always survive unchanged (prevents losing the best).

### Evolution Cycle

```python
class GeneticEvolver:

    def run_generation(self, population: List[StrategyGenome]) -> List[StrategyGenome]:
        """One generation cycle"""

        # 1. Evaluate fitness (backtest each genome)
        for genome in population:
            if genome.fitness is None:  # Only evaluate new/changed genomes
                results = self.backtester.run(genome)
                genome.fitness = FitnessScore.calculate(results)

        # 2. Select survivors
        survivors = self.select(population)

        # 3. Elitism — preserve top 2 unchanged
        elite = survivors[:2]

        # 4. Breed new population
        new_population = list(elite)

        while len(new_population) < self.population_size:
            # Pick two parents
            parent_a = random.choice(survivors)
            parent_b = random.choice(survivors)

            # Crossover
            child = self.crossover(parent_a, parent_b)

            # Mutate
            child = self.mutate(child)

            new_population.append(child)

        return new_population

    def evolve(self, n_generations: int = 20):
        """Run full evolution"""
        population = self.initialize_population()

        for gen in range(n_generations):
            population = self.run_generation(population)

            best = max(population, key=lambda g: g.fitness.composite)
            logger.info(
                f"Gen {gen}: Best fitness={best.fitness.composite:.3f} "
                f"WR={best.fitness.win_rate:.1%} RR={best.fitness.avg_rr:.2f} "
                f"PF={best.fitness.profit_factor:.2f}"
            )

        return population
```

### Initial Population Seeding

Don't start random — seed with your current configs:

```python
def initialize_population(self) -> List[StrategyGenome]:
    """Seed with current configs + random mutations of them"""
    population = []

    # Seed 1: Current XAUUSD SMC Quant config (what's actually running)
    population.append(StrategyGenome(
        pipeline="smc_quant",
        timeframe="D1",
        min_confidence=0.65,
        min_adx=25,
        trailing_stop_atr_mult=2.0,
        # ... (from automation_configs.json)
    ))

    # Seed 2-8: Other current configs
    for config in load_automation_configs():
        population.append(genome_from_config(config))

    # Seed 9-50: Random mutations of the seeds
    while len(population) < 50:
        parent = random.choice(population[:8])
        mutant = self.mutate(parent, mutation_rate=0.4)  # Higher rate for diversity
        population.append(mutant)

    return population
```

### Per-Pair Specialization

The evolver runs **per pair**. Each pair gets its own population:

```
evolution_results/
├── XAUUSD/
│   ├── generation_001.json    # Full population + fitness
│   ├── generation_002.json
│   └── best_genome.json       # Current champion
├── GBPJPY/
│   ├── generation_001.json
│   └── best_genome.json
├── EURUSD/
│   └── ...
└── global/
    └── best_per_pair.json     # Summary: best genome per pair
```

### Overfitting Protection

This is the biggest risk. Mitigations:

1. **Walk-forward validation**: Train on months 1-4, test on month 5-6. Fitness only counts on test period.

2. **Minimum trade count**: Genomes with <15 trades in test period get penalized:
   ```python
   if fitness.total_trades < 15:
       fitness.composite *= 0.5  # Harsh penalty for insufficient data
   ```

3. **Stability penalty**: Genomes that are extremely sensitive to small param changes are fragile. Check neighbors:
   ```python
   # Perturb each gene slightly, re-evaluate
   # If fitness drops >30% from a 1-step nudge, add penalty
   stability = avg_fitness_of_neighbors / genome_fitness
   if stability < 0.7:
       fitness.composite *= stability
   ```

4. **Regime diversity**: Must perform at least break-even across all regimes (can't only work in trending and blow up in ranging):
   ```python
   if any(regime_wr < 0.35 for regime_wr in regime_win_rates.values()):
       fitness.composite *= 0.6  # Can't have a catastrophic regime
   ```

5. **Paper trading graduation**: Top genome must paper trade for 2 weeks (10+ trades) with positive results before going live.

### New Files

```
tradingagents/evolution/
├── genome.py               # StrategyGenome dataclass
├── genetic.py              # GeneticEvolver (selection, crossover, mutation)
├── fitness.py              # FitnessScore calculation
├── overfitting.py          # Walk-forward validation, stability checks
```

---

## Phase 5: Graduation Pipeline

### From Backtest to Live

```
Backtest Champion (Gen 20 winner)
    │
    ├── Walk-Forward Validation
    │   └── Must be profitable on unseen data
    │
    ├── Paper Trading (2 weeks)
    │   ├── Runs as automation instance with auto_execute=False
    │   ├── Records what it WOULD have traded
    │   └── Must achieve >45% WR and >1.0 RR on paper
    │
    ├── Small Live (2 weeks)
    │   ├── auto_execute=True, lot_size=0.01 (minimum)
    │   ├── max_risk=0.5% (half normal)
    │   └── Must not trigger circuit breaker
    │
    └── Full Live
        ├── Normal lot sizing
        └── Monitored, can be demoted back if performance degrades
```

### Demotion

If a live strategy's rolling 20-trade win rate drops below 35%, it gets:
1. Demoted to paper trading
2. Re-evolved (new generation run with updated data)
3. Must graduate again

### Integration with Scanner

The full loop:

```
Scanner → "GBPJPY is moving LONG, score 82"
    │
    ▼
Matcher → "GBPJPY + LONG → SMC Quant (genome_v47), 62% WR, 1.9 RR"
    │
    ▼
Executor → Runs SMC Quant with genome_v47's parameters on GBPJPY
    │  (min_confidence=0.60, min_zone_strength=0.5, sl_atr=1.75, tp_rr=2.5)
    │
    ▼
Trade closes → outcome recorded
    │
    ▼
Evolver → (every 30 trades) new generation, genome_v48 replaces v47 if better
```

---

## Implementation Order & Estimates

### Phase 1: Backtester
- `market_snapshot.py` — data container
- `strategy_rules.py` — rule-based strategy approximations (6 pipelines)
- `backtester.py` — walk-forward engine
- `backtest_results.py` — result storage

**Depends on**: Existing SMC analysis (`smart_money.py`), indicators, MT5 data
**Produces**: Performance matrix for all pairs × strategies

### Phase 2: Scanner
- `scanner.py` — momentum scoring
- `watchlist.py` — pair list management
- Integration into quant_automation or new automation type

**Depends on**: MT5 data (candles + indicators)
**Produces**: Ranked shortlist of pairs to trade

### Phase 3: Matcher
- `matcher.py` — strategy selection logic
- `performance_matrix.py` — blends live + backtest data

**Depends on**: Phase 1 (backtest data) + existing trade decisions
**Produces**: Best pipeline recommendation per pair

### Phase 4: Evolver
- `genome.py` — strategy parameter definition
- `genetic.py` — evolution operators
- `fitness.py` — multi-factor scoring
- `overfitting.py` — protection mechanisms

**Depends on**: Phase 1 (backtester for fitness evaluation)
**Produces**: Evolved strategy parameters per pair

### Phase 5: Graduation
- Promotion/demotion logic in automation
- Paper → small live → full live pipeline

**Depends on**: All previous phases

---

## Web UI Additions

### Scanner Dashboard
- Watchlist with live momentum scores (heatmap style)
- Direction indicators (green up arrow / red down arrow)
- "On the move" badge for shortlisted pairs
- Historical scan results

### Evolution Dashboard
- Per-pair: current generation, best genome fitness, lineage tree
- Performance matrix table (pairs × pipelines with color-coded WR)
- Backtest results viewer
- "Evolve Now" button (trigger manual generation)
- Genome inspector (see all parameters of the champion)

### Strategy Performance
- Live vs backtest comparison charts
- Rolling win rate per pair per strategy
- Demotion/promotion history

---

## What This Gives You

1. **You never miss an opportunity** — the scanner watches 16+ pairs, not just the 2 you're manually configured for

2. **The right tool for the job** — instead of guessing which pipeline works on GBPJPY, the data tells you

3. **Self-improving** — strategies evolve their parameters based on what actually works, not what you think should work

4. **Overfitting-resistant** — walk-forward validation, stability checks, and graduation pipeline prevent paper-only champions from losing money live

5. **Compounding knowledge** — every trade makes the system smarter. The 6 wins and 28 losses you have today become the seed data that bootstraps the first generation

6. **Survival of the fittest** — bad strategies die, good ones breed. Over time, only profitable configurations survive

The system doesn't just trade — it learns what works, where it works, and evolves to do more of it.
