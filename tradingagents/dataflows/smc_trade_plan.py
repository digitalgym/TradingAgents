"""
SMC Trade Plan Generator

Rule-based trade plan generation following Smart Money Concepts strategy.
Generates systematic entry/SL/TP levels based on detected SMC zones.

This provides the "disciplined foundation" that the LLM refiner builds upon.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get an attribute from an object or dict.

    Works with:
    - Dataclass objects (uses getattr)
    - Dictionaries (uses .get())
    - Any object with the attribute

    Args:
        obj: Object or dict to get attribute from
        attr: Attribute name
        default: Default value if attribute not found

    Returns:
        The attribute value or default
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


class SetupType(Enum):
    """Types of SMC trade setups."""
    OB_ENTRY = "ob_entry"  # Entry at Order Block
    OB_FVG_CONFLUENCE = "ob_fvg_confluence"  # OB with FVG overlap
    FVG_ENTRY = "fvg_entry"  # Entry at Fair Value Gap
    BREAKER_ENTRY = "breaker_entry"  # Entry at Breaker Block
    OTE_ENTRY = "ote_entry"  # Entry at Optimal Trade Entry zone
    LIQUIDITY_SWEEP = "liquidity_sweep"  # Entry after liquidity grab


@dataclass
class EntryChecklist:
    """Validates all conditions before recommending a trade."""
    htf_trend_aligned: bool = False
    zone_unmitigated: bool = False
    has_confluence: bool = False
    liquidity_target_exists: bool = False
    structure_confirmed: bool = False  # BOS or CHoCH
    in_discount_premium: bool = False  # Buy in discount, sell in premium
    session_favorable: bool = False

    @property
    def passed_count(self) -> int:
        """Count how many checks passed."""
        return sum([
            self.htf_trend_aligned,
            self.zone_unmitigated,
            self.has_confluence,
            self.liquidity_target_exists,
            self.structure_confirmed,
            self.in_discount_premium,
            self.session_favorable,
        ])

    @property
    def total_count(self) -> int:
        return 7

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "htf_trend_aligned": self.htf_trend_aligned,
            "zone_unmitigated": self.zone_unmitigated,
            "has_confluence": self.has_confluence,
            "liquidity_target_exists": self.liquidity_target_exists,
            "structure_confirmed": self.structure_confirmed,
            "in_discount_premium": self.in_discount_premium,
            "session_favorable": self.session_favorable,
            "passed": self.passed_count,
            "total": self.total_count,
            "pass_rate": self.pass_rate,
        }


@dataclass
class SMCTradePlan:
    """A systematic trade plan generated from SMC rules."""
    # Core trade parameters
    signal: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float

    # Quality metrics
    zone_quality_score: float  # 0-100
    setup_type: SetupType
    checklist: EntryChecklist

    # Zone details
    entry_zone: Dict[str, Any] = field(default_factory=dict)
    sl_zone: Dict[str, Any] = field(default_factory=dict)
    tp_zone: Dict[str, Any] = field(default_factory=dict)

    # Confluence factors
    confluence_factors: List[str] = field(default_factory=list)

    # Risk metrics
    risk_reward_ratio: float = 0.0
    sl_distance_atr: float = 0.0  # SL distance in ATR units

    # Recommendation
    recommendation: str = "SKIP"  # "TAKE", "SKIP", "WAIT"
    skip_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if plan meets minimum criteria."""
        return (
            self.zone_quality_score >= 60 and
            self.risk_reward_ratio >= 1.5 and
            self.checklist.pass_rate >= 0.5
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "zone_quality_score": self.zone_quality_score,
            "setup_type": self.setup_type.value,
            "checklist": self.checklist.to_dict(),
            "entry_zone": self.entry_zone,
            "sl_zone": self.sl_zone,
            "tp_zone": self.tp_zone,
            "confluence_factors": self.confluence_factors,
            "risk_reward_ratio": self.risk_reward_ratio,
            "sl_distance_atr": self.sl_distance_atr,
            "recommendation": self.recommendation,
            "skip_reason": self.skip_reason,
            "is_valid": self.is_valid,
        }


class SMCTradePlanGenerator:
    """
    Generates systematic trade plans based on SMC rules.

    Strategy Steps (from SMC Reference):
    1. Determine trend direction (only trade with trend)
    2. Identify high-probability Order Block (score >= 60)
    3. Calculate entry at zone level
    4. Place SL below/above zone with buffer
    5. Target next liquidity or opposing zone for TP
    """

    def __init__(
        self,
        min_quality_score: float = 60.0,
        min_rr_ratio: float = 1.5,
        sl_buffer_atr: float = 1.0,  # ATR multiplier for SL buffer (minimum 1x ATR)
        entry_zone_percent: float = 0.5,  # Enter at 50% of zone by default
    ):
        """
        Initialize the trade plan generator.

        Args:
            min_quality_score: Minimum zone quality to consider (0-100)
            min_rr_ratio: Minimum risk:reward ratio
            sl_buffer_atr: ATR multiplier for stop loss buffer beyond zone
            entry_zone_percent: Where in the zone to enter (0=edge, 0.5=middle, 1=far edge)
        """
        self.min_quality_score = min_quality_score
        self.min_rr_ratio = min_rr_ratio
        self.sl_buffer_atr = sl_buffer_atr
        self.entry_zone_percent = entry_zone_percent

    def generate_plan(
        self,
        smc_analysis: Dict[str, Any],
        current_price: float,
        atr: float,
        market_regime: Optional[str] = None,
        session: Optional[str] = None,
    ) -> Optional[SMCTradePlan]:
        """
        Generate a trade plan from SMC analysis.

        Args:
            smc_analysis: Full SMC analysis from SmartMoneyAnalyzer
            current_price: Current market price
            atr: Current ATR value
            market_regime: Current market regime ("trending-up", "trending-down", "ranging")
            session: Current trading session

        Returns:
            SMCTradePlan or None if no valid setup found
        """
        # Step 1: Determine trend and trading direction
        trend = self._determine_trend(smc_analysis, market_regime)
        if trend == "unknown":
            return None

        trade_direction = "BUY" if trend in ["bullish", "trending-up"] else "SELL"

        # Step 2: Find best entry zone
        entry_zone, setup_type = self._find_best_entry_zone(
            smc_analysis, trade_direction, current_price
        )

        if not entry_zone:
            return None

        # Step 3: Calculate zone quality score
        quality_score = self._calculate_zone_quality(
            entry_zone, smc_analysis, trade_direction, current_price
        )

        # Step 4: Calculate entry, SL, TP levels
        entry_price = self._calculate_entry_price(entry_zone, trade_direction, current_price)
        sl_price, sl_zone = self._calculate_stop_loss(
            entry_zone, trade_direction, atr, smc_analysis
        )
        tp_price, tp_zone = self._calculate_take_profit(
            entry_price, trade_direction, smc_analysis, current_price, market_regime
        )

        # Step 5: Validate risk:reward
        if trade_direction == "BUY":
            risk = entry_price - sl_price
            reward = tp_price - entry_price
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price

        rr_ratio = reward / risk if risk > 0 else 0
        sl_distance_atr = risk / atr if atr > 0 else 0

        # Step 6: Build entry checklist
        checklist = self._build_checklist(
            smc_analysis, trade_direction, entry_zone,
            market_regime, session, current_price
        )

        # Step 7: Determine confluence factors
        confluence_factors = self._get_confluence_factors(
            entry_zone, smc_analysis, trade_direction, current_price
        )

        # Step 8: Build the plan
        plan = SMCTradePlan(
            signal=trade_direction,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            zone_quality_score=quality_score,
            setup_type=setup_type,
            checklist=checklist,
            entry_zone=entry_zone,
            sl_zone=sl_zone,
            tp_zone=tp_zone,
            confluence_factors=confluence_factors,
            risk_reward_ratio=rr_ratio,
            sl_distance_atr=sl_distance_atr,
        )

        # Step 9: Determine recommendation
        plan.recommendation, plan.skip_reason = self._determine_recommendation(plan)

        return plan

    def _determine_trend(
        self,
        smc_analysis: Dict[str, Any],
        market_regime: Optional[str]
    ) -> str:
        """Determine the current trend direction."""
        # Prefer explicit regime if provided
        if market_regime:
            if "up" in market_regime.lower():
                return "bullish"
            elif "down" in market_regime.lower():
                return "bearish"
            elif "rang" in market_regime.lower():
                return "ranging"

        # Fall back to SMC bias
        bias = smc_analysis.get("bias")
        if bias:
            return bias

        # Analyze structure
        structure = smc_analysis.get("structure", {})
        recent_bos = structure.get("recent_bos", [])
        recent_choc = structure.get("recent_choc", [])

        if recent_choc:
            # CHoCH indicates potential reversal
            last_choc = recent_choc[-1]
            if hasattr(last_choc, 'type'):
                return "bullish" if last_choc.type == "high" else "bearish"

        if recent_bos:
            last_bos = recent_bos[-1]
            if hasattr(last_bos, 'type'):
                return "bullish" if last_bos.type == "high" else "bearish"

        return "unknown"

    def _find_best_entry_zone(
        self,
        smc_analysis: Dict[str, Any],
        direction: str,
        current_price: float,
    ) -> Tuple[Optional[Dict[str, Any]], SetupType]:
        """Find the best zone for entry based on direction and proximity."""
        candidates = []

        # Get order blocks
        obs = smc_analysis.get("order_blocks", {})
        zone_type = "bullish" if direction == "BUY" else "bearish"

        for ob in obs.get(zone_type, []):
            if self._is_zone_valid_for_entry(ob, direction, current_price):
                score = self._score_entry_zone(ob, smc_analysis, direction, current_price)
                candidates.append((ob, SetupType.OB_ENTRY, score, "ob"))

        # Get FVGs
        fvgs = smc_analysis.get("fair_value_gaps", {})
        for fvg in fvgs.get(zone_type, []):
            if self._is_zone_valid_for_entry(fvg, direction, current_price):
                score = self._score_entry_zone(fvg, smc_analysis, direction, current_price)
                candidates.append((fvg, SetupType.FVG_ENTRY, score, "fvg"))

        # Get breaker blocks
        breakers = smc_analysis.get("breaker_blocks", [])
        for bb in breakers:
            bb_type = safe_get(bb, 'new_type', None)
            if bb_type == zone_type:
                if self._is_zone_valid_for_entry(bb, direction, current_price):
                    score = self._score_entry_zone(bb, smc_analysis, direction, current_price)
                    candidates.append((bb, SetupType.BREAKER_ENTRY, score, "breaker"))

        # Check for OB+FVG confluence
        for ob, setup, score, ztype in candidates:
            if ztype == "ob":
                for fvg in fvgs.get(zone_type, []):
                    if self._zones_overlap(ob, fvg):
                        # Boost score for confluence
                        score += 20
                        candidates.append((ob, SetupType.OB_FVG_CONFLUENCE, score, "ob_fvg"))

        if not candidates:
            return None, SetupType.OB_ENTRY

        # Sort by score and return best
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]

        # Convert to dict if dataclass
        zone = best[0]
        if hasattr(zone, '__dataclass_fields__'):
            zone_dict = {
                "type": getattr(zone, 'type', zone_type),
                "top": getattr(zone, 'top', 0),
                "bottom": getattr(zone, 'bottom', 0),
                "strength": getattr(zone, 'strength', 0.5),
                "mitigated": getattr(zone, 'mitigated', False),
            }
        else:
            zone_dict = dict(zone) if isinstance(zone, dict) else {"raw": zone}

        return zone_dict, best[1]

    def _is_zone_valid_for_entry(
        self,
        zone: Any,
        direction: str,
        current_price: float,
    ) -> bool:
        """Check if a zone is valid for entry."""
        # Get zone bounds
        top = safe_get(zone, 'top', 0)
        bottom = safe_get(zone, 'bottom', 0)
        mitigated = safe_get(zone, 'mitigated', False)

        # Skip mitigated zones
        if mitigated:
            return False

        # Price must be inside or very close to the zone for a market-entry signal.
        # We use 1% proximity threshold — beyond that, it would be a limit order setup
        # which the automation doesn't support (it executes at market).
        max_distance_pct = 0.01  # 1% of price

        if direction == "BUY":
            # Zone should be at or below current price
            # Price can be inside zone or slightly above (already touched/bouncing)
            if bottom > current_price:
                return False  # Zone entirely above price — wrong side
            distance = max(0, current_price - top)  # 0 if price inside zone
            return distance / current_price < max_distance_pct
        else:
            # Zone should be at or above current price
            # Price can be inside zone or slightly below (already touched/rejecting)
            if top < current_price:
                return False  # Zone entirely below price — wrong side
            distance = max(0, bottom - current_price)  # 0 if price inside zone
            return distance / current_price < max_distance_pct

    def _score_entry_zone(
        self,
        zone: Any,
        smc_analysis: Dict[str, Any],
        direction: str,
        current_price: float,
    ) -> float:
        """Score a zone for entry quality."""
        score = 0.0

        # Base strength (0-40 points)
        strength = safe_get(zone, 'strength', 0.5)
        score += strength * 40

        # Proximity to current price (0-20 points) - closer is better for immediate entry
        top = safe_get(zone, 'top', current_price)
        bottom = safe_get(zone, 'bottom', current_price)
        zone_mid = (top + bottom) / 2
        distance_pct = abs(current_price - zone_mid) / current_price
        proximity_score = max(0, 20 - (distance_pct * 400))  # 0-20 points
        score += proximity_score

        # Unmitigated bonus (10 points)
        mitigated = safe_get(zone, 'mitigated', False)
        if not mitigated:
            score += 10

        # Check for FVG confluence (15 points)
        fvgs = smc_analysis.get("fair_value_gaps", {})
        zone_type = "bullish" if direction == "BUY" else "bearish"
        for fvg in fvgs.get(zone_type, []):
            if self._zones_overlap(zone, fvg):
                score += 15
                break

        # Recent BOS/CHoCH confirmation (15 points)
        structure = smc_analysis.get("structure", {})
        if structure.get("recent_bos") or structure.get("recent_choc"):
            score += 15

        return min(score, 100)

    def _zones_overlap(self, zone1: Any, zone2: Any) -> bool:
        """Check if two zones overlap."""
        top1 = safe_get(zone1, 'top', 0)
        bottom1 = safe_get(zone1, 'bottom', 0)
        top2 = safe_get(zone2, 'top', 0)
        bottom2 = safe_get(zone2, 'bottom', 0)

        return not (top1 < bottom2 or top2 < bottom1)

    def _calculate_zone_quality(
        self,
        zone: Dict[str, Any],
        smc_analysis: Dict[str, Any],
        direction: str,
        current_price: float,
    ) -> float:
        """
        Calculate overall zone quality score (0-100).

        Scoring:
        - Zone strength: 0-30 points
        - Caused BOS/CHoCH: 0-20 points
        - Has liquidity above/below: 0-15 points
        - FVG confluence: 0-15 points
        - Premium/discount alignment: 0-10 points
        - Session: 0-10 points
        """
        score = 0.0

        # Zone strength (0-30)
        strength = safe_get(zone, 'strength', 0.5)
        score += strength * 30

        # Structure break confirmation (0-20)
        structure = smc_analysis.get("structure", {})
        if structure.get("recent_bos") or structure.get("recent_choc"):
            score += 20

        # Liquidity target exists (0-15)
        liquidity_zones = smc_analysis.get("liquidity_zones", [])
        has_target = False
        for lz in liquidity_zones:
            lz_price = safe_get(lz, 'price', 0)
            if direction == "BUY" and lz_price > current_price:
                has_target = True
                break
            elif direction == "SELL" and lz_price < current_price:
                has_target = True
                break
        if has_target:
            score += 15

        # FVG confluence (0-15)
        fvgs = smc_analysis.get("fair_value_gaps", {})
        zone_type = "bullish" if direction == "BUY" else "bearish"
        for fvg in fvgs.get(zone_type, []):
            fvg_dict = fvg if isinstance(fvg, dict) else {"top": getattr(fvg, 'top', 0), "bottom": getattr(fvg, 'bottom', 0)}
            if self._zones_overlap(zone, fvg_dict):
                score += 15
                break

        # Premium/Discount alignment (0-10)
        pd_zone = smc_analysis.get("premium_discount")
        if pd_zone:
            pd_zone_name = safe_get(pd_zone, 'zone', None)
            if direction == "BUY" and pd_zone_name == "discount":
                score += 10
            elif direction == "SELL" and pd_zone_name == "premium":
                score += 10

        # Session bonus (0-10) - placeholder, would need session data
        score += 5  # Default mid-score

        return min(score, 100)

    def _calculate_entry_price(
        self,
        zone: Dict[str, Any],
        direction: str,
        current_price: float = 0,
    ) -> float:
        """
        Calculate entry price for market execution.

        Uses current market price when price is inside or near the zone,
        since this is for market orders (not limit orders).
        """
        top = safe_get(zone, 'top', 0)
        bottom = safe_get(zone, 'bottom', 0)

        # If price is inside the zone, use current price (market entry)
        if current_price and bottom <= current_price <= top:
            return current_price

        # Price is near but outside zone — use the nearest zone edge
        if direction == "BUY":
            # Price is slightly above zone top, enter at market
            return current_price if current_price else top
        else:
            # Price is slightly below zone bottom, enter at market
            return current_price if current_price else bottom

    def _calculate_stop_loss(
        self,
        entry_zone: Dict[str, Any],
        direction: str,
        atr: float,
        smc_analysis: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate stop loss level.

        Rules:
        - BUY: SL below the entry zone bottom with ATR buffer
        - SELL: SL above the entry zone top with ATR buffer
        - Look for liquidity below/above to avoid sweep
        """
        top = safe_get(entry_zone, 'top', 0)
        bottom = safe_get(entry_zone, 'bottom', 0)
        buffer = atr * self.sl_buffer_atr

        if direction == "BUY":
            # SL below zone
            sl = bottom - buffer

            # Check if there's liquidity just below - might need to extend
            liquidity_zones = smc_analysis.get("liquidity_zones", [])
            for lz in liquidity_zones:
                lz_price = safe_get(lz, 'price', 0)
                lz_type = safe_get(lz, 'type', '')
                # If sell-side liquidity is just below our SL, extend past it
                if lz_type == 'sell-side' and sl > lz_price > (sl - atr):
                    sl = lz_price - (atr * 0.3)  # Place SL below liquidity
                    break
        else:
            # SL above zone
            sl = top + buffer

            # Check for buy-side liquidity above
            liquidity_zones = smc_analysis.get("liquidity_zones", [])
            for lz in liquidity_zones:
                lz_price = safe_get(lz, 'price', 0)
                lz_type = safe_get(lz, 'type', '')
                if lz_type == 'buy-side' and sl < lz_price < (sl + atr):
                    sl = lz_price + (atr * 0.3)
                    break

        sl_zone = {"type": "stop_loss", "price": sl, "reason": "below_zone_with_buffer"}
        return sl, sl_zone

    def _calculate_take_profit(
        self,
        entry_price: float,
        direction: str,
        smc_analysis: Dict[str, Any],
        current_price: float,
        market_regime: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate take profit level.

        Rules:
        - Target next liquidity zone, opposing OB, or equal level (proven support/resistance)
        - In strong trends, prefer stronger/deeper targets over nearest
        - In weak/ranging markets, prefer nearest conservative target
        - Ensure minimum R:R ratio
        - IMPORTANT: For limit orders, TP must be BEYOND current price
          (above current for BUY, below current for SELL)
        """
        tp_candidates = []

        # For limit orders, we need TP beyond current price, not just beyond entry
        min_tp_price = max(entry_price, current_price) if direction == "BUY" else 0
        max_tp_price = min(entry_price, current_price) if direction == "SELL" else float('inf')

        # Look for liquidity targets
        liquidity_zones = smc_analysis.get("liquidity_zones", [])
        for lz in liquidity_zones:
            lz_price = safe_get(lz, 'price', 0)
            lz_strength = safe_get(lz, 'strength', 50)

            if direction == "BUY" and lz_price > min_tp_price:
                tp_candidates.append((lz_price, lz_strength, "liquidity", lz))
            elif direction == "SELL" and lz_price < max_tp_price:
                tp_candidates.append((lz_price, lz_strength, "liquidity", lz))

        # Look for opposing order blocks
        obs = smc_analysis.get("order_blocks", {})
        opposing_type = "bearish" if direction == "BUY" else "bullish"
        for ob in obs.get(opposing_type, []):
            ob_top = safe_get(ob, 'top', 0)
            ob_bottom = safe_get(ob, 'bottom', 0)
            ob_strength = safe_get(ob, 'strength', 0.5)
            ob_mid = (ob_top + ob_bottom) / 2

            if direction == "BUY" and ob_mid > min_tp_price:
                tp_candidates.append((ob_bottom, ob_strength * 100, "opposing_ob", ob))
            elif direction == "SELL" and ob_mid < max_tp_price:
                tp_candidates.append((ob_top, ob_strength * 100, "opposing_ob", ob))

        # Look for equal levels (proven support/resistance) as TP targets
        equal_levels = smc_analysis.get("equal_levels", {})
        eq_highs = equal_levels.get("equal_highs", []) if isinstance(equal_levels, dict) else []
        eq_lows = equal_levels.get("equal_lows", []) if isinstance(equal_levels, dict) else []

        if direction == "BUY":
            # For BUY: target equal highs above (resistance / liquidity magnet)
            for eq in eq_highs:
                eq_price = safe_get(eq, 'price', 0)
                eq_touches = safe_get(eq, 'touches', 2)
                eq_swept = safe_get(eq, 'swept', False)
                if eq_price > min_tp_price and not eq_swept:
                    # Strength scales with touches: 2 touches = 60, 6 touches = 100
                    eq_strength = min(40 + eq_touches * 10, 100)
                    tp_candidates.append((eq_price, eq_strength, "equal_level", eq))
        else:
            # For SELL: target equal lows below (support / liquidity magnet)
            for eq in eq_lows:
                eq_price = safe_get(eq, 'price', 0)
                eq_touches = safe_get(eq, 'touches', 2)
                eq_swept = safe_get(eq, 'swept', False)
                if eq_price < max_tp_price and not eq_swept:
                    eq_strength = min(40 + eq_touches * 10, 100)
                    tp_candidates.append((eq_price, eq_strength, "equal_level", eq))

        if not tp_candidates:
            # Fallback: project TP beyond current price
            risk_distance = abs(entry_price - current_price) if entry_price != current_price else current_price * 0.01
            if direction == "BUY":
                tp_price = current_price + (risk_distance * 2)
            else:
                tp_price = current_price - (risk_distance * 2)
            return tp_price, {"type": "calculated", "price": tp_price, "reason": "no_target_found"}

        # Determine trend strength to adjust TP selection strategy
        # In strong trends: prefer stronger targets (even if farther)
        # In weak/ranging: prefer nearest targets (conservative)
        is_strong_trend = False
        if market_regime:
            regime_lower = market_regime.lower()
            is_strong_trend = "trending" in regime_lower

        # Score each candidate: balance between strength and proximity
        # Normalize distance relative to entry price
        for i, (price, strength, tp_type, zone) in enumerate(tp_candidates):
            distance = abs(price - entry_price)
            distance_pct = distance / entry_price * 100  # Distance as % of price

            if is_strong_trend:
                # Strong trend: weight strength heavily (70%), proximity less (30%)
                # Penalize very close targets (< 0.2% away = likely noise)
                proximity_penalty = 0 if distance_pct >= 0.2 else -30
                score = strength * 0.7 + proximity_penalty
            else:
                # Ranging/weak: weight proximity heavily, strength as tiebreaker
                # Closer = higher score (invert distance)
                proximity_score = max(0, 100 - distance_pct * 20)  # 0% = 100, 5% = 0
                score = proximity_score * 0.6 + strength * 0.4

            tp_candidates[i] = (price, strength, tp_type, zone, score)

        # Sort by score descending (highest score = best target)
        tp_candidates.sort(key=lambda x: -x[4])

        tp_price, strength, tp_type, zone, score = tp_candidates[0]
        tp_zone = {"type": tp_type, "price": tp_price, "strength": strength}

        return tp_price, tp_zone

    def _build_checklist(
        self,
        smc_analysis: Dict[str, Any],
        direction: str,
        entry_zone: Dict[str, Any],
        market_regime: Optional[str],
        session: Optional[str],
        current_price: float,
    ) -> EntryChecklist:
        """Build the entry criteria checklist."""
        checklist = EntryChecklist()

        # HTF trend aligned
        if market_regime:
            if direction == "BUY" and "up" in market_regime.lower():
                checklist.htf_trend_aligned = True
            elif direction == "SELL" and "down" in market_regime.lower():
                checklist.htf_trend_aligned = True
        else:
            # Check bias from analysis
            bias = smc_analysis.get("bias")
            if bias:
                if direction == "BUY" and bias == "bullish":
                    checklist.htf_trend_aligned = True
                elif direction == "SELL" and bias == "bearish":
                    checklist.htf_trend_aligned = True

        # Zone unmitigated
        mitigated = safe_get(entry_zone, 'mitigated', False)
        checklist.zone_unmitigated = not mitigated

        # Has confluence (FVG overlap or multiple factors)
        fvgs = smc_analysis.get("fair_value_gaps", {})
        zone_type = "bullish" if direction == "BUY" else "bearish"
        for fvg in fvgs.get(zone_type, []):
            fvg_dict = fvg if isinstance(fvg, dict) else {"top": getattr(fvg, 'top', 0), "bottom": getattr(fvg, 'bottom', 0)}
            if self._zones_overlap(entry_zone, fvg_dict):
                checklist.has_confluence = True
                break

        # Liquidity target exists
        liquidity_zones = smc_analysis.get("liquidity_zones", [])
        for lz in liquidity_zones:
            lz_price = safe_get(lz, 'price', 0)
            if direction == "BUY" and lz_price > current_price:
                checklist.liquidity_target_exists = True
                break
            elif direction == "SELL" and lz_price < current_price:
                checklist.liquidity_target_exists = True
                break

        # Structure confirmed (BOS or CHoCH)
        structure = smc_analysis.get("structure", {})
        if structure.get("recent_bos") or structure.get("recent_choc"):
            checklist.structure_confirmed = True

        # In discount/premium
        pd_zone = smc_analysis.get("premium_discount")
        if pd_zone:
            pd_zone_name = safe_get(pd_zone, 'zone', None)
            if direction == "BUY" and pd_zone_name == "discount":
                checklist.in_discount_premium = True
            elif direction == "SELL" and pd_zone_name == "premium":
                checklist.in_discount_premium = True

        # Session favorable
        if session:
            favorable_sessions = ["london", "ny", "london_ny_overlap"]
            checklist.session_favorable = any(s in session.lower() for s in favorable_sessions)
        else:
            checklist.session_favorable = True  # Assume favorable if not specified

        return checklist

    def _get_confluence_factors(
        self,
        entry_zone: Dict[str, Any],
        smc_analysis: Dict[str, Any],
        direction: str,
        current_price: float,
    ) -> List[str]:
        """Get list of confluence factors for this setup."""
        factors = []

        # Zone type
        zone_type = safe_get(entry_zone, 'type', 'unknown')
        factors.append(f"{zone_type.capitalize()} Order Block")

        # FVG confluence
        fvgs = smc_analysis.get("fair_value_gaps", {})
        fvg_type = "bullish" if direction == "BUY" else "bearish"
        for fvg in fvgs.get(fvg_type, []):
            fvg_dict = fvg if isinstance(fvg, dict) else {"top": getattr(fvg, 'top', 0), "bottom": getattr(fvg, 'bottom', 0)}
            if self._zones_overlap(entry_zone, fvg_dict):
                factors.append("FVG Confluence")
                break

        # Structure
        structure = smc_analysis.get("structure", {})
        if structure.get("recent_bos"):
            factors.append("Recent BOS Confirmation")
        if structure.get("recent_choc"):
            factors.append("Recent CHoCH")

        # Premium/Discount
        pd_zone = smc_analysis.get("premium_discount")
        if pd_zone:
            pd_zone_name = safe_get(pd_zone, 'zone', None)
            if direction == "BUY" and pd_zone_name == "discount":
                factors.append("In Discount Zone")
            elif direction == "SELL" and pd_zone_name == "premium":
                factors.append("In Premium Zone")

        # Liquidity
        liquidity_zones = smc_analysis.get("liquidity_zones", [])
        for lz in liquidity_zones:
            lz_price = safe_get(lz, 'price', 0)
            if direction == "BUY" and lz_price > current_price:
                factors.append("Liquidity Target Above")
                break
            elif direction == "SELL" and lz_price < current_price:
                factors.append("Liquidity Target Below")
                break

        return factors

    def _determine_recommendation(
        self,
        plan: SMCTradePlan,
    ) -> Tuple[str, Optional[str]]:
        """Determine whether to take, skip, or wait on this setup."""

        # Must pass minimum quality
        if plan.zone_quality_score < self.min_quality_score:
            return "SKIP", f"Zone quality {plan.zone_quality_score:.0f}% below minimum {self.min_quality_score:.0f}%"

        # Must have acceptable R:R
        if plan.risk_reward_ratio < self.min_rr_ratio:
            return "SKIP", f"R:R {plan.risk_reward_ratio:.2f} below minimum {self.min_rr_ratio:.2f}"

        # Check critical checklist items
        if not plan.checklist.zone_unmitigated:
            return "SKIP", "Zone has been mitigated"

        if not plan.checklist.htf_trend_aligned:
            return "SKIP", "Trade against HTF trend"

        # Warn if low checklist pass rate
        if plan.checklist.pass_rate < 0.5:
            return "SKIP", f"Only {plan.checklist.passed_count}/{plan.checklist.total_count} checklist items passed"

        # All checks passed
        return "TAKE", None
