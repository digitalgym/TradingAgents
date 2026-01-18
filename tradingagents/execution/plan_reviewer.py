"""
Plan Reviewer

Reviews and adapts trading plans based on:
- Market conditions changes
- Position performance
- New information
- Risk events
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PlanAdjustment:
    """Recommended plan adjustment"""
    component: str  # entry, stop, target, size
    current_value: Any
    recommended_value: Any
    reason: str
    urgency: str  # low, medium, high
    confidence: float


class PlanReviewer:
    """
    Reviews trading plans and recommends adjustments.
    
    Checks:
    - Regime changes
    - Price action vs plan
    - Risk metrics
    - New catalysts
    - Performance vs expectations
    """
    
    def __init__(self):
        pass
    
    def review_plan(
        self,
        plan: Any,  # ParsedTradingPlan
        current_price: float,
        entry_price: Optional[float] = None,
        position_size: float = 0.0,
        time_in_trade: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Review a trading plan and recommend adjustments.
        
        Args:
            plan: ParsedTradingPlan object
            current_price: Current market price
            entry_price: Actual entry price (if in position)
            position_size: Current position size %
            time_in_trade: Days in trade (if in position)
        
        Returns:
            dict with review results and recommendations
        """
        adjustments = []
        
        # Check entry strategy
        if entry_price is None:
            # Not in position yet - review entry plan
            entry_adj = self._review_entry_strategy(plan, current_price)
            if entry_adj:
                adjustments.extend(entry_adj)
        
        # Check stop loss
        if entry_price:
            stop_adj = self._review_stop_loss(plan, current_price, entry_price)
            if stop_adj:
                adjustments.extend(stop_adj)
        
        # Check take profit targets
        if entry_price:
            tp_adj = self._review_take_profits(plan, current_price, entry_price)
            if tp_adj:
                adjustments.extend(tp_adj)
        
        # Check position sizing
        size_adj = self._review_position_size(plan, position_size, current_price, entry_price)
        if size_adj:
            adjustments.extend(size_adj)
        
        # Check time-based factors
        if time_in_trade:
            time_adj = self._review_time_factors(plan, time_in_trade, current_price, entry_price)
            if time_adj:
                adjustments.extend(time_adj)
        
        # Prioritize adjustments
        high_priority = [a for a in adjustments if a.urgency == "high"]
        medium_priority = [a for a in adjustments if a.urgency == "medium"]
        low_priority = [a for a in adjustments if a.urgency == "low"]
        
        return {
            "review_time": datetime.now().isoformat(),
            "total_adjustments": len(adjustments),
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "all_adjustments": adjustments
        }
    
    def _review_entry_strategy(
        self,
        plan: Any,
        current_price: float
    ) -> List[PlanAdjustment]:
        """Review entry strategy vs current price"""
        adjustments = []
        
        if not plan.entry_tranches:
            return adjustments
        
        # Check if price has moved significantly from planned entries
        first_tranche = plan.entry_tranches[0]
        target_price = first_tranche.price_level
        
        if plan.direction == "BUY":
            distance_pct = ((current_price - target_price) / target_price) * 100
            
            if distance_pct > 3.0:
                # Price moved 3%+ above entry - may have missed it
                adjustments.append(PlanAdjustment(
                    component="entry",
                    current_value=target_price,
                    recommended_value=current_price * 0.995,  # 0.5% below current
                    reason=f"Price moved {distance_pct:.1f}% above planned entry. Consider adjusting up or entering at market.",
                    urgency="high",
                    confidence=0.8
                ))
            elif distance_pct < -5.0:
                # Price moved 5%+ below entry - better opportunity
                adjustments.append(PlanAdjustment(
                    component="entry",
                    current_value=target_price,
                    recommended_value=current_price,
                    reason=f"Price moved {abs(distance_pct):.1f}% below planned entry. Favorable entry opportunity.",
                    urgency="medium",
                    confidence=0.7
                ))
        else:  # SELL
            distance_pct = ((target_price - current_price) / target_price) * 100
            
            if distance_pct > 3.0:
                # Price moved 3%+ below entry - may have missed it
                adjustments.append(PlanAdjustment(
                    component="entry",
                    current_value=target_price,
                    recommended_value=current_price * 1.005,  # 0.5% above current
                    reason=f"Price moved {distance_pct:.1f}% below planned entry. Consider adjusting down or entering at market.",
                    urgency="high",
                    confidence=0.8
                ))
            elif distance_pct < -5.0:
                # Price moved 5%+ above entry - better opportunity
                adjustments.append(PlanAdjustment(
                    component="entry",
                    current_value=target_price,
                    recommended_value=current_price,
                    reason=f"Price moved {abs(distance_pct):.1f}% above planned entry. Favorable entry opportunity.",
                    urgency="medium",
                    confidence=0.7
                ))
        
        return adjustments
    
    def _review_stop_loss(
        self,
        plan: Any,
        current_price: float,
        entry_price: float
    ) -> List[PlanAdjustment]:
        """Review stop loss placement"""
        adjustments = []
        
        current_stop = plan.stop_loss.initial_price
        
        # Calculate profit
        if plan.direction == "BUY":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            stop_distance_pct = ((current_price - current_stop) / current_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            stop_distance_pct = ((current_stop - current_price) / current_price) * 100
        
        # If in profit >2%, consider moving to breakeven
        if profit_pct > 2.0:
            if plan.direction == "BUY":
                be_stop = entry_price * 1.001  # Breakeven + 0.1%
                if be_stop > current_stop:
                    adjustments.append(PlanAdjustment(
                        component="stop",
                        current_value=current_stop,
                        recommended_value=be_stop,
                        reason=f"Position up {profit_pct:.1f}%. Move stop to breakeven to protect profit.",
                        urgency="medium",
                        confidence=0.85
                    ))
            else:
                be_stop = entry_price * 0.999
                if be_stop < current_stop:
                    adjustments.append(PlanAdjustment(
                        component="stop",
                        current_value=current_stop,
                        recommended_value=be_stop,
                        reason=f"Position up {profit_pct:.1f}%. Move stop to breakeven to protect profit.",
                        urgency="medium",
                        confidence=0.85
                    ))
        
        # Check if stop is too tight
        if abs(stop_distance_pct) < 0.5:
            adjustments.append(PlanAdjustment(
                component="stop",
                current_value=current_stop,
                recommended_value=current_price * (0.985 if plan.direction == "BUY" else 1.015),
                reason=f"Stop very tight ({stop_distance_pct:.2f}%). Risk of premature exit. Consider widening.",
                urgency="low",
                confidence=0.6
            ))
        
        return adjustments
    
    def _review_take_profits(
        self,
        plan: Any,
        current_price: float,
        entry_price: float
    ) -> List[PlanAdjustment]:
        """Review take profit targets"""
        adjustments = []
        
        if not plan.take_profits:
            return adjustments
        
        # Check if approaching first TP
        first_tp = plan.take_profits[0]
        
        if plan.direction == "BUY":
            distance_to_tp = ((first_tp.price_level - current_price) / current_price) * 100
        else:
            distance_to_tp = ((current_price - first_tp.price_level) / current_price) * 100
        
        # If within 1% of TP, prepare to take profit
        if 0 < distance_to_tp < 1.0:
            adjustments.append(PlanAdjustment(
                component="target",
                current_value=first_tp.price_level,
                recommended_value=first_tp.price_level,
                reason=f"Within {distance_to_tp:.1f}% of TP1 (${first_tp.price_level:.2f}). Prepare to take {first_tp.size_pct:.0f}% profit.",
                urgency="high",
                confidence=0.9
            ))
        
        # If price moved way past TP without hitting, adjust
        if distance_to_tp < -2.0:
            adjustments.append(PlanAdjustment(
                component="target",
                current_value=first_tp.price_level,
                recommended_value=current_price * (1.01 if plan.direction == "BUY" else 0.99),
                reason=f"Price moved {abs(distance_to_tp):.1f}% past TP1. Consider taking profit now or adjusting target.",
                urgency="high",
                confidence=0.85
            ))
        
        return adjustments
    
    def _review_position_size(
        self,
        plan: Any,
        current_size: float,
        current_price: float,
        entry_price: Optional[float]
    ) -> List[PlanAdjustment]:
        """Review position sizing"""
        adjustments = []
        
        # Check if under-sized
        if current_size < plan.total_size_pct * 0.5:
            adjustments.append(PlanAdjustment(
                component="size",
                current_value=current_size,
                recommended_value=plan.total_size_pct,
                reason=f"Position only {current_size:.1f}% vs planned {plan.total_size_pct:.1f}%. Consider adding if conditions still favorable.",
                urgency="low",
                confidence=0.6
            ))
        
        # Check if over-sized
        if current_size > plan.max_size_pct:
            adjustments.append(PlanAdjustment(
                component="size",
                current_value=current_size,
                recommended_value=plan.max_size_pct,
                reason=f"Position {current_size:.1f}% exceeds max {plan.max_size_pct:.1f}%. Consider reducing.",
                urgency="high",
                confidence=0.9
            ))
        
        return adjustments
    
    def _review_time_factors(
        self,
        plan: Any,
        days_in_trade: int,
        current_price: float,
        entry_price: float
    ) -> List[PlanAdjustment]:
        """Review time-based factors"""
        adjustments = []
        
        # Calculate profit
        if plan.direction == "BUY":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100
        
        # If in trade >7 days with minimal profit, consider exiting
        if days_in_trade > 7 and abs(profit_pct) < 1.0:
            adjustments.append(PlanAdjustment(
                component="exit",
                current_value="hold",
                recommended_value="consider exit",
                reason=f"In trade {days_in_trade} days with only {profit_pct:+.1f}% profit. Consider exiting if no catalyst.",
                urgency="low",
                confidence=0.5
            ))
        
        # If in trade >14 days with good profit, consider taking some
        if days_in_trade > 14 and profit_pct > 5.0:
            adjustments.append(PlanAdjustment(
                component="target",
                current_value="hold",
                recommended_value="take partial profit",
                reason=f"In trade {days_in_trade} days with {profit_pct:+.1f}% profit. Consider taking partial profit.",
                urgency="medium",
                confidence=0.7
            ))
        
        return adjustments
    
    def format_review_report(self, review: Dict[str, Any]) -> str:
        """Format review as readable report"""
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"PLAN REVIEW - {review['review_time']}")
        lines.append(f"{'='*70}\n")
        
        lines.append(f"Total Adjustments: {review['total_adjustments']}")
        lines.append(f"  High Priority: {len(review['high_priority'])}")
        lines.append(f"  Medium Priority: {len(review['medium_priority'])}")
        lines.append(f"  Low Priority: {len(review['low_priority'])}\n")
        
        if review['high_priority']:
            lines.append(f"🔴 HIGH PRIORITY ADJUSTMENTS:")
            for adj in review['high_priority']:
                lines.append(f"\n  {adj.component.upper()}: {adj.reason}")
                lines.append(f"    Current: {adj.current_value}")
                lines.append(f"    Recommended: {adj.recommended_value}")
                lines.append(f"    Confidence: {adj.confidence:.0%}")
        
        if review['medium_priority']:
            lines.append(f"\n🟡 MEDIUM PRIORITY ADJUSTMENTS:")
            for adj in review['medium_priority']:
                lines.append(f"\n  {adj.component.upper()}: {adj.reason}")
                lines.append(f"    Current: {adj.current_value}")
                lines.append(f"    Recommended: {adj.recommended_value}")
        
        if review['low_priority']:
            lines.append(f"\n🟢 LOW PRIORITY SUGGESTIONS:")
            for adj in review['low_priority']:
                lines.append(f"  • {adj.reason}")
        
        if not review['all_adjustments']:
            lines.append(f"\n✅ No adjustments needed. Plan is on track.")
        
        lines.append(f"\n{'='*70}")
        
        return '\n'.join(lines)
