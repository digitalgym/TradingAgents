"""
Trading Plan Parser

Extracts structured data from analyst trading plans:
- Entry levels and conditions
- Stop loss levels (static and dynamic)
- Take profit targets
- Position sizing
- Monitoring triggers
- Exit conditions
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EntryTranche:
    """Single entry tranche"""
    tranche_number: int
    size_pct: float
    price_level: float
    price_range: Optional[tuple] = None  # (min, max)
    conditions: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class StopLoss:
    """Stop loss configuration"""
    initial_price: float
    type: str = "static"  # static, trailing, dynamic
    atr_multiple: Optional[float] = None
    trail_rules: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)


@dataclass
class TakeProfit:
    """Take profit target"""
    target_number: int
    price_level: float
    size_pct: float  # % of position to close
    description: str = ""


@dataclass
class MonitoringTrigger:
    """Monitoring condition"""
    category: str  # daily, geo, fundamental, technical
    indicator: str
    condition: str
    action: str
    description: str = ""


@dataclass
class ParsedTradingPlan:
    """Complete parsed trading plan"""
    symbol: str
    direction: str  # BUY or SELL
    
    # Position sizing
    total_size_pct: float
    max_size_pct: float
    
    # Entry strategy
    entry_tranches: List[EntryTranche]
    entry_type: str = "staged"  # market, limit, staged
    
    # Risk management
    stop_loss: StopLoss
    take_profits: List[TakeProfit]
    
    # Monitoring
    monitoring_triggers: List[MonitoringTrigger]
    
    # Exit conditions
    exit_conditions: List[str]
    
    # Hedge
    hedge_symbol: Optional[str] = None
    hedge_size_pct: Optional[float] = None
    
    # Raw plan text
    raw_plan: str = ""


class TradingPlanParser:
    """
    Parses trading plans from analyst output text.
    
    Extracts:
    - Entry levels and tranches
    - Stop loss (static/dynamic/trailing)
    - Take profit targets
    - Position sizing
    - Monitoring triggers
    - Exit conditions
    """
    
    def __init__(self):
        self.price_pattern = r'\$(\d+(?:\.\d+)?)'
        self.percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        self.range_pattern = r'\$(\d+(?:\.\d+)?)-\$?(\d+(?:\.\d+)?)'
    
    def parse_plan(self, plan_text: str, symbol: str = None) -> ParsedTradingPlan:
        """
        Parse a trading plan from text.
        
        Args:
            plan_text: The trading plan text (from analyst output)
            symbol: Optional symbol override
        
        Returns:
            ParsedTradingPlan with all extracted data
        """
        
        # Extract symbol if not provided
        if symbol is None:
            symbol = self._extract_symbol(plan_text)
        
        # Determine direction
        direction = self._extract_direction(plan_text)
        
        # Extract position sizing
        total_size, max_size = self._extract_position_sizing(plan_text)
        
        # Extract entry tranches
        entry_tranches = self._extract_entry_tranches(plan_text)
        
        # Determine entry type
        entry_type = "staged" if len(entry_tranches) > 1 else "limit"
        
        # Extract stop loss
        stop_loss = self._extract_stop_loss(plan_text)
        
        # Extract take profits
        take_profits = self._extract_take_profits(plan_text)
        
        # Extract monitoring triggers
        monitoring_triggers = self._extract_monitoring_triggers(plan_text)
        
        # Extract exit conditions
        exit_conditions = self._extract_exit_conditions(plan_text)
        
        # Extract hedge
        hedge_symbol, hedge_size = self._extract_hedge(plan_text)
        
        return ParsedTradingPlan(
            symbol=symbol,
            direction=direction,
            total_size_pct=total_size,
            max_size_pct=max_size,
            entry_tranches=entry_tranches,
            entry_type=entry_type,
            stop_loss=stop_loss,
            take_profits=take_profits,
            monitoring_triggers=monitoring_triggers,
            exit_conditions=exit_conditions,
            hedge_symbol=hedge_symbol,
            hedge_size_pct=hedge_size,
            raw_plan=plan_text
        )
    
    def _extract_symbol(self, text: str) -> str:
        """Extract symbol from text"""
        # Look for common patterns
        patterns = [
            r'(?:symbol|ticker|asset):\s*([A-Z]+)',
            r'\b([A-Z]{3,6})\b',  # 3-6 letter uppercase
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return "UNKNOWN"
    
    def _extract_direction(self, text: str) -> str:
        """Extract trade direction"""
        text_lower = text.lower()
        
        # Look for buy/long signals
        buy_keywords = ['buy', 'long', 'bullish', 'upside', 'entry (staged']
        sell_keywords = ['sell', 'short', 'bearish', 'downside']
        
        buy_count = sum(1 for kw in buy_keywords if kw in text_lower)
        sell_count = sum(1 for kw in sell_keywords if kw in text_lower)
        
        return "BUY" if buy_count > sell_count else "SELL"
    
    def _extract_position_sizing(self, text: str) -> tuple:
        """Extract total and max position size"""
        
        # Look for sizing patterns
        # "Cap at 4-5% total"
        # "3-5% sizing to 7-8% max"
        # "Start 2.5-3% initial"
        
        total_size = 3.0  # Default
        max_size = 5.0    # Default
        
        # Cap/max pattern
        cap_match = re.search(r'cap\s+at\s+(\d+(?:\.\d+)?)-?(\d+(?:\.\d+)?)?\s*%', text, re.IGNORECASE)
        if cap_match:
            max_size = float(cap_match.group(2) or cap_match.group(1))
        
        # Initial sizing
        initial_match = re.search(r'(?:start|initial)\s+(\d+(?:\.\d+)?)-?(\d+(?:\.\d+)?)?\s*%', text, re.IGNORECASE)
        if initial_match:
            total_size = float(initial_match.group(1))
        
        # Total sizing
        total_match = re.search(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*%\s+(?:total|sizing)', text, re.IGNORECASE)
        if total_match:
            total_size = float(total_match.group(1))
            max_size = float(total_match.group(2))
        
        return total_size, max_size
    
    def _extract_entry_tranches(self, text: str) -> List[EntryTranche]:
        """Extract entry tranches from plan"""
        tranches = []
        
        # Look for tranche patterns
        # "Tranche 1: 2% at $77-80"
        # "Tranche 2: 1.5% at $75"
        
        tranche_pattern = r'Tranche\s+(\d+):\s+(\d+(?:\.\d+)?)\s*%\s+at\s+\$(\d+(?:\.\d+)?)-?\$?(\d+(?:\.\d+)?)?'
        
        for match in re.finditer(tranche_pattern, text, re.IGNORECASE):
            tranche_num = int(match.group(1))
            size_pct = float(match.group(2))
            price_low = float(match.group(3))
            price_high = float(match.group(4)) if match.group(4) else price_low
            
            # Extract conditions (text after the price until next line or period)
            start_pos = match.end()
            end_pos = text.find('\n', start_pos)
            if end_pos == -1:
                end_pos = text.find('.', start_pos)
            if end_pos == -1:
                end_pos = len(text)
            
            condition_text = text[start_pos:end_pos].strip()
            conditions = [c.strip() for c in condition_text.split('+') if c.strip()]
            
            # Get description (parenthetical text)
            desc_match = re.search(r'\((.*?)\)', condition_text)
            description = desc_match.group(1) if desc_match else condition_text
            
            tranches.append(EntryTranche(
                tranche_number=tranche_num,
                size_pct=size_pct,
                price_level=(price_low + price_high) / 2,
                price_range=(price_low, price_high) if price_high > price_low else None,
                conditions=conditions,
                description=description
            ))
        
        # If no tranches found, create single entry from general text
        if not tranches:
            # Look for entry price
            entry_match = re.search(r'entry\s+(?:at\s+)?\$(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if entry_match:
                price = float(entry_match.group(1))
                tranches.append(EntryTranche(
                    tranche_number=1,
                    size_pct=3.0,  # Default
                    price_level=price,
                    description="Single entry"
                ))
        
        return tranches
    
    def _extract_stop_loss(self, text: str) -> StopLoss:
        """Extract stop loss configuration"""
        
        # Look for stop loss patterns
        # "Stop-Loss: Dynamic trailing, 1.5-2x ATR (~$72 initial"
        # "trails to $75 post-$82 close"
        # "Full exit <$72"
        
        stop_type = "static"
        initial_price = 0.0
        atr_multiple = None
        trail_rules = []
        exit_conditions = []
        
        # Check for dynamic/trailing
        if re.search(r'dynamic|trailing', text, re.IGNORECASE):
            stop_type = "trailing"
        
        # Extract ATR multiple
        atr_match = re.search(r'(\d+(?:\.\d+)?)-?(\d+(?:\.\d+)?)?\s*x?\s*ATR', text, re.IGNORECASE)
        if atr_match:
            atr_multiple = float(atr_match.group(1))
            stop_type = "dynamic"
        
        # Extract initial stop price
        initial_match = re.search(r'(?:stop|SL).*?\$(\d+(?:\.\d+)?)\s+initial', text, re.IGNORECASE)
        if not initial_match:
            initial_match = re.search(r'stop.*?at\s+\$(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if initial_match:
            initial_price = float(initial_match.group(1))
        
        # Extract trailing rules
        trail_pattern = r'trails?\s+to\s+\$(\d+(?:\.\d+)?)\s+(?:post|after)[- ]?\$(\d+(?:\.\d+)?)'
        for match in re.finditer(trail_pattern, text, re.IGNORECASE):
            trail_to = float(match.group(1))
            trigger_price = float(match.group(2))
            trail_rules.append(f"Trail to ${trail_to} after ${trigger_price}")
        
        # Extract exit conditions
        exit_pattern = r'(?:full\s+)?exit\s+(?:if\s+)?[<>]?\s*\$?(\d+(?:\.\d+)?)'
        for match in re.finditer(exit_pattern, text, re.IGNORECASE):
            exit_price = float(match.group(1))
            exit_conditions.append(f"Exit below ${exit_price}")
        
        # RSI exit condition
        rsi_match = re.search(r'RSI\s*<\s*(\d+)', text, re.IGNORECASE)
        if rsi_match:
            rsi_level = int(rsi_match.group(1))
            exit_conditions.append(f"Exit if RSI < {rsi_level}")
        
        return StopLoss(
            initial_price=initial_price,
            type=stop_type,
            atr_multiple=atr_multiple,
            trail_rules=trail_rules,
            exit_conditions=exit_conditions
        )
    
    def _extract_take_profits(self, text: str) -> List[TakeProfit]:
        """Extract take profit targets"""
        targets = []
        
        # Look for profit-taking patterns
        # "30% at $87"
        # "30% at $100 (consensus short-term)"
        # "Trail 40% to $108-125"
        
        tp_pattern = r'(\d+)\s*%\s+at\s+\$(\d+(?:\.\d+)?)'
        
        target_num = 1
        for match in re.finditer(tp_pattern, text, re.IGNORECASE):
            size_pct = float(match.group(1))
            price = float(match.group(2))
            
            # Get description
            start_pos = match.end()
            end_pos = text.find('\n', start_pos)
            if end_pos == -1:
                end_pos = text.find('.', start_pos)
            if end_pos == -1:
                end_pos = start_pos + 100
            
            desc_text = text[start_pos:end_pos].strip()
            desc_match = re.search(r'\((.*?)\)', desc_text)
            description = desc_match.group(1) if desc_match else desc_text[:50]
            
            targets.append(TakeProfit(
                target_number=target_num,
                price_level=price,
                size_pct=size_pct,
                description=description
            ))
            target_num += 1
        
        # Look for trailing profit
        trail_match = re.search(r'trail\s+(\d+)\s*%\s+to\s+\$(\d+(?:\.\d+)?)-?\$?(\d+(?:\.\d+)?)?', text, re.IGNORECASE)
        if trail_match:
            size_pct = float(trail_match.group(1))
            price_low = float(trail_match.group(2))
            price_high = float(trail_match.group(3)) if trail_match.group(3) else price_low
            
            targets.append(TakeProfit(
                target_number=target_num,
                price_level=(price_low + price_high) / 2,
                size_pct=size_pct,
                description=f"Trailing target ${price_low}-${price_high}"
            ))
        
        return targets
    
    def _extract_monitoring_triggers(self, text: str) -> List[MonitoringTrigger]:
        """Extract monitoring triggers and conditions"""
        triggers = []
        
        # Look for monitoring section
        monitoring_match = re.search(r'Monitoring.*?:(.*?)(?:Exit|Hedge|$)', text, re.DOTALL | re.IGNORECASE)
        if not monitoring_match:
            return triggers
        
        monitoring_text = monitoring_match.group(1)
        
        # Daily triggers
        # "Daily: DXY (trim 25% on +1% surge)"
        daily_pattern = r'Daily:\s*(.*?)(?:Geo|Fundamentals|Hedge|•|\n\n)'
        daily_match = re.search(daily_pattern, monitoring_text, re.DOTALL | re.IGNORECASE)
        if daily_match:
            daily_text = daily_match.group(1)
            
            # Parse individual daily triggers
            # "DXY (trim 25% on +1% surge)"
            trigger_pattern = r'([A-Z]+)\s*\((.*?)\)'
            for match in re.finditer(trigger_pattern, daily_text):
                indicator = match.group(1)
                action_text = match.group(2)
                
                triggers.append(MonitoringTrigger(
                    category="daily",
                    indicator=indicator,
                    condition=action_text,
                    action=self._extract_action(action_text),
                    description=action_text
                ))
        
        # Geo/Events triggers
        geo_pattern = r'Geo/Events:\s*(.*?)(?:Fundamentals|Hedge|•|\n\n)'
        geo_match = re.search(geo_pattern, monitoring_text, re.DOTALL | re.IGNORECASE)
        if geo_match:
            geo_text = geo_match.group(1)
            triggers.append(MonitoringTrigger(
                category="geo",
                indicator="geopolitical",
                condition=geo_text.strip(),
                action="scale",
                description=geo_text.strip()[:100]
            ))
        
        # Fundamental triggers
        fund_pattern = r'Fundamentals:\s*(.*?)(?:Hedge|Exit|•|\n\n)'
        fund_match = re.search(fund_pattern, monitoring_text, re.DOTALL | re.IGNORECASE)
        if fund_match:
            fund_text = fund_match.group(1)
            triggers.append(MonitoringTrigger(
                category="fundamental",
                indicator="fundamentals",
                condition=fund_text.strip(),
                action="scale",
                description=fund_text.strip()[:100]
            ))
        
        return triggers
    
    def _extract_action(self, text: str) -> str:
        """Extract action from trigger text"""
        text_lower = text.lower()
        
        if 'trim' in text_lower or 'reduce' in text_lower or 'scale out' in text_lower:
            return "reduce"
        elif 'add' in text_lower or 'increase' in text_lower or 'scale in' in text_lower:
            return "add"
        elif 'exit' in text_lower or 'close' in text_lower:
            return "exit"
        else:
            return "monitor"
    
    def _extract_exit_conditions(self, text: str) -> List[str]:
        """Extract exit conditions"""
        conditions = []
        
        # Look for exit contingencies section
        exit_match = re.search(r'Exit\s+Contingencies:\s*(.*?)(?:\n\n|$)', text, re.DOTALL | re.IGNORECASE)
        if exit_match:
            exit_text = exit_match.group(1)
            
            # Split by semicolon or "or"
            parts = re.split(r';|\s+or\s+', exit_text)
            for part in parts:
                part = part.strip()
                if part and len(part) > 10:
                    conditions.append(part)
        
        return conditions
    
    def _extract_hedge(self, text: str) -> tuple:
        """Extract hedge information"""
        hedge_symbol = None
        hedge_size = None
        
        # Look for hedge section
        # "Hedge: 1.5-2% gold (XAUUSD ETF)"
        hedge_match = re.search(r'Hedge:\s+(\d+(?:\.\d+)?)-?(\d+(?:\.\d+)?)?\s*%\s+(\w+)\s*\(([A-Z]+)', text, re.IGNORECASE)
        if hedge_match:
            hedge_size = float(hedge_match.group(1))
            hedge_symbol = hedge_match.group(4)
        
        return hedge_symbol, hedge_size
    
    def format_plan_summary(self, plan: ParsedTradingPlan) -> str:
        """Format parsed plan as readable summary"""
        
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"PARSED TRADING PLAN: {plan.symbol} {plan.direction}")
        lines.append(f"{'='*70}\n")
        
        # Position sizing
        lines.append(f"POSITION SIZING:")
        lines.append(f"  Total: {plan.total_size_pct:.1f}% | Max: {plan.max_size_pct:.1f}%\n")
        
        # Entry strategy
        lines.append(f"ENTRY STRATEGY ({plan.entry_type.upper()}):")
        for tranche in plan.entry_tranches:
            if tranche.price_range:
                price_str = f"${tranche.price_range[0]:.2f}-${tranche.price_range[1]:.2f}"
            else:
                price_str = f"${tranche.price_level:.2f}"
            lines.append(f"  Tranche {tranche.tranche_number}: {tranche.size_pct:.1f}% at {price_str}")
            if tranche.conditions:
                lines.append(f"    Conditions: {', '.join(tranche.conditions[:2])}")
        lines.append("")
        
        # Stop loss
        lines.append(f"STOP LOSS ({plan.stop_loss.type.upper()}):")
        lines.append(f"  Initial: ${plan.stop_loss.initial_price:.2f}")
        if plan.stop_loss.atr_multiple:
            lines.append(f"  ATR Multiple: {plan.stop_loss.atr_multiple:.1f}x")
        for rule in plan.stop_loss.trail_rules[:2]:
            lines.append(f"  {rule}")
        lines.append("")
        
        # Take profits
        lines.append(f"TAKE PROFIT TARGETS:")
        for tp in plan.take_profits:
            lines.append(f"  TP{tp.target_number}: {tp.size_pct:.0f}% at ${tp.price_level:.2f}")
            if tp.description:
                lines.append(f"    ({tp.description[:50]})")
        lines.append("")
        
        # Monitoring
        if plan.monitoring_triggers:
            lines.append(f"KEY MONITORING TRIGGERS:")
            for trigger in plan.monitoring_triggers[:3]:
                lines.append(f"  {trigger.category.upper()}: {trigger.indicator} → {trigger.action}")
            lines.append("")
        
        # Hedge
        if plan.hedge_symbol:
            lines.append(f"HEDGE: {plan.hedge_size_pct:.1f}% in {plan.hedge_symbol}\n")
        
        lines.append(f"{'='*70}")
        
        return '\n'.join(lines)
