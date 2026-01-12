"""
Risk Guardrails and Circuit Breakers

Hard risk limits to prevent catastrophic losses:
- Daily loss limit (3% of account)
- Consecutive loss limit (2 losses in a row)
- Maximum position size limits
- Cooldown periods after breaches
"""

import os
import json
import pickle
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class RiskGuardrails:
    """Enforce hard risk limits and circuit breakers"""
    
    def __init__(
        self,
        state_file: Optional[str] = None,
        daily_loss_limit_pct: float = 3.0,
        max_consecutive_losses: int = 2,
        max_position_size_pct: float = 2.0,
        cooldown_hours: int = 24
    ):
        """
        Initialize risk guardrails.
        
        Args:
            state_file: Path to save/load state (default: tradingagents/examples/risk_state.pkl)
            daily_loss_limit_pct: Maximum daily loss as % of account (default 3%)
            max_consecutive_losses: Max consecutive losses before cooldown (default 2)
            max_position_size_pct: Max position size as % of account (default 2%)
            cooldown_hours: Hours to wait after breach (default 24)
        """
        if state_file is None:
            base_dir = Path(__file__).parent.parent.parent
            self.state_file = base_dir / "examples" / "risk_state.pkl"
        else:
            self.state_file = Path(state_file)
        
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_position_size_pct = max_position_size_pct
        self.cooldown_hours = cooldown_hours
        
        # Load or initialize state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or return defaults."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        return {
            "consecutive_losses": 0,
            "daily_loss_pct": 0.0,
            "last_trade_date": None,
            "cooldown_until": None,
            "breach_history": [],
            "total_breaches": 0
        }
    
    def _save_state(self):
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)
    
    def check_can_trade(self, account_balance: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on guardrails.
        
        Args:
            account_balance: Current account balance
        
        Returns:
            (can_trade, reason)
        """
        # Check cooldown
        if self.state["cooldown_until"]:
            cooldown_end = datetime.fromisoformat(self.state["cooldown_until"])
            if datetime.now() < cooldown_end:
                remaining = cooldown_end - datetime.now()
                hours = remaining.total_seconds() / 3600
                return False, f"COOLDOWN: {hours:.1f} hours remaining"
        
        # Check daily loss limit
        if self.state["daily_loss_pct"] >= self.daily_loss_limit_pct:
            return False, f"DAILY LOSS LIMIT: {self.state['daily_loss_pct']:.2f}% >= {self.daily_loss_limit_pct}%"
        
        # Check consecutive losses
        if self.state["consecutive_losses"] >= self.max_consecutive_losses:
            return False, f"CONSECUTIVE LOSSES: {self.state['consecutive_losses']} >= {self.max_consecutive_losses}"
        
        return True, "OK"
    
    def validate_position_size(
        self,
        position_size_pct: float,
        account_balance: float
    ) -> Tuple[bool, str, float]:
        """
        Validate and potentially adjust position size.
        
        Args:
            position_size_pct: Requested position size as % of account
            account_balance: Current account balance
        
        Returns:
            (is_valid, reason, adjusted_size_pct)
        """
        if position_size_pct <= self.max_position_size_pct:
            return True, "OK", position_size_pct
        
        # Cap at maximum
        adjusted = self.max_position_size_pct
        return False, f"POSITION TOO LARGE: Capped at {self.max_position_size_pct}%", adjusted
    
    def record_trade_result(
        self,
        was_win: bool,
        pnl_pct: float,
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Record trade result and update guardrail state.
        
        Args:
            was_win: Whether trade was profitable
            pnl_pct: P&L as % of account
            account_balance: Current account balance
        
        Returns:
            {
                "breach_triggered": bool,
                "breach_type": str or None,
                "cooldown_until": str or None,
                "status": str
            }
        """
        today = datetime.now().date().isoformat()
        
        # Reset daily loss if new day
        if self.state["last_trade_date"] != today:
            self.state["daily_loss_pct"] = 0.0
            self.state["last_trade_date"] = today
        
        # Update consecutive losses
        if was_win:
            self.state["consecutive_losses"] = 0
        else:
            self.state["consecutive_losses"] += 1
        
        # Update daily loss (only count losses)
        if pnl_pct < 0:
            self.state["daily_loss_pct"] += abs(pnl_pct)
        
        # Check for breaches
        breach_triggered = False
        breach_type = None
        
        # Daily loss breach
        if self.state["daily_loss_pct"] >= self.daily_loss_limit_pct:
            breach_triggered = True
            breach_type = "daily_loss_limit"
            self._trigger_cooldown(breach_type)
        
        # Consecutive loss breach
        elif self.state["consecutive_losses"] >= self.max_consecutive_losses:
            breach_triggered = True
            breach_type = "consecutive_losses"
            self._trigger_cooldown(breach_type)
        
        # Save state
        self._save_state()
        
        return {
            "breach_triggered": breach_triggered,
            "breach_type": breach_type,
            "cooldown_until": self.state.get("cooldown_until"),
            "status": self._get_status_summary()
        }
    
    def _trigger_cooldown(self, breach_type: str):
        """Trigger cooldown period."""
        cooldown_end = datetime.now() + timedelta(hours=self.cooldown_hours)
        self.state["cooldown_until"] = cooldown_end.isoformat()
        
        # Record breach
        breach_record = {
            "timestamp": datetime.now().isoformat(),
            "type": breach_type,
            "cooldown_until": cooldown_end.isoformat(),
            "consecutive_losses": self.state["consecutive_losses"],
            "daily_loss_pct": self.state["daily_loss_pct"]
        }
        
        self.state["breach_history"].append(breach_record)
        self.state["total_breaches"] += 1
        
        # Keep only last 20 breaches
        self.state["breach_history"] = self.state["breach_history"][-20:]
    
    def _get_status_summary(self) -> str:
        """Get current status summary."""
        lines = []
        
        # Cooldown status
        if self.state["cooldown_until"]:
            cooldown_end = datetime.fromisoformat(self.state["cooldown_until"])
            if datetime.now() < cooldown_end:
                remaining = cooldown_end - datetime.now()
                hours = remaining.total_seconds() / 3600
                lines.append(f"⛔ COOLDOWN: {hours:.1f}h remaining")
        
        # Daily loss
        daily_pct = self.state["daily_loss_pct"]
        daily_limit = self.daily_loss_limit_pct
        if daily_pct > 0:
            lines.append(f"📉 Daily Loss: {daily_pct:.2f}% / {daily_limit}%")
        
        # Consecutive losses
        consec = self.state["consecutive_losses"]
        if consec > 0:
            lines.append(f"📊 Consecutive Losses: {consec} / {self.max_consecutive_losses}")
        
        if not lines:
            return "✅ All systems normal"
        
        return " | ".join(lines)
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status."""
        can_trade, reason = self.check_can_trade(0)  # Balance not needed for status
        
        return {
            "can_trade": can_trade,
            "reason": reason,
            "consecutive_losses": self.state["consecutive_losses"],
            "daily_loss_pct": self.state["daily_loss_pct"],
            "cooldown_until": self.state.get("cooldown_until"),
            "total_breaches": self.state["total_breaches"],
            "status_summary": self._get_status_summary()
        }
    
    def reset_cooldown(self):
        """Manually reset cooldown (use with caution)."""
        self.state["cooldown_until"] = None
        self._save_state()
    
    def reset_daily_loss(self):
        """Manually reset daily loss counter."""
        self.state["daily_loss_pct"] = 0.0
        self.state["last_trade_date"] = datetime.now().date().isoformat()
        self._save_state()
    
    def reset_consecutive_losses(self):
        """Manually reset consecutive loss counter."""
        self.state["consecutive_losses"] = 0
        self._save_state()
    
    def get_breach_history(self, n: int = 10) -> list:
        """Get recent breach history."""
        return self.state["breach_history"][-n:]
    
    def format_report(self) -> str:
        """Format status as human-readable report."""
        status = self.get_status()
        
        report = f"""
RISK GUARDRAILS STATUS

Trading Allowed: {"✅ YES" if status['can_trade'] else "⛔ NO"}
Reason: {status['reason']}

Current Metrics:
- Consecutive Losses: {status['consecutive_losses']} / {self.max_consecutive_losses}
- Daily Loss: {status['daily_loss_pct']:.2f}% / {self.daily_loss_limit_pct}%
- Total Breaches: {status['total_breaches']}

Limits:
- Daily Loss Limit: {self.daily_loss_limit_pct}%
- Max Consecutive Losses: {self.max_consecutive_losses}
- Max Position Size: {self.max_position_size_pct}%
- Cooldown Period: {self.cooldown_hours} hours

Status: {status['status_summary']}
"""
        
        # Add cooldown info if active
        if status['cooldown_until']:
            cooldown_end = datetime.fromisoformat(status['cooldown_until'])
            if datetime.now() < cooldown_end:
                remaining = cooldown_end - datetime.now()
                hours = remaining.total_seconds() / 3600
                report += f"\n⏰ Cooldown ends in {hours:.1f} hours ({cooldown_end.strftime('%Y-%m-%d %H:%M')})\n"
        
        # Add recent breaches
        breaches = self.get_breach_history(5)
        if breaches:
            report += "\nRecent Breaches:\n"
            for breach in breaches:
                timestamp = datetime.fromisoformat(breach['timestamp']).strftime('%Y-%m-%d %H:%M')
                report += f"  {timestamp}: {breach['type']}\n"
        
        return report
