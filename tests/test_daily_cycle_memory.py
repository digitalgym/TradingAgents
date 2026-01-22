"""
Test that daily_cycle.py works with the tiered memory system.
"""

import sys
sys.path.insert(0, '.')

from examples.daily_cycle import (
    determine_tier_from_evaluation,
    TIER_SHORT, TIER_MID, TIER_LONG
)

print("Testing daily_cycle tiered memory integration...")

# Test tier determination
test_cases = [
    {"hypothetical_pnl": 0.5, "prediction_correct": False, "expected": TIER_SHORT},
    {"hypothetical_pnl": 1.0, "prediction_correct": True, "expected": TIER_MID},
    {"hypothetical_pnl": 2.0, "prediction_correct": False, "expected": TIER_MID},
    {"hypothetical_pnl": 4.0, "prediction_correct": True, "expected": TIER_LONG},
]

all_passed = True
for tc in test_cases:
    tier = determine_tier_from_evaluation(tc)
    expected = tc["expected"]
    status = "✓" if tier == expected else "✗"
    if tier != expected:
        all_passed = False
    print(f"  {status} P&L: {tc['hypothetical_pnl']:+.1f}%, Correct: {tc['prediction_correct']} -> Tier: {tier} (expected: {expected})")

# Verify tier constants
assert TIER_SHORT == "short", f"TIER_SHORT should be 'short', got {TIER_SHORT}"
assert TIER_MID == "mid", f"TIER_MID should be 'mid', got {TIER_MID}"
assert TIER_LONG == "long", f"TIER_LONG should be 'long', got {TIER_LONG}"

print()
if all_passed:
    print("✅ daily_cycle.py tiered memory integration verified!")
else:
    print("❌ Some tests failed!")
    sys.exit(1)
