"""Run trade_commodities.py with suppressed verbose output to see SMC results."""
import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress LangChain's pretty_print by monkeypatching
from langchain_core.messages.base import BaseMessage

original_pretty_print = BaseMessage.pretty_print

def quiet_pretty_print(self, **kwargs):
    """Suppress emoji output that causes encoding errors."""
    try:
        original_pretty_print(self, **kwargs)
    except UnicodeEncodeError:
        # Silently skip if encoding fails
        pass

BaseMessage.pretty_print = quiet_pretty_print

# Now run the main script
if __name__ == "__main__":
    # Import after monkeypatch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from examples.trade_commodities import main

    main()
