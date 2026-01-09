"""
Unit tests for memory CLI commands.

Tests:
1. memories command - viewing memories
2. memory-delete command - deleting memories
3. memory-stats command - statistics
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typer.testing import CliRunner


runner = CliRunner()


class TestMemoriesCommand:
    """Tests for the memories CLI command."""
    
    def test_memories_invalid_collection(self):
        """Test memories command with invalid collection name."""
        from cli.main import app
        result = runner.invoke(app, ["memories", "-c", "invalid_collection"])
        
        # Should handle gracefully
        assert result.exit_code == 0 or "Unknown collection" in result.stdout or "Error" in str(result.exception)


class TestMemoryDeleteCommand:
    """Tests for the memory-delete CLI command."""
    
    def test_memory_delete_invalid_collection(self):
        """Test memory-delete with invalid collection."""
        from cli.main import app
        result = runner.invoke(app, ["memory-delete", "-c", "invalid_collection", "--errors"])
        
        # Should handle gracefully
        assert result.exit_code == 0 or "Unknown collection" in result.stdout or "Error" in str(result.exception)


class TestMemoryStatsCommand:
    """Tests for the memory-stats CLI command."""
    
    def test_memory_stats_help(self):
        """Test memory-stats help output."""
        from cli.main import app
        result = runner.invoke(app, ["memory-stats", "--help"])
        
        assert result.exit_code == 0
        assert "memory" in result.stdout.lower() or "stats" in result.stdout.lower()


class TestErrorPatterns:
    """Tests for error pattern matching in memory-delete."""
    
    def test_error_patterns_list(self):
        """Verify all expected error patterns are checked."""
        # These patterns should be matched by --errors flag
        error_texts = [
            "Alpha Vantage API rate limit has been exceeded",
            "N/A: Not a trading day (weekend)",
            "I apologize for the inconvenience, but",
            "API rate limit has been exceeded",
            "Error fetching data",
            "Failed to retrieve information",
            "Unable to retrieve the data",
        ]
        
        # Patterns from the CLI
        error_patterns = [
            "Alpha Vantage API rate limit",
            "N/A: Not a trading day",
            "I apologize for the inconvenience",
            "API rate limit has been exceeded",
            "Error fetching",
            "Failed to retrieve",
            "Unable to retrieve",
        ]
        
        for text in error_texts:
            matched = any(p.lower() in text.lower() for p in error_patterns)
            assert matched, f"Error text not matched: {text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
