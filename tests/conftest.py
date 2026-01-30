"""
Pytest configuration and fixtures for TradingAgents tests.

This conftest handles mocking of heavy dependencies (torch, transformers, OpenAI)
that can cause import failures or require API keys during testing.
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Set a fake API key to prevent OpenAI client from raising errors
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")

# Mock problematic modules BEFORE any imports that might use them
# This prevents torch/transformers from ever being loaded


class MockTransformers:
    """Mock transformers module."""
    GPT2TokenizerFast = MagicMock()


class MockTorch:
    """Mock torch module."""
    device = MagicMock()
    Tensor = MagicMock()
    no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

    @staticmethod
    def tensor(*args, **kwargs):
        return MagicMock()

    @staticmethod
    def zeros(*args, **kwargs):
        return MagicMock()


# Only mock if torch causes issues (check if it's importable)
_torch_works = False
try:
    import torch as _real_torch
    _torch_works = True
except (OSError, ImportError):
    # Torch has DLL issues or isn't installed - mock it
    sys.modules['torch'] = MockTorch()
    sys.modules['torch.cuda'] = MagicMock()
    sys.modules['torch.nn'] = MagicMock()
    sys.modules['torch.nn.functional'] = MagicMock()

# Mock transformers if torch doesn't work (transformers requires torch)
if not _torch_works:
    sys.modules['transformers'] = MockTransformers()
    sys.modules['transformers.utils'] = MagicMock()
    sys.modules['transformers.utils.generic'] = MagicMock()


# Mock chromadb PersistentClient
class MockCollection:
    """Mock ChromaDB collection with where clause support."""
    def __init__(self, name="test"):
        self.name = name
        self._data = {"ids": [], "embeddings": [], "metadatas": [], "documents": []}

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        for i, id_ in enumerate(ids):
            self._data["ids"].append(id_)
            self._data["embeddings"].append(embeddings[i] if embeddings else None)
            self._data["metadatas"].append(metadatas[i] if metadatas else {})
            self._data["documents"].append(documents[i] if documents else "")

    def _matches_where(self, metadata, where):
        """Check if metadata matches where clause."""
        if not where:
            return True
        for key, condition in where.items():
            if isinstance(condition, dict):
                # Handle operators like {"$eq": value}
                for op, value in condition.items():
                    actual = metadata.get(key)
                    if op == "$eq" and actual != value:
                        return False
                    elif op == "$ne" and actual == value:
                        return False
                    elif op == "$gt" and (actual is None or actual <= value):
                        return False
                    elif op == "$gte" and (actual is None or actual < value):
                        return False
                    elif op == "$lt" and (actual is None or actual >= value):
                        return False
                    elif op == "$lte" and (actual is None or actual > value):
                        return False
            else:
                # Direct equality
                if metadata.get(key) != condition:
                    return False
        return True

    def query(self, query_embeddings=None, n_results=5, where=None, include=None):
        # Filter by where clause
        filtered_indices = []
        for i, metadata in enumerate(self._data["metadatas"]):
            if self._matches_where(metadata, where):
                filtered_indices.append(i)

        n = min(n_results, len(filtered_indices))
        indices = filtered_indices[:n]

        return {
            "ids": [[self._data["ids"][i] for i in indices]],
            "distances": [[0.1 * j for j in range(len(indices))]],
            "metadatas": [[self._data["metadatas"][i] for i in indices]],
            "documents": [[self._data["documents"][i] for i in indices]],
        }

    def get(self, ids=None, where=None, include=None, limit=None, offset=None):
        result = {"ids": [], "metadatas": [], "documents": []}

        if ids:
            for i, id_ in enumerate(self._data["ids"]):
                if id_ in ids:
                    result["ids"].append(id_)
                    result["metadatas"].append(self._data["metadatas"][i])
                    result["documents"].append(self._data["documents"][i])
        elif where:
            # Filter by where clause
            for i, metadata in enumerate(self._data["metadatas"]):
                if self._matches_where(metadata, where):
                    result["ids"].append(self._data["ids"][i])
                    result["metadatas"].append(metadata)
                    result["documents"].append(self._data["documents"][i])
        else:
            result = {
                "ids": self._data["ids"][:],
                "metadatas": self._data["metadatas"][:],
                "documents": self._data["documents"][:],
            }

        # Apply offset and limit
        if offset:
            result["ids"] = result["ids"][offset:]
            result["metadatas"] = result["metadatas"][offset:]
            result["documents"] = result["documents"][offset:]
        if limit:
            result["ids"] = result["ids"][:limit]
            result["metadatas"] = result["metadatas"][:limit]
            result["documents"] = result["documents"][:limit]

        return result

    def update(self, ids, metadatas=None, documents=None):
        for i, id_ in enumerate(ids):
            if id_ in self._data["ids"]:
                idx = self._data["ids"].index(id_)
                if metadatas:
                    self._data["metadatas"][idx] = metadatas[i]
                if documents:
                    self._data["documents"][idx] = documents[i]

    def count(self):
        return len(self._data["ids"])


class MockChromaClient:
    """Mock ChromaDB client."""
    def __init__(self, path=None, settings=None):
        self._collections = {}

    def get_or_create_collection(self, name, embedding_function=None):
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]

    def list_collections(self):
        return list(self._collections.values())


# Provide the mock classes for tests to use
import pytest


@pytest.fixture
def mock_chroma_client():
    """Fixture providing a mock ChromaDB client."""
    return MockChromaClient()


@pytest.fixture
def mock_collection():
    """Fixture providing a mock ChromaDB collection."""
    return MockCollection()


@pytest.fixture(autouse=True)
def mock_openai_embeddings():
    """Auto-fixture to mock OpenAI embeddings API calls."""
    import hashlib

    def fake_embedding(text):
        """Generate deterministic fake embedding from text."""
        h = hashlib.md5(text.encode()).hexdigest()
        embedding = []
        for i in range(0, min(len(h) * 4, 1536), 4):
            idx = i % len(h)
            val = int(h[idx:idx+2], 16) / 255.0 - 0.5
            embedding.append(val)
        while len(embedding) < 1536:
            embedding.append(0.0)
        return embedding[:1536]

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding("test"))]

    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        yield mock_openai
