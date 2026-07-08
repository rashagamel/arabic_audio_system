"""
tests/conftest.py
=================
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

# Ensure project root is on the path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skip with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_models: marks tests that download large models"
    )
