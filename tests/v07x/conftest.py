"""Local pytest config for v0.7-x tests."""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that load the real fastembed model "
        "(skipped unless RUN_VEC_TESTS=1)",
    )
    config.addinivalue_line(
        "markers",
        "daemon: tests that spin up the real local daemon subprocess "
        "(skipped unless RUN_DAEMON_TESTS=1)",
    )
