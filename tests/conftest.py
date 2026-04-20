"""Shared pytest configuration."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_root: test requires CERN ROOT / PyROOT installed (skipped in basic CI)",
    )
