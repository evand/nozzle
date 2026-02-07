"""Shared fixtures for nozzle tests."""

import pytest


@pytest.fixture
def gamma_air():
    """Standard gamma for air / cold-gas."""
    return 1.4


@pytest.fixture
def gamma_exhaust():
    """Typical gamma for rocket exhaust (LOX/RP-1 ≈ 1.2)."""
    return 1.2
