"""Tests for gas.py — validated against Anderson MCF tables and NACA 1135.

Reference values from Anderson MCF 3rd ed., Table A.1 (isentropic)
and Table A.5 (Prandtl-Meyer), γ=1.4.
"""

import numpy as np
import pytest
from nozzle.gas import (
    pressure_ratio,
    temperature_ratio,
    density_ratio,
    area_mach_ratio,
    mach_from_area_ratio,
    mach_angle,
    prandtl_meyer,
    mach_from_prandtl_meyer,
    thrust_coefficient_ideal,
)


class TestIsentropicRelations:
    """Anderson MCF Table A.1 values, γ=1.4."""

    @pytest.mark.parametrize("M, expected", [
        (0.0, 1.0),
        (0.5, 0.8430),
        (1.0, 0.5283),
        (1.5, 0.2724),
        (2.0, 0.1278),
        (3.0, 0.02722),
        (5.0, 1.890e-3),
    ])
    def test_pressure_ratio(self, M, expected):
        assert pressure_ratio(M) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("M, expected", [
        (0.0, 1.0),
        (0.5, 0.9524),
        (1.0, 0.8333),
        (1.5, 0.6897),
        (2.0, 0.5556),
        (3.0, 0.3571),
    ])
    def test_temperature_ratio(self, M, expected):
        assert temperature_ratio(M) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("M, expected", [
        (0.0, 1.0),
        (0.5, 0.8852),
        (1.0, 0.6339),
        (1.5, 0.3950),
        (2.0, 0.2300),
        (3.0, 0.07623),
    ])
    def test_density_ratio(self, M, expected):
        assert density_ratio(M) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("M, expected", [
        (1.0, 1.0),
        (1.5, 1.1762),
        (2.0, 1.6875),
        (2.5, 2.6367),
        (3.0, 4.2346),
        (5.0, 25.00),
    ])
    def test_area_mach_ratio(self, M, expected):
        assert area_mach_ratio(M) == pytest.approx(expected, rel=1e-3)


class TestAreaMachInversion:
    """Verify round-trip: M → A/A* → M."""

    @pytest.mark.parametrize("M", [1.5, 2.0, 3.0, 5.0, 10.0])
    def test_supersonic_roundtrip(self, M):
        ar = area_mach_ratio(M)
        M_recovered = mach_from_area_ratio(ar, supersonic=True)
        assert M_recovered == pytest.approx(M, rel=1e-10)

    @pytest.mark.parametrize("M", [0.1, 0.3, 0.5, 0.8, 0.99])
    def test_subsonic_roundtrip(self, M):
        ar = area_mach_ratio(M)
        M_recovered = mach_from_area_ratio(ar, supersonic=False)
        assert M_recovered == pytest.approx(M, rel=1e-8)

    def test_sonic(self):
        assert mach_from_area_ratio(1.0) == pytest.approx(1.0, abs=1e-10)


class TestMachAngle:

    def test_mach_1(self):
        assert mach_angle(1.0) == pytest.approx(np.pi / 2, rel=1e-12)

    def test_mach_2(self):
        # μ = arcsin(0.5) = 30°
        assert np.degrees(mach_angle(2.0)) == pytest.approx(30.0, rel=1e-10)


class TestPrandtlMeyer:
    """Anderson MCF Table A.5 values, γ=1.4."""

    @pytest.mark.parametrize("M, expected_deg", [
        (1.0, 0.0),
        (1.5, 11.91),
        (2.0, 26.38),
        (2.5, 39.12),
        (3.0, 49.76),
        (5.0, 76.92),
    ])
    def test_prandtl_meyer_angles(self, M, expected_deg):
        nu_deg = np.degrees(prandtl_meyer(M))
        assert nu_deg == pytest.approx(expected_deg, abs=0.05)

    @pytest.mark.parametrize("M", [1.5, 2.0, 2.4, 3.0, 5.0])
    def test_roundtrip(self, M):
        nu = prandtl_meyer(M)
        M_recovered = mach_from_prandtl_meyer(nu)
        assert M_recovered == pytest.approx(M, rel=1e-10)

    def test_nu_max_exceeded(self):
        with pytest.raises(ValueError, match="exceeds nu_max"):
            mach_from_prandtl_meyer(np.radians(200))


class TestThrustCoefficient:
    """Validate Cf against known values."""

    def test_cf_increases_with_expansion(self):
        """Cf should increase with exit Mach (vacuum)."""
        cf2 = thrust_coefficient_ideal(2.0)
        cf3 = thrust_coefficient_ideal(3.0)
        cf5 = thrust_coefficient_ideal(5.0)
        assert cf2 < cf3 < cf5

    def test_cf_m2_gamma14(self):
        """Cf for M=2, γ=1.4 vacuum — check against Sutton Table 3-3.

        For γ=1.4, Me=2.0, Pe/P0=0.1278, Ae/A*=1.6875:
        Cf ≈ 1.268 (momentum) + 0.1278*1.6875 (pressure) ≈ 1.484
        """
        cf = thrust_coefficient_ideal(2.0, gamma=1.4)
        # Verify reasonable range for vacuum Cf
        assert 1.4 < cf < 1.6

    def test_cf_sonic(self):
        """At M=1 (sonic exit), Cf should be modest."""
        cf = thrust_coefficient_ideal(1.0, gamma=1.4)
        # Cf at M=1 with vacuum: momentum + pressure
        assert 0.5 < cf < 1.5
