"""Tests for exit_plane_integral and synthesize_exit_plane.

Validates the unified Cf integration against analytical results
and cross-checks between methods.
"""

import numpy as np
import pytest
from nozzle.analysis import (
    exit_plane_integral, synthesize_exit_plane,
    moc_performance, quasi_1d_performance,
)
from nozzle.gas import thrust_coefficient_ideal, mach_from_area_ratio
from nozzle.contours import minimum_length_nozzle, truncated_ideal_contour


class TestExitPlaneIntegral:

    def test_uniform_parallel_gives_cf_ideal(self):
        """Uniform M, theta=0 should give Cf_ideal exactly."""
        M_exit = 2.0
        AR = 1.6875  # area ratio for M=2 at gamma=1.4
        y_exit = np.sqrt(AR)
        y = np.linspace(0, y_exit, 500)
        M = np.full_like(y, M_exit)
        theta = np.zeros_like(y)

        Cf = exit_plane_integral(y, M, theta, gamma=1.4)
        Cf_ideal = thrust_coefficient_ideal(M_exit, gamma=1.4)
        assert Cf == pytest.approx(Cf_ideal, rel=1e-4)

    def test_conical_15deg_matches_analytical(self):
        """Conical 15 deg: integral lambda matches (1+cos alpha)/2."""
        alpha = np.radians(15)
        AR = 10
        M = mach_from_area_ratio(AR)
        y_exit = np.sqrt(AR)
        y = np.linspace(0, y_exit, 500)
        M_arr = np.full_like(y, M)
        theta = alpha * y / y_exit

        Cf = exit_plane_integral(y, M_arr, theta, 1.4)
        Cf_ideal = thrust_coefficient_ideal(M, 1.4)
        lam_integral = Cf / Cf_ideal
        lam_analytical = (1 + np.cos(alpha)) / 2

        assert lam_integral == pytest.approx(lam_analytical, rel=0.002)

    def test_conical_30deg_matches_analytical(self):
        """Conical 30 deg: integral lambda matches (1+cos alpha)/2."""
        alpha = np.radians(30)
        AR = 10
        M = mach_from_area_ratio(AR)
        y_exit = np.sqrt(AR)
        y = np.linspace(0, y_exit, 500)
        M_arr = np.full_like(y, M)
        theta = alpha * y / y_exit

        Cf = exit_plane_integral(y, M_arr, theta, 1.4)
        Cf_ideal = thrust_coefficient_ideal(M, 1.4)
        lam_integral = Cf / Cf_ideal
        lam_analytical = (1 + np.cos(alpha)) / 2

        assert lam_integral == pytest.approx(lam_analytical, rel=0.002)

    def test_larger_angle_lower_cf(self):
        """Larger wall angle should give lower Cf (more divergence loss)."""
        AR = 10
        M = mach_from_area_ratio(AR)
        y_exit = np.sqrt(AR)
        y = np.linspace(0, y_exit, 500)
        M_arr = np.full_like(y, M)

        theta_10 = np.radians(10) * y / y_exit
        theta_25 = np.radians(25) * y / y_exit

        Cf_10 = exit_plane_integral(y, M_arr, theta_10, 1.4)
        Cf_25 = exit_plane_integral(y, M_arr, theta_25, 1.4)

        assert Cf_10 > Cf_25

    def test_zero_angle_equals_no_angle(self):
        """theta=0 everywhere should match zero-array theta."""
        M_exit = 3.0
        AR = mach_from_area_ratio.__wrapped__(M_exit) if hasattr(mach_from_area_ratio, '__wrapped__') else None
        # Just use a known AR
        from nozzle.gas import area_mach_ratio
        AR = area_mach_ratio(M_exit, 1.4)
        y_exit = np.sqrt(AR)

        y = np.linspace(0, y_exit, 300)
        M = np.full_like(y, M_exit)
        theta_zeros = np.zeros_like(y)

        Cf = exit_plane_integral(y, M, theta_zeros, 1.4)
        Cf_ideal = thrust_coefficient_ideal(M_exit, 1.4)
        assert Cf == pytest.approx(Cf_ideal, rel=1e-4)


class TestSynthesizeExitPlane:

    def test_axis_theta_zero(self):
        """theta at y=0 (axis) should be 0."""
        x = np.linspace(0, 5, 100)
        y = 1.0 + 0.3 * x  # simple diverging
        y_ep, M_ep, theta_ep = synthesize_exit_plane(x, y)
        assert theta_ep[0] == pytest.approx(0.0, abs=1e-10)

    def test_wall_theta_matches_slope(self):
        """theta at wall should match arctan(dy/dx) from last segment."""
        x = np.linspace(0, 5, 100)
        y = 1.0 + 0.3 * x
        y_ep, M_ep, theta_ep = synthesize_exit_plane(x, y)

        # Expected wall angle
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        expected = np.arctan2(dy, dx)
        assert theta_ep[-1] == pytest.approx(expected, rel=1e-6)

    def test_m_uniform(self):
        """All M values should be equal (quasi-1D assumption)."""
        x = np.linspace(0, 5, 100)
        y = 1.0 + 0.3 * x
        y_ep, M_ep, theta_ep = synthesize_exit_plane(x, y)

        assert np.all(M_ep == M_ep[0])

    def test_at_intermediate_x(self):
        """x_exit < x_wall[-1] should give smaller y_exit and different M."""
        x = np.linspace(0, 10, 200)
        y = 1.0 + 0.2 * x
        y_full, M_full, _ = synthesize_exit_plane(x, y)
        y_mid, M_mid, _ = synthesize_exit_plane(x, y, x_exit=5.0)

        # Intermediate station should have smaller exit radius
        assert y_mid[-1] < y_full[-1]
        # And lower Mach
        assert M_mid[0] < M_full[0]


class TestConsistency:

    def test_mln_both_methods_agree(self):
        """moc_performance and quasi_1d should give similar Cf for MLN."""
        x, y, mesh = minimum_length_nozzle(2.0, n_chars=15)
        perf_moc = moc_performance(mesh)
        perf_q1d = quasi_1d_performance(x, y)

        # MLN has near-zero exit angle, so both should be close to Cf_ideal
        assert perf_q1d['Cf'] == pytest.approx(perf_moc['Cf'], rel=0.02)

    def test_tic_100pct_matches_mln(self):
        """TIC at 100% should match MLN Cf."""
        _, _, mesh_mln = minimum_length_nozzle(2.0, n_chars=15)
        perf_mln = moc_performance(mesh_mln)

        x_tic, y_tic, _ = truncated_ideal_contour(2.0, 1.0, n_chars=15)
        perf_tic = quasi_1d_performance(x_tic, y_tic)

        assert perf_tic['Cf'] == pytest.approx(perf_mln['Cf'], rel=0.01)

    def test_conical_integral_vs_analytical(self):
        """Exit plane integral for conical nozzle matches analytical lambda < 0.2%."""
        from nozzle.analysis import conical_performance

        alpha_deg = 15
        AR = 10
        perf_analytical = conical_performance(alpha_deg, AR)

        # Build equivalent exit plane
        M = mach_from_area_ratio(AR)
        y_exit = np.sqrt(AR)
        y = np.linspace(0, y_exit, 500)
        M_arr = np.full_like(y, M)
        theta = np.radians(alpha_deg) * y / y_exit

        Cf_integral = exit_plane_integral(y, M_arr, theta, 1.4)

        assert Cf_integral == pytest.approx(perf_analytical['Cf'], rel=0.002)

    def test_quasi1d_unchanged_for_tic(self):
        """TIC Cf values should be reasonable and in expected range."""
        x, y, _ = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        perf = quasi_1d_performance(x, y)

        # Cf should be between 0.9*Cf_ideal and Cf_ideal
        assert perf['Cf'] < perf['Cf_ideal']
        assert perf['Cf'] > 0.9 * perf['Cf_ideal']
        assert perf['lambda'] < 1.0
        assert perf['lambda'] > 0.9
