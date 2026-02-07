"""Tests for Rao parabolic approximation.

Validates angles against Sutton & Biblarz Table 3-4.
"""

import numpy as np
import pytest
from nozzle.contours import rao_parabolic_nozzle, rao_angles
from nozzle.analysis import rao_performance


class TestRaoAngles:
    """Validate angle lookup against Sutton Table 3-4."""

    def test_ar10_80pct_theta_n(self):
        """AR=10, 80% bell: θ_n ≈ 26.5° from Sutton Table 3-4."""
        theta_n, theta_e = rao_angles(10, 0.8)
        assert np.degrees(theta_n) == pytest.approx(26.5, abs=2.0)

    def test_ar10_80pct_theta_e(self):
        """AR=10, 80% bell: θ_e ≈ 10.0° from Sutton Table 3-4."""
        theta_n, theta_e = rao_angles(10, 0.8)
        assert np.degrees(theta_e) == pytest.approx(10.0, abs=2.0)

    def test_ar4_60pct(self):
        """AR=4, 60% bell: θ_n ≈ 28°, θ_e ≈ 17°."""
        theta_n, theta_e = rao_angles(4, 0.6)
        assert np.degrees(theta_n) == pytest.approx(28.0, abs=2.0)
        assert np.degrees(theta_e) == pytest.approx(17.0, abs=2.0)

    def test_theta_n_decreases_with_bell_fraction(self):
        """Higher bell fraction → smaller θ_n (more gradual expansion)."""
        _, _ = rao_angles(10, 0.6)
        tn_60, _ = rao_angles(10, 0.6)
        tn_80, _ = rao_angles(10, 0.8)
        tn_100, _ = rao_angles(10, 1.0)
        assert tn_60 > tn_80 > tn_100

    def test_theta_e_decreases_with_bell_fraction(self):
        """Higher bell fraction → smaller θ_e."""
        _, te_60 = rao_angles(10, 0.6)
        _, te_80 = rao_angles(10, 0.8)
        _, te_100 = rao_angles(10, 1.0)
        assert te_60 > te_80 > te_100

    def test_theta_n_greater_than_theta_e(self):
        """θ_n should always be greater than θ_e."""
        for ar in [4, 10, 25, 50]:
            for bf in [0.6, 0.8, 1.0]:
                theta_n, theta_e = rao_angles(ar, bf)
                assert theta_n > theta_e


class TestRaoContour:

    def test_rao_contour_runs(self):
        """Rao parabolic contour should generate without error."""
        x, y, theta_n, theta_e = rao_parabolic_nozzle(10, 0.8)
        assert len(x) > 10
        assert len(y) > 10

    def test_rao_starts_at_throat(self):
        """Contour should start near the throat."""
        x, y, theta_n, theta_e = rao_parabolic_nozzle(10, 0.8)
        assert x[0] == pytest.approx(0.0, abs=0.2)
        assert y[0] == pytest.approx(1.0, abs=0.2)

    def test_rao_exit_radius(self):
        """Exit y should match sqrt(AR)."""
        ar = 10
        x, y, theta_n, theta_e = rao_parabolic_nozzle(ar, 0.8)
        expected_y = np.sqrt(ar)
        assert y[-1] == pytest.approx(expected_y, rel=0.05)

    def test_rao_monotonic_y(self):
        """Wall y should generally increase."""
        x, y, theta_n, theta_e = rao_parabolic_nozzle(10, 0.8)
        # After the throat region, y should increase
        mid = len(y) // 3
        for i in range(mid, len(y) - 1):
            assert y[i + 1] >= y[i] - 1e-6

    def test_bell_fraction_affects_length(self):
        """Longer bell → longer nozzle."""
        x60, _, _, _ = rao_parabolic_nozzle(10, 0.6)
        x80, _, _, _ = rao_parabolic_nozzle(10, 0.8)
        x100, _, _, _ = rao_parabolic_nozzle(10, 1.0)
        assert x60[-1] < x80[-1] < x100[-1]


class TestRaoPerformance:

    def test_rao_cf_between_conical_and_ideal(self):
        """Rao Cf should be between conical and 1D ideal."""
        from nozzle.analysis import conical_performance
        from nozzle.gas import thrust_coefficient_ideal
        perf_con = conical_performance(15, 10)
        perf_rao = rao_performance(10, 0.8)
        Cf_ideal = thrust_coefficient_ideal(perf_con['M_exit'])
        assert perf_con['Cf'] < perf_rao['Cf'] < Cf_ideal

    def test_longer_bell_better_cf(self):
        """Longer bell fraction → higher Cf."""
        cf_60 = rao_performance(10, 0.6)['Cf']
        cf_80 = rao_performance(10, 0.8)['Cf']
        cf_100 = rao_performance(10, 1.0)['Cf']
        assert cf_60 < cf_80 < cf_100

    def test_rao_performance_keys(self):
        perf = rao_performance(10, 0.8)
        for key in ['lambda', 'Cf_ideal', 'Cf', 'M_exit',
                     'theta_n_deg', 'theta_e_deg']:
            assert key in perf

    def test_rao_lambda_reasonable(self):
        """Rao λ should be between 0.95 and 1.0."""
        perf = rao_performance(10, 0.8)
        assert 0.95 < perf['lambda'] < 1.0
