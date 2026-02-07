"""Tests for conical nozzle contour and analytical performance.

Validates against Sutton & Biblarz and basic geometry.
"""

import numpy as np
import pytest
from nozzle.contours import conical_nozzle, conical_divergence_loss
from nozzle.analysis import conical_performance


class TestConicalGeometry:

    def test_throat_at_origin(self):
        x, y = conical_nozzle(15, area_ratio=10)
        assert x[0] == pytest.approx(0.0)
        assert y[0] == pytest.approx(1.0)

    def test_exit_radius(self):
        """y_exit = sqrt(Ae/A*)."""
        ar = 10
        x, y = conical_nozzle(15, area_ratio=ar)
        assert y[-1] == pytest.approx(np.sqrt(ar), rel=1e-10)

    def test_length_15deg(self):
        """L = (y_exit - 1) / tan(15°)."""
        ar = 10
        x, y = conical_nozzle(15, area_ratio=ar)
        expected_length = (np.sqrt(ar) - 1.0) / np.tan(np.radians(15))
        assert x[-1] == pytest.approx(expected_length, rel=1e-10)

    def test_straight_line(self):
        """All points should lie on a straight line."""
        x, y = conical_nozzle(20, area_ratio=5, n_points=100)
        # slope = tan(alpha)
        slope = np.tan(np.radians(20))
        y_expected = 1.0 + slope * x
        np.testing.assert_allclose(y, y_expected, atol=1e-12)


class TestDivergenceLoss:

    def test_lambda_15deg(self):
        """λ = (1 + cos 15°)/2 = 0.9830."""
        lam = conical_divergence_loss(15)
        assert lam == pytest.approx(0.9830, abs=0.0001)

    def test_lambda_0deg(self):
        """Zero angle → perfect alignment → λ=1."""
        assert conical_divergence_loss(0) == pytest.approx(1.0, abs=1e-12)

    def test_lambda_30deg(self):
        """λ = (1 + cos 30°)/2 ≈ 0.9330."""
        lam = conical_divergence_loss(30)
        assert lam == pytest.approx((1 + np.cos(np.radians(30))) / 2, abs=1e-10)

    def test_lambda_monotonic(self):
        """λ decreases with increasing half-angle."""
        angles = [5, 10, 15, 20, 25, 30]
        lambdas = [conical_divergence_loss(a) for a in angles]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] > lambdas[i + 1]


class TestConicalPerformance:

    def test_performance_keys(self):
        result = conical_performance(15, 10)
        assert 'lambda' in result
        assert 'Cf_ideal' in result
        assert 'Cf' in result
        assert 'M_exit' in result

    def test_cf_less_than_ideal(self):
        """Conical Cf should be less than ideal (divergence loss)."""
        result = conical_performance(15, 10)
        assert result['Cf'] < result['Cf_ideal']

    def test_cf_equals_lambda_times_ideal(self):
        result = conical_performance(15, 10)
        assert result['Cf'] == pytest.approx(
            result['lambda'] * result['Cf_ideal'], rel=1e-12
        )

    def test_larger_angle_worse_performance(self):
        """Larger half-angle → worse Cf due to divergence."""
        cf_15 = conical_performance(15, 10)['Cf']
        cf_30 = conical_performance(30, 10)['Cf']
        assert cf_15 > cf_30
