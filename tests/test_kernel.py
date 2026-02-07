"""Tests for the transonic kernel (Hall 1962 initial data line).

Validates physical constraints on the initial data line.
"""

import numpy as np
import pytest
from nozzle.kernel import hall_initial_line


class TestHallInitialLine:

    def test_all_supersonic(self):
        """All Mach numbers on initial line must be > 1."""
        x, y, M, theta = hall_initial_line(20, gamma=1.4)
        assert np.all(M > 1.0)

    def test_axis_theta_zero(self):
        """Flow angle at axis (y=0) should be zero."""
        x, y, M, theta = hall_initial_line(20, gamma=1.4)
        assert theta[0] == pytest.approx(0.0, abs=1e-12)

    def test_axis_is_first_point(self):
        """First point should be on axis (y=0)."""
        x, y, M, theta = hall_initial_line(20)
        assert y[0] == pytest.approx(0.0, abs=1e-12)

    def test_wall_is_last_point(self):
        """Last point should be near y=1 (wall at throat)."""
        x, y, M, theta = hall_initial_line(20, x_start=0.05)
        # At x=0.05, wall is at y ≈ 1 + 0.05²/(2·1.5) ≈ 1.00083
        assert y[-1] == pytest.approx(1.0 + 0.05**2 / (2 * 1.5), rel=1e-6)

    def test_mach_monotonic_axis_to_wall(self):
        """M should be highest on axis, decreasing toward wall.

        The axis accelerates first in a converging-diverging nozzle.
        """
        x, y, M, theta = hall_initial_line(50, gamma=1.4)
        # M should decrease monotonically from axis to wall
        for i in range(len(M) - 1):
            assert M[i] >= M[i + 1] - 1e-12

    def test_theta_monotonic(self):
        """θ should increase from 0 at axis to positive at wall."""
        x, y, M, theta = hall_initial_line(50, gamma=1.4)
        assert theta[0] == pytest.approx(0.0, abs=1e-12)
        assert theta[-1] > 0
        for i in range(len(theta) - 1):
            assert theta[i] <= theta[i + 1] + 1e-12

    def test_constant_x(self):
        """All points should be at the same x position."""
        x, y, M, theta = hall_initial_line(20, x_start=0.1)
        np.testing.assert_allclose(x, 0.1, atol=1e-14)

    def test_n_points(self):
        for n in [10, 50, 100]:
            x, y, M, theta = hall_initial_line(n)
            assert len(x) == n
            assert len(y) == n
            assert len(M) == n
            assert len(theta) == n

    def test_larger_R_wall_slower_accel(self):
        """Larger throat radius of curvature → slower acceleration → lower M."""
        _, _, M_small, _ = hall_initial_line(20, R_wall=1.0, x_start=0.05)
        _, _, M_large, _ = hall_initial_line(20, R_wall=3.0, x_start=0.05)
        # Smaller R_wall → more acceleration → higher M on axis
        assert M_small[0] > M_large[0]

    @pytest.mark.parametrize("gamma", [1.2, 1.3, 1.4, 1.667])
    def test_different_gamma(self, gamma):
        """Should work for various gamma values."""
        x, y, M, theta = hall_initial_line(20, gamma=gamma)
        assert np.all(M > 1.0)
        assert theta[0] == pytest.approx(0.0, abs=1e-12)
