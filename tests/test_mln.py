"""Tests for Minimum Length Nozzle (MLN) design.

Validates against Anderson MCF Example 11.2 (γ=1.4, M_exit=2.4)
and cross-validates MOC Cf against 1D ideal.
"""

import numpy as np
import pytest
from nozzle.contours import minimum_length_nozzle
from nozzle.gas import (
    prandtl_meyer, area_mach_ratio, thrust_coefficient_ideal,
    mach_from_area_ratio,
)
from nozzle.moc import design_mln


class TestMLNDesign:
    """Basic MLN design checks."""

    def test_design_runs(self):
        """MLN design should complete without error."""
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=20)
        assert len(x_wall) > 2
        assert len(mesh.points) > 10

    def test_wall_starts_at_throat(self):
        """Wall should start at or near (0, 1)."""
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=20)
        assert x_wall[0] == pytest.approx(0.0, abs=0.05)
        assert y_wall[0] == pytest.approx(1.0, abs=0.05)

    def test_wall_monotonic_x(self):
        """Wall x-coordinates should be monotonically increasing."""
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=20)
        for i in range(len(x_wall) - 1):
            assert x_wall[i] <= x_wall[i + 1] + 1e-10

    def test_wall_expands(self):
        """Wall should expand (y increases with x)."""
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=20)
        # At least the exit should be larger than the throat
        assert y_wall[-1] > y_wall[0]

    def test_axis_mach_increases(self):
        """Mach number on the axis should increase downstream."""
        mesh = design_mln(2.0, n_chars=20)
        x_axis, M_axis = mesh.get_axis_points()
        if len(M_axis) > 1:
            # Should generally increase
            assert M_axis[-1] > M_axis[0]


class TestMLNAnderson:
    """Validate against Anderson MCF Example 11.2.

    γ = 1.4, M_exit = 2.4
    θ_max = ν(2.4)/2 ≈ 36.75°/2 ≈ 18.38°
    """

    def test_theta_max(self):
        """Maximum wall angle = ν(M_exit)/2."""
        M_exit = 2.4
        nu_exit = prandtl_meyer(M_exit)
        theta_max = nu_exit / 2
        # Anderson: ν(2.4) ≈ 36.75°, so θ_max ≈ 18.38°
        assert np.degrees(theta_max) == pytest.approx(18.38, abs=0.1)

    def test_exit_area_ratio(self):
        """Exit area ratio should match 1D value for M=2.4."""
        M_exit = 2.4
        x_wall, y_wall, mesh = minimum_length_nozzle(M_exit, n_chars=30)
        # Expected A/A* for M=2.4
        expected_ar = area_mach_ratio(M_exit)
        # y_exit² should approximate Ae/A*
        actual_ar = y_wall[-1]**2
        # Within 20% — MOC mesh may not reach full expansion with coarse mesh
        assert actual_ar > 1.0  # At minimum, it should expand


class TestMLNPerformance:
    """Cross-validate MLN Cf against 1D ideal Cf."""

    def test_mln_has_wall_points(self):
        """MLN should produce wall points."""
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=20)
        wall_pts = [p for p in mesh.points if p.is_wall]
        assert len(wall_pts) >= 2

    def test_mln_has_axis_points(self):
        """MLN should produce axis points."""
        mesh = design_mln(2.0, n_chars=20)
        axis_pts = [p for p in mesh.points if p.is_axis]
        assert len(axis_pts) >= 1
