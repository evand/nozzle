"""Tests for sivells_nozzle() contour wrapper.

Validates the integration of sivells_axial + sivells_perfc into
the standard contour interface (x, y in r*-normalized coordinates).
"""

import numpy as np
import pytest
from nozzle.contours import sivells_nozzle


class TestSivellsNozzle:
    """Test sivells_nozzle() wrapper function."""

    def test_basic_call(self):
        """Returns two arrays of equal length."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4,
                              rc=6.0, inflection_angle_deg=8.67)
        assert len(x) == len(y)
        assert len(x) > 10

    def test_x_monotonic(self):
        """Wall x-coordinates increase monotonically (validated params)."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4,
                              rc=6.0, inflection_angle_deg=8.67)
        assert np.all(np.diff(x) > 0), "x must be strictly increasing"

    def test_y_monotonic(self):
        """Wall y-coordinates increase monotonically (validated params)."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4,
                              rc=6.0, inflection_angle_deg=8.67)
        assert np.all(np.diff(y) >= 0), "y must be non-decreasing"

    def test_throat_at_origin(self):
        """First point (throat) should be near x=0."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4,
                              rc=6.0, inflection_angle_deg=8.67)
        assert abs(x[0]) < 0.01, f"Throat x should be ~0, got {x[0]}"

    def test_throat_y_near_one(self):
        """First point (throat) should have y near 1.0 (r*-normalized)."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4,
                              rc=6.0, inflection_angle_deg=8.67)
        assert abs(y[0] - 1.0) < 0.01, f"Throat y should be ~1.0, got {y[0]}"

    def test_y_above_one(self):
        """All wall points should have y >= 1.0 (above throat)."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4,
                              rc=6.0, inflection_angle_deg=8.67)
        assert np.all(y >= 0.999), f"y min = {y.min()}, expected >= 1.0"

    def test_mach4_reference_params(self):
        """With Mach 4 reference params, contour should have 41 points."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13, ie=0
        )
        assert len(x) == 41

    def test_mach4_reference_endpoint(self):
        """Inflection point should be at known position for Mach 4 case."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13, ie=0
        )
        # Last point is inflection; y must be expanding
        assert y[-1] > y[0], "Inflection point should be above throat"
        assert y[-1] / y[0] > 1.1, "Exit should be >10% wider than throat"

    def test_mach4_reference_coord_values(self):
        """Spot-check normalized coordinates against CONTUR Mach 4 output."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13, ie=0
        )
        # Throat at x=0, y=1; inflection at x>0, y>1
        assert abs(x[0]) < 1e-5, f"Throat x should be 0, got {x[0]}"
        # Inflection x ≈ (1.3564 - 0.8966) / 0.1514 ≈ 3.04
        assert x[-1] > 2.5
        assert x[-1] < 4.0
        # Inflection y ≈ 0.2068 / 0.1514 ≈ 1.366
        assert y[-1] > 1.3
        assert y[-1] < 1.5

    def test_auto_defaults_m3(self):
        """Auto-derived defaults work for M=3.0."""
        x, y = sivells_nozzle(M_exit=3.0, gamma=1.4)
        assert len(x) > 10
        assert abs(y[0] - 1.0) < 0.01

    def test_auto_defaults_m4(self):
        """Auto-derived defaults work for M=4.0."""
        x, y = sivells_nozzle(M_exit=4.0, gamma=1.4)
        assert len(x) > 10
        assert abs(y[0] - 1.0) < 0.01

    def test_different_rc(self):
        """Different rc values produce different contours."""
        x1, y1 = sivells_nozzle(M_exit=3.0, gamma=1.4, rc=1.5)
        x2, y2 = sivells_nozzle(M_exit=3.0, gamma=1.4, rc=4.0)
        assert not np.allclose(x1, x2)
