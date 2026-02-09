"""Tests for convergent section geometry."""

import numpy as np
import pytest
from nozzle.contours import convergent_section, conical_nozzle


class TestConvergentSection:
    """Test convergent_section() geometry and constraints."""

    def test_throat_at_origin(self):
        """Last point should be at (0, 1) = throat."""
        x, y = convergent_section()
        assert abs(x[-1] - 0.0) < 1e-10
        assert abs(y[-1] - 1.0) < 1e-10

    def test_all_x_nonpositive(self):
        """All x values should be <= 0 (upstream of throat)."""
        x, y = convergent_section()
        assert np.all(x <= 1e-10)

    def test_y_decreasing_toward_throat(self):
        """y should monotonically decrease from chamber to throat."""
        x, y = convergent_section()
        # Allow small tolerance for numerical noise at arc junctions
        dy = np.diff(y)
        assert np.all(dy <= 1e-10), f"Non-monotonic y: max increase = {dy.max():.6e}"

    def test_chamber_radius(self):
        """First points should be at chamber radius = sqrt(contraction_ratio)."""
        cr = 4.0
        x, y = convergent_section(contraction_ratio=cr)
        expected = np.sqrt(cr)
        assert abs(y[0] - expected) < 1e-6

    def test_no_sudden_jumps(self):
        """Adjacent points should not have large jumps (smoothness check)."""
        x, y = convergent_section(n_points=200)
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        # Max step should be small relative to total extent
        x_extent = abs(x[0] - x[-1])
        y_extent = abs(y[0] - y[-1])
        assert np.max(dx) < 0.1 * x_extent
        assert np.max(dy) < 0.1 * y_extent

    def test_prepend_to_conical(self):
        """Convergent + conical should give a continuous full nozzle profile."""
        x_conv, y_conv = convergent_section(contraction_ratio=3.0)
        x_div, y_div = conical_nozzle(15, area_ratio=4.0)

        # Prepend (skip last convergent point = first divergent point)
        x_full = np.concatenate([x_conv[:-1], x_div])
        y_full = np.concatenate([y_conv[:-1], y_div])

        # Check continuity at junction
        assert abs(x_conv[-1] - x_div[0]) < 1e-10
        assert abs(y_conv[-1] - y_div[0]) < 1e-10

        # Full profile: x goes from negative to positive
        assert x_full[0] < 0
        assert x_full[-1] > 0

    def test_contraction_ratio_2(self):
        """Should work for contraction ratio 2."""
        x, y = convergent_section(contraction_ratio=2.0)
        assert abs(y[0] - np.sqrt(2.0)) < 1e-6
        assert abs(y[-1] - 1.0) < 1e-10

    def test_contraction_ratio_8(self):
        """Should work for contraction ratio 8."""
        x, y = convergent_section(contraction_ratio=8.0)
        assert abs(y[0] - np.sqrt(8.0)) < 1e-6
        assert abs(y[-1] - 1.0) < 1e-10

    def test_angle_20_deg(self):
        """Should work for 20° half-angle."""
        x, y = convergent_section(convergent_half_angle_deg=20.0)
        assert abs(y[-1] - 1.0) < 1e-10
        assert np.all(x <= 1e-10)

    def test_angle_45_deg(self):
        """Should work for 45° half-angle."""
        x, y = convergent_section(convergent_half_angle_deg=45.0)
        assert abs(y[-1] - 1.0) < 1e-10
        assert np.all(x <= 1e-10)

    def test_different_arc_radii(self):
        """Should work with different upstream/downstream arc radii."""
        x, y = convergent_section(rc_upstream=2.0, rc_downstream=0.5)
        assert abs(y[-1] - 1.0) < 1e-10
        dy = np.diff(y)
        assert np.all(dy <= 1e-10)


class TestConvergentConfig:
    """Test convergent section config integration."""

    def test_config_passthrough(self):
        """Convergent sub-dict should appear in spec."""
        from nozzle.config import build_nozzle_spec
        cfg = {
            'type': 'conical',
            'M_exit': 2.0,
            'convergent': {
                'contraction_ratio': 4.0,
                'half_angle_deg': 25,
            },
        }
        spec = build_nozzle_spec(cfg)
        assert 'convergent' in spec
        assert spec['convergent']['contraction_ratio'] == 4.0
        assert spec['convergent']['convergent_half_angle_deg'] == 25.0

    def test_config_defaults(self):
        """Convergent with empty dict should use defaults."""
        from nozzle.config import build_nozzle_spec
        cfg = {
            'type': 'conical',
            'M_exit': 2.0,
            'convergent': {},
        }
        spec = build_nozzle_spec(cfg)
        assert spec['convergent']['contraction_ratio'] == 3.0
        assert spec['convergent']['convergent_half_angle_deg'] == 30.0
        assert spec['convergent']['rc_upstream'] == 1.5
        assert spec['convergent']['rc_downstream'] == 0.382
