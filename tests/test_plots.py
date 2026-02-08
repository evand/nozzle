"""Tests for nozzle.plots — tolerance band visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from nozzle.plots import plot_tolerance_band


def _make_contour(y_exit=2.0, n=100):
    """Simple linear contour from throat (x=0, y=1) to exit."""
    x = np.linspace(0, 5, n)
    y = 1 + (y_exit - 1) * x / x[-1]
    return x, y


class TestToleranceBand:

    def test_tolerance_band_runs(self):
        """Smoke test: two contours produce (fig, axes) with 3 panels."""
        x_a, y_a = _make_contour(y_exit=2.0)
        x_b, y_b = _make_contour(y_exit=2.1)
        fig, axes = plot_tolerance_band(x_a, y_a, "nom",
                                        x_b, y_b, "act")
        assert len(axes) == 3
        assert fig is not None
        plt.close(fig)

    def test_tolerance_band_with_tol(self):
        """With tol parameter: still runs, 3 panels."""
        x_a, y_a = _make_contour(y_exit=2.0)
        x_b, y_b = _make_contour(y_exit=2.05)
        fig, axes = plot_tolerance_band(x_a, y_a, "nom",
                                        x_b, y_b, "act",
                                        tol=0.01)
        assert len(axes) == 3
        plt.close(fig)

    def test_tolerance_band_identical_contours(self):
        """Identical contours: dy=0 everywhere, no crash."""
        x, y = _make_contour()
        fig, axes = plot_tolerance_band(x, y, "A", x, y, "B")
        assert len(axes) == 3
        plt.close(fig)

    def test_tolerance_band_area_deviation_sign(self):
        """Wider actual contour gives positive area deviation."""
        x_nom, y_nom = _make_contour(y_exit=2.0)
        x_act, y_act = _make_contour(y_exit=2.2)  # wider

        # Compute expected area deviation at exit
        # (y_act/y_nom)^2 - 1 should be positive
        area_dev_exit = (2.2 / 2.0) ** 2 - 1
        assert area_dev_exit > 0

        fig, axes = plot_tolerance_band(x_nom, y_nom, "nom",
                                        x_act, y_act, "act")
        # Check panel 3 (area deviation) has data — y-data should include
        # positive values
        lines = axes[2].get_lines()
        ydata = lines[0].get_ydata()  # first line is the area_pct curve
        assert np.max(ydata) > 0, "Wider actual should give positive area deviation"
        plt.close(fig)
