"""Tests for truncated ideal contour (TIC) nozzle."""

import numpy as np
import pytest
from nozzle.contours import truncated_ideal_contour, minimum_length_nozzle
from nozzle.analysis import (
    moc_performance, quasi_1d_performance,
    exit_plane_integral, synthesize_exit_plane,
)


class TestTIC:

    def test_tic_runs(self):
        """TIC should generate without error."""
        x, y, mesh = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        assert len(x) > 2
        assert len(y) > 2

    def test_tic_shorter_than_mln(self):
        """TIC should be shorter than full MLN."""
        x_mln, _, _ = minimum_length_nozzle(2.0, n_chars=15)
        x_tic, _, _ = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        assert x_tic[-1] < x_mln[-1]

    def test_tic_fraction_1_equals_mln(self):
        """TIC at 100% should equal full MLN."""
        x_mln, y_mln, _ = minimum_length_nozzle(2.0, n_chars=15)
        x_tic, y_tic, _ = truncated_ideal_contour(2.0, 1.0, n_chars=15)
        np.testing.assert_array_almost_equal(x_tic, x_mln)
        np.testing.assert_array_almost_equal(y_tic, y_mln)

    def test_tic_exit_radius_less_than_mln(self):
        """TIC exit radius should be less than full MLN exit radius."""
        _, y_mln, _ = minimum_length_nozzle(2.0, n_chars=15)
        _, y_tic, _ = truncated_ideal_contour(2.0, 0.6, n_chars=15)
        assert y_tic[-1] < y_mln[-1]

    def test_tic_starts_at_throat(self):
        """TIC wall should start near x=0, y=1."""
        x, y, _ = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        assert x[0] == pytest.approx(0.0, abs=0.1)
        assert y[0] == pytest.approx(1.0, abs=0.2)

    def test_tic_monotonic_fractions(self):
        """Longer TIC should have more length."""
        x60, _, _ = truncated_ideal_contour(2.0, 0.6, n_chars=15)
        x80, _, _ = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        assert x60[-1] < x80[-1]


class TestTICPerformance:

    def test_tic_cf_less_than_mln(self):
        """TIC Cf < MLN Cf due to non-zero exit wall angle (divergence loss)."""
        x_mln, y_mln, mesh_mln = minimum_length_nozzle(2.0, n_chars=15)
        perf_mln = moc_performance(mesh_mln)

        x_tic, y_tic, _ = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        perf_tic = quasi_1d_performance(x_tic, y_tic)

        assert perf_tic['Cf'] < perf_mln['Cf']

    def test_tic_cf_increases_with_fraction(self):
        """Longer TIC fraction → smaller exit angle → higher Cf."""
        x60, y60, _ = truncated_ideal_contour(2.0, 0.6, n_chars=15)
        x90, y90, _ = truncated_ideal_contour(2.0, 0.9, n_chars=15)

        perf60 = quasi_1d_performance(x60, y60)
        perf90 = quasi_1d_performance(x90, y90)

        assert perf90['Cf'] > perf60['Cf']

    def test_tic_100pct_cf_equals_mln(self):
        """At 100% truncation, TIC Cf ≈ MLN Cf (wall angle → 0)."""
        x_mln, y_mln, mesh_mln = minimum_length_nozzle(2.0, n_chars=15)
        perf_mln = moc_performance(mesh_mln)

        x_tic, y_tic, _ = truncated_ideal_contour(2.0, 1.0, n_chars=15)
        perf_tic = quasi_1d_performance(x_tic, y_tic)

        assert perf_tic['Cf'] == pytest.approx(perf_mln['Cf'], rel=0.01)

    def test_tic_quasi1d_matches_direct_integral(self):
        """quasi_1d_performance and synthesize+integral should agree exactly."""
        x, y, _ = truncated_ideal_contour(2.0, 0.8, n_chars=15)
        perf = quasi_1d_performance(x, y)

        y_ep, M_ep, theta_ep = synthesize_exit_plane(x, y)
        Cf_direct = exit_plane_integral(y_ep, M_ep, theta_ep)

        assert Cf_direct == pytest.approx(perf['Cf'], rel=1e-10)
