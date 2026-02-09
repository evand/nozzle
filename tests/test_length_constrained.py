"""Tests for length-constrained optimal nozzle.

Validates that bisection over M_design produces a truncated MLN contour
that matches the target area ratio at the given length budget.
"""

import numpy as np
import pytest

from nozzle.contours import (
    length_constrained_nozzle,
    minimum_length_nozzle,
    truncated_ideal_contour,
)
from nozzle.gas import area_mach_ratio, thrust_coefficient_ideal
from nozzle.analysis import quasi_1d_performance, moc_performance


class TestLengthConstrainedNozzle:
    """Core functionality tests."""

    def test_runs(self):
        """Generates without error."""
        x, y, mesh, info = length_constrained_nozzle(
            M_exit=2.0, target_length=2.0, n_chars=15)
        assert len(x) > 2
        assert len(y) == len(x)
        assert info['iterations'] >= 0

    def test_length_matches_target(self):
        """Contour length matches target_length within 1%."""
        target = 2.0
        x, y, _, info = length_constrained_nozzle(
            M_exit=2.0, target_length=target, n_chars=15)
        L = x[-1] - x[0]
        assert abs(L - target) / target < 0.01

    def test_area_ratio_matches_exit_mach(self):
        """Exit area ratio matches area_mach_ratio(M_exit) within 1%."""
        M_exit = 2.0
        gamma = 1.4
        target_ar = area_mach_ratio(M_exit, gamma)
        x, y, _, info = length_constrained_nozzle(
            M_exit=M_exit, target_length=2.0, n_chars=15, gamma=gamma)
        ar = y[-1] ** 2
        assert abs(ar - target_ar) / target_ar < 0.01

    def test_m_design_gt_m_exit(self):
        """M_design >= M_exit (over-designed then truncated)."""
        x, y, _, info = length_constrained_nozzle(
            M_exit=2.0, target_length=2.0, n_chars=15)
        assert info['M_design'] >= 2.0

    def test_full_mln_when_fits(self):
        """If target_length exceeds full MLN, return it unchanged."""
        M_exit = 2.0
        gamma = 1.4
        x_full, y_full, _ = minimum_length_nozzle(M_exit, 15, gamma)
        L_full = x_full[-1] - x_full[0]

        # Give a generous target length
        x, y, _, info = length_constrained_nozzle(
            M_exit=M_exit, target_length=L_full * 1.5, n_chars=15,
            gamma=gamma)
        assert info['truncation_fraction'] == 1.0
        assert info['M_design'] == M_exit
        assert info['iterations'] == 0
        np.testing.assert_array_equal(x, x_full)
        np.testing.assert_array_equal(y, y_full)

    def test_starts_at_throat(self):
        """Contour starts near x=0, y=1."""
        x, y, _, _ = length_constrained_nozzle(
            M_exit=2.0, target_length=2.0, n_chars=15)
        assert abs(x[0]) < 0.1
        assert abs(y[0] - 1.0) < 0.1

    def test_monotonic_wall(self):
        """Wall y-coordinates are monotonically increasing."""
        x, y, _, _ = length_constrained_nozzle(
            M_exit=2.0, target_length=2.0, n_chars=15)
        dy = np.diff(y)
        assert np.all(dy >= -1e-10), "Wall y should be monotonically increasing"

    def test_cf_between_tic_and_mln(self):
        """Performance bounded: TIC(same length) <= LC <= full MLN."""
        M_exit = 2.0
        gamma = 1.4
        n_chars = 15

        # Full MLN
        x_mln, y_mln, mesh_mln = minimum_length_nozzle(M_exit, n_chars, gamma)
        perf_mln = moc_performance(mesh_mln, gamma)
        L_mln = x_mln[-1] - x_mln[0]

        # Length-constrained at 60% of MLN length
        target_length = 0.6 * L_mln
        x_lc, y_lc, _, _ = length_constrained_nozzle(
            M_exit, target_length, n_chars, gamma)
        perf_lc = quasi_1d_performance(x_lc, y_lc, gamma)

        # TIC at same truncation fraction
        x_tic, y_tic, _ = truncated_ideal_contour(
            M_exit, 0.6, n_chars, gamma)
        perf_tic = quasi_1d_performance(x_tic, y_tic, gamma)

        # LC should be >= TIC (it's optimized for this length)
        # Allow small tolerance for quasi-1D approximation differences
        assert perf_lc['Cf'] >= perf_tic['Cf'] - 0.005, (
            f"LC Cf={perf_lc['Cf']:.4f} should be >= TIC Cf={perf_tic['Cf']:.4f}")

        # LC should be <= full MLN
        assert perf_lc['Cf'] <= perf_mln['Cf'] + 0.001

    def test_higher_mach(self):
        """Works at M=3.0."""
        x, y, _, info = length_constrained_nozzle(
            M_exit=3.0, target_length=3.0, n_chars=15)
        target_ar = area_mach_ratio(3.0, 1.4)
        ar = y[-1] ** 2
        assert abs(ar - target_ar) / target_ar < 0.01
        assert info['M_design'] >= 3.0


class TestLengthConstrainedPerformance:
    """Performance trend tests."""

    def test_longer_gives_higher_cf(self):
        """More length budget gives better (higher) Cf."""
        M_exit = 2.0
        gamma = 1.4
        n_chars = 15

        _, y_short, _, _ = length_constrained_nozzle(
            M_exit, target_length=1.5, n_chars=n_chars, gamma=gamma)
        x_short_wall, y_short_wall, _, _ = length_constrained_nozzle(
            M_exit, target_length=1.5, n_chars=n_chars, gamma=gamma)
        perf_short = quasi_1d_performance(x_short_wall, y_short_wall, gamma)

        x_long_wall, y_long_wall, _, _ = length_constrained_nozzle(
            M_exit, target_length=2.5, n_chars=n_chars, gamma=gamma)
        perf_long = quasi_1d_performance(x_long_wall, y_long_wall, gamma)

        assert perf_long['Cf'] >= perf_short['Cf'] - 0.001

    def test_cf_less_than_mln(self):
        """Truncated nozzle always has Cf <= full MLN."""
        M_exit = 2.0
        gamma = 1.4
        n_chars = 15

        _, _, mesh_mln = minimum_length_nozzle(M_exit, n_chars, gamma)
        perf_mln = moc_performance(mesh_mln, gamma)

        x_lc, y_lc, _, _ = length_constrained_nozzle(
            M_exit, target_length=1.5, n_chars=n_chars, gamma=gamma)
        perf_lc = quasi_1d_performance(x_lc, y_lc, gamma)

        assert perf_lc['Cf'] <= perf_mln['Cf'] + 0.001

    def test_cf_reasonable(self):
        """Cf is in a physically reasonable range."""
        M_exit = 2.0
        gamma = 1.4
        Cf_ideal = thrust_coefficient_ideal(M_exit, gamma)

        x, y, _, _ = length_constrained_nozzle(
            M_exit, target_length=2.0, n_chars=15, gamma=gamma)
        perf = quasi_1d_performance(x, y, gamma)

        # Should be at least 90% of ideal and no more than ideal
        assert perf['Cf'] >= 0.90 * Cf_ideal
        assert perf['Cf'] <= Cf_ideal * 1.01
