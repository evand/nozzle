"""Tests for sivells_nozzle() contour wrapper and downstream contour.

Validates the integration of sivells_axial + sivells_perfc into
the standard contour interface (x, y in r*-normalized coordinates).

Downstream validation uses CONTUR Mach 4 planar reference output from
docs/references/external_codes/contur/downstream_run/mach4_planar_full_output.txt
"""

import numpy as np
import pytest
from nozzle.contours import sivells_nozzle
from nozzle.sivells import (
    sivells_axial,
    sivells_perfc,
    sivells_axial_downstream,
    sivells_perfc_downstream,
)


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


# =========================================================================
# Downstream axial distribution tests
# =========================================================================

class TestAxialDownstream:
    """Test sivells_axial_downstream() against CONTUR Mach 4 planar output.

    CONTUR reference (planar ie=0, bmach=3.2, cmach=4.0):
        C1=3.2000000, C2=2.71467351, C3=-3.3440205,
        C4=1.7440205, C5=-0.31467351, C6=0
        XB=5.1209575, XBC=13.1696977, XC=18.2906552, XD=24.5724861
        BMP=0.20613028, BMPP=-3.8560914E-02
        Axis M(k=1)=3.200000, M(k=49)=4.000000
    """

    @pytest.fixture
    def mach4_upstream(self):
        """Run upstream axial once."""
        return sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.2, cmach=4.0,
            ie=0, n_char=41, n_axis=21, nx=13
        )

    @pytest.fixture
    def downstream_result(self, mach4_upstream):
        """Run downstream axial."""
        return sivells_axial_downstream(
            mach4_upstream, gamma=1.4, ie=0,
            ip=10, md=41, nd=49, nf=-61
        )

    def test_polynomial_c1(self, downstream_result):
        """C1 = bmach = 3.2000000."""
        assert abs(downstream_result['c'][0] - 3.2000000) < 1e-6

    def test_polynomial_c2(self, downstream_result):
        """C2 = 2.71467351."""
        assert abs(downstream_result['c'][1] - 2.71467351) < 1e-5

    def test_polynomial_c3(self, downstream_result):
        """C3 = -3.3440205."""
        assert abs(downstream_result['c'][2] - (-3.3440205)) < 1e-4

    def test_polynomial_c4(self, downstream_result):
        """C4 = 1.7440205."""
        assert abs(downstream_result['c'][3] - 1.7440205) < 1e-4

    def test_polynomial_c5(self, downstream_result):
        """C5 = -0.31467351."""
        assert abs(downstream_result['c'][4] - (-0.31467351)) < 1e-4

    def test_polynomial_c6(self, downstream_result):
        """C6 = 0."""
        assert abs(downstream_result['c'][5]) < 1e-10

    def test_xb(self, downstream_result):
        """XB = 5.1209575."""
        assert abs(downstream_result['xb'] - 5.1209575) < 1e-5

    def test_xbc(self, downstream_result):
        """XBC = 13.1696977."""
        assert abs(downstream_result['xbc'] - 13.1696977) < 1e-4

    def test_xc(self, downstream_result):
        """XC = 18.2906552."""
        assert abs(downstream_result['xc'] - 18.2906552) < 1e-4

    def test_xd(self, downstream_result):
        """XD = 24.5724861."""
        assert abs(downstream_result['xd'] - 24.5724861) < 1e-3

    def test_axis_first_mach(self, downstream_result):
        """First axis point M = 3.200000 (bmach)."""
        assert abs(downstream_result['axis'][2, 0] - 3.200000) < 1e-6

    def test_axis_last_mach(self, downstream_result):
        """Last axis point M = 4.0000000 (cmach)."""
        assert abs(downstream_result['axis'][2, -1] - 4.0000000) < 1e-6

    def test_axis_mach_monotonic(self, downstream_result):
        """Mach should increase monotonically along downstream axis."""
        mach = downstream_result['axis'][2, :]
        assert np.all(np.diff(mach) > 0), "Mach must be strictly increasing"

    def test_axis_x_monotonic(self, downstream_result):
        """X should increase monotonically."""
        x = downstream_result['axis'][0, :]
        assert np.all(np.diff(x) > 0), "x must be strictly increasing"

    def test_axis_psi_positive(self, downstream_result):
        """Prandtl-Meyer angle should be positive at all points."""
        psi = downstream_result['axis'][3, :]
        assert np.all(psi > 0), "psi must be positive"

    def test_np_pts(self, downstream_result):
        """Number of exit characteristic points = 61 for Mach 4 case."""
        assert downstream_result['np_pts'] == 61

    def test_n_axis_points(self, downstream_result):
        """Should have 49 axis points."""
        assert downstream_result['n'] == 49
        assert downstream_result['axis'].shape[1] == 49

    def test_bmp(self, downstream_result):
        """BMP = 0.20613028."""
        assert abs(downstream_result['bmp'] - 0.20613028) < 1e-7

    def test_bmpp(self, downstream_result):
        """BMPP = -3.8560914E-02."""
        assert abs(downstream_result['bmpp'] - (-3.8560914e-02)) < 1e-7


# =========================================================================
# Downstream perfc tests
# =========================================================================

class TestPerfcDownstream:
    """Test sivells_perfc_downstream() against CONTUR Mach 4 planar output.

    Reference wall data (planar ie=0, 109 points, CONTUR indices 41-149):
        First: x=3.3152830, y=0.5055332, M=2.7550181, θ=8.67°
        Exit:  x=24.5724861, y=1.6219617, M=4.0000000, θ=0.0°
    Generated from contur/src/input.txt (bmach=3.2, cmach=4.0, jd=-1 ie=0).
    """

    # CONTUR downstream wall reference: first and last 5 points
    REF_WALL_FIRST_X = np.array([
        3.3152830, 3.4647390, 3.6132901, 3.7612318, 3.9088485])
    REF_WALL_FIRST_Y = np.array([
        0.5055332, 0.5283244, 0.5509710, 0.5735086, 0.5959680])
    REF_WALL_FIRST_M = np.array([
        2.7550181, 2.8013721, 2.8453402, 2.8871211, 2.9268919])
    REF_WALL_FIRST_ANGLE = np.array([
        8.6700000, 8.6692303, 8.6649595, 8.6564571, 8.6429386])

    REF_WALL_LAST_X = np.array([
        24.3630918, 24.5724861])
    REF_WALL_LAST_Y = np.array([
        1.6219617, 1.6219617])
    REF_WALL_LAST_M = np.array([
        3.9999983, 4.0000000])

    @pytest.fixture(scope='class')
    def full_downstream(self):
        """Run upstream + downstream to get wall contour."""
        axial = sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.2, cmach=4.0,
            ie=0, n_char=41, n_axis=21, nx=13
        )
        ds_axial = sivells_axial_downstream(
            axial, gamma=1.4, ie=0,
            ip=10, md=41, nd=49, nf=-61
        )
        ds_perfc = sivells_perfc_downstream(
            axial, ds_axial, gamma=1.4, ie=0
        )
        return ds_perfc

    def test_n_wall_points(self, full_downstream):
        """109 wall points from inflection to exit."""
        assert full_downstream['n_wall'] == 109

    def test_mass_flow(self, full_downstream):
        """Mass flow from first characteristic should be ~1.0."""
        assert abs(full_downstream['mass'] - 1.0) < 1e-5

    def test_first_wall_x(self, full_downstream):
        """First 5 wall x-coordinates match CONTUR."""
        wax = full_downstream['wax']
        np.testing.assert_allclose(
            wax[:5], self.REF_WALL_FIRST_X, atol=1e-4,
            err_msg="First wall x-coordinates disagree"
        )

    def test_first_wall_y(self, full_downstream):
        """First 5 wall y-coordinates match CONTUR."""
        way = full_downstream['way']
        np.testing.assert_allclose(
            way[:5], self.REF_WALL_FIRST_Y, atol=1e-3,
            err_msg="First wall y-coordinates disagree"
        )

    def test_first_wall_mach(self, full_downstream):
        """First 5 wall Mach numbers match CONTUR."""
        wmn = full_downstream['wmn']
        np.testing.assert_allclose(
            wmn[:5], self.REF_WALL_FIRST_M, atol=1e-3,
            err_msg="First wall Mach numbers disagree"
        )

    def test_first_wall_angle(self, full_downstream):
        """First 5 wall flow angles match CONTUR."""
        wan = full_downstream['wan']
        np.testing.assert_allclose(
            wan[:5], self.REF_WALL_FIRST_ANGLE, atol=0.05,
            err_msg="First wall flow angles disagree"
        )

    def test_last_wall_x(self, full_downstream):
        """Last 2 wall x-coordinates match CONTUR."""
        wax = full_downstream['wax']
        np.testing.assert_allclose(
            wax[-2:], self.REF_WALL_LAST_X, atol=1e-3,
            err_msg="Last wall x-coordinates disagree"
        )

    def test_last_wall_y(self, full_downstream):
        """Last 2 wall y-coordinates match CONTUR."""
        way = full_downstream['way']
        np.testing.assert_allclose(
            way[-2:], self.REF_WALL_LAST_Y, atol=1e-2,
            err_msg="Last wall y-coordinates disagree"
        )

    def test_last_wall_mach(self, full_downstream):
        """Last wall Mach = 4.0 (exit)."""
        wmn = full_downstream['wmn']
        assert abs(wmn[-1] - 4.0) < 1e-3

    def test_exit_angle_zero(self, full_downstream):
        """Exit flow angle should be 0 (parallel flow)."""
        wan = full_downstream['wan']
        assert abs(wan[-1]) < 0.01

    def test_wall_x_monotonic(self, full_downstream):
        """Wall x should increase monotonically."""
        wax = full_downstream['wax']
        assert np.all(np.diff(wax) > 0), "x must be strictly increasing"

    def test_wall_y_monotonic(self, full_downstream):
        """Wall y should increase monotonically."""
        way = full_downstream['way']
        assert np.all(np.diff(way) >= -1e-6), "y must be non-decreasing"

    def test_wall_mach_monotonic(self, full_downstream):
        """Wall Mach should increase monotonically from inflection to exit."""
        wmn = full_downstream['wmn']
        # Allow very small non-monotonicity near exit (numerical)
        diffs = np.diff(wmn)
        assert np.all(diffs > -1e-4), (
            f"Mach not increasing: min diff = {diffs.min()}"
        )

    def test_wall_angle_decreasing(self, full_downstream):
        """Wall angle should generally decrease from eta to 0."""
        wan = full_downstream['wan']
        assert wan[0] > wan[-1], "Angle should decrease"
        assert wan[0] > 8.0, "First angle should be near eta=8.67"


# =========================================================================
# Full contour integration tests
# =========================================================================

class TestSivellsFullContour:
    """Test sivells_nozzle() with downstream=True."""

    def test_full_contour_basic(self):
        """Full contour runs and returns two arrays."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13,
            ie=0, downstream=True
        )
        assert len(x) == len(y)
        assert len(x) > 41  # more than upstream alone

    def test_full_contour_x_monotonic(self):
        """Full contour x should be strictly increasing."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13,
            ie=0, downstream=True
        )
        assert np.all(np.diff(x) > 0), "x must be strictly increasing"

    def test_full_contour_y_monotonic(self):
        """Full contour y should be non-decreasing."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13,
            ie=0, downstream=True
        )
        assert np.all(np.diff(y) >= -1e-6), "y must be non-decreasing"

    def test_throat_at_origin(self):
        """First point should be at throat (x≈0, y≈1)."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, ie=0, downstream=True
        )
        assert abs(x[0]) < 0.01
        assert abs(y[0] - 1.0) < 0.01

    def test_exit_area_ratio(self):
        """Exit y/y_throat should be close to the design area ratio (planar)."""
        from nozzle.gas import area_mach_ratio
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, ie=0, downstream=True
        )
        # For planar (ie=0), area ratio = y_exit / y_throat (linear, not squared)
        ar_actual = y[-1] / y[0]
        ar_design = area_mach_ratio(4.0, 1.4)
        # Allow 5% tolerance for the planar case
        assert abs(ar_actual - ar_design) / ar_design < 0.05, (
            f"Exit AR={ar_actual:.4f}, design AR={ar_design:.4f}"
        )

    def test_inflection_continuity(self):
        """Contour should be continuous — no large jumps relative to spacing."""
        x, y = sivells_nozzle(
            M_exit=4.0, gamma=1.4, rc=6.0,
            inflection_angle_deg=8.67, n_char=41, n_axis=21, nx=13,
            ie=0, downstream=True
        )
        # Max gap should be less than 1% of total nozzle length
        dx = np.diff(x)
        L = x[-1] - x[0]
        assert np.max(dx) / L < 0.01, (
            f"Large gap: max dx={np.max(dx):.4f}, L={L:.2f}, "
            f"ratio={np.max(dx)/L:.4f}"
        )
