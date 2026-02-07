"""Tests for ported Sivells CONTUR utility functions.

Reference values from CONTUR Mach 4 test case output and
known analytical results.
"""

import numpy as np
import pytest
from nozzle.sivells import (
    contur_gas_constants,
    contur_fg_constants,
    cubic_solve,
    conic_derivatives,
    source_derivatives,
    parabolic_derivative,
    lagrange_interp_coeffs,
    ofeld,
    sivells_throat_characteristic_with_wo,
    sivells_axial,
    sivells_perfc,
)

GAMMA = 1.4


class TestGasConstants:
    """Validate CONTUR gas constant computation."""

    def test_gamma_14_constants(self):
        """Known values from main.f comments for γ=1.4."""
        g = contur_gas_constants(1.4)
        assert abs(g['gm'] - 0.4) < 1e-10
        assert abs(g['g1'] - 2.5) < 1e-10
        assert abs(g['g9'] - 5.0) < 1e-10
        assert abs(g['g8'] - 0.2) < 1e-10
        assert abs(g['g7'] - 1.2) < 1e-10
        assert abs(g['g6'] - 5.0/6.0) < 1e-10
        assert abs(g['g5'] - 1.0/6.0) < 1e-10
        assert abs(g['g2'] - np.sqrt(6.0)) < 1e-10
        assert abs(g['ga'] - 3.0) < 1e-10

    def test_fg_constants_axisymmetric(self):
        """Known values from axial.f comment for γ=1.4, ie=1."""
        fg = contur_fg_constants(1.4, ie=1)
        assert abs(fg['gc'] - 0.10833333) < 1e-6
        assert abs(fg['gd'] - 0.236099537) < 1e-6
        assert abs(fg['ge'] - 0.65833333) < 1e-6
        assert abs(fg['gf'] - 1.40036111) < 1e-5
        assert abs(fg['gh'] - 0.13055556) < 1e-6
        assert abs(fg['gi'] - 0.2020177469) < 1e-6

    def test_fg_constants_planar(self):
        """Known values from axial.f comment for γ=1.4, ie=0."""
        fg = contur_fg_constants(1.4, ie=0)
        assert abs(fg['gc'] - (-0.011111)) < 1e-4
        assert abs(fg['gd'] - 0.2041851852) < 1e-6
        assert abs(fg['ge'] - 0.8761904762) < 1e-6
        assert abs(fg['gf'] - 1.155513228) < 1e-5


class TestCubicSolve:
    """Test Cardano's formula for cubic equations."""

    def test_simple_cubic(self):
        """x³ - 6x² + 11x - 6 = 0 has roots 1, 2, 3."""
        # Coefficients: ea=1, eb=-6, ec=11, ed=-6
        root = cubic_solve(1.0, -6.0, 11.0, -6.0)
        assert abs(root - 1.0) < 1e-10

    def test_cubic_one_positive_root(self):
        """x³ + x² + x - 3 = 0. Only one positive root near x≈1."""
        root = cubic_solve(1.0, 1.0, 1.0, -3.0)
        # Check it's actually a root
        val = root**3 + root**2 + root - 3.0
        assert abs(val) < 1e-8

    def test_cubic_from_contur_output(self):
        """CONTUR reports: FROM CUBIC, X/Y* = 0.10203110 FOR W= 1.0.

        This solves: cwoppp/6 * x³ + awopp/2 * x² + awop * x + (wo - 1.0) = 0
        With wo = 0.97501029, awop = 0.24495849, awopp = 6.6054897e-4,
        cwoppp = -4.0199752e-2.
        """
        cwoppp = -4.0199752e-2
        awopp = 6.6054897e-4
        awop = 0.24495849
        wo = 0.97501029

        # ea*x³ + eb*x² + ec*x + ed = 0
        ea = cwoppp / 6.0
        eb = awopp / 2.0
        ec = awop
        ed = wo - 1.0

        root = cubic_solve(ea, eb, ec, ed)
        assert abs(root - 0.10203110) < 1e-5, (
            f"cubic_solve gave {root}, expected 0.10203110"
        )


class TestConicDerivatives:
    """Test Mach number derivatives in radial flow."""

    def test_conic_m2(self):
        """At M=2.0, b[0] = (A/A*)^qt = sqrt(A/A*)."""
        from nozzle.gas import area_mach_ratio
        b = conic_derivatives(2.0, gamma=1.4, ie=1)
        ar = area_mach_ratio(2.0, 1.4)
        assert abs(b[0] - np.sqrt(ar)) < 1e-8

    def test_conic_m1_limit(self):
        """Near M=1, derivatives should be finite."""
        b = conic_derivatives(1.001, gamma=1.4, ie=1)
        assert np.isfinite(b[0])
        assert np.isfinite(b[1])
        # b[2] and b[3] may be large but finite
        assert np.isfinite(b[2])

    def test_conic_m4(self):
        """At M=4.0, check area ratio."""
        from nozzle.gas import area_mach_ratio
        b = conic_derivatives(4.0, gamma=1.4, ie=1)
        ar = area_mach_ratio(4.0, 1.4)
        assert abs(b[0] - np.sqrt(ar)) < 1e-8


class TestSourceDerivatives:
    """Test velocity derivatives in radial flow."""

    def test_source_at_wwo(self):
        """At the throat characteristic velocity, check radius ratio."""
        # From CONTUR output: wwo = 1.0524571 for Mach 4 case
        wwo = 1.0524571
        b = source_derivatives(wwo, gamma=1.4, ie=1)
        # b[0] should be r/r* at this velocity
        assert b[0] > 0
        assert np.isfinite(b[1])

    def test_source_near_sonic(self):
        """Near w=1.0 (throat), r/r* ≈ 1. Exact w=1.0 is singular."""
        # w = V/a* = 1 at sonic throat is a singularity (ww1 = w²-1 = 0)
        # Test slightly supersonic instead
        b = source_derivatives(1.001, gamma=1.4, ie=1)
        assert b[0] > 0.99 and b[0] < 1.01
        assert np.isfinite(b[1])


class TestParabolicDerivative:
    """Test scond.f port — finite difference derivatives."""

    def test_linear_function(self):
        """For y = 2x + 1, derivative should be 2 everywhere."""
        x = np.array([0.0, 0.5, 1.0, 2.0, 3.5])
        y = 2.0 * x + 1.0
        c = parabolic_derivative(x, y)
        np.testing.assert_allclose(c, 2.0, atol=1e-10)

    def test_quadratic_function(self):
        """For y = x², derivative should be 2x."""
        x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.0])
        y = x**2
        c = parabolic_derivative(x, y)
        expected = 2.0 * x
        np.testing.assert_allclose(c, expected, atol=1e-8)

    def test_unequal_spacing(self):
        """Works with unequally spaced points."""
        x = np.array([0.0, 0.1, 0.5, 1.5, 4.0])
        y = np.sin(x)
        c = parabolic_derivative(x, y)
        expected = np.cos(x)
        # Parabolic approximation to sin(x) derivative
        # Tolerance is loose: 5 widely-spaced points for a transcendental
        np.testing.assert_allclose(c[1:-1], expected[1:-1], atol=0.15)


class TestLagrangeInterp:
    """Test twixt.f port — 4-point Lagrange interpolation."""

    def test_exact_for_cubic(self):
        """4-point Lagrange should be exact for cubic polynomials."""
        s = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = s**3  # cubic function

        # Interpolate at x=2.5
        xbl = 2.5
        gma, gmb, gmc, gmd, kbl = lagrange_interp_coeffs(s, xbl)
        j = kbl - 1
        y_interp = gma * y[j - 2] + gmb * y[j - 1] + gmc * y[j] + gmd * y[j + 1]
        assert abs(y_interp - 2.5**3) < 1e-8

    def test_linear_interp(self):
        """Exact for linear functions."""
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * s + 2.0

        xbl = 3.7
        gma, gmb, gmc, gmd, kbl = lagrange_interp_coeffs(s, xbl)
        j = kbl - 1
        y_interp = gma * y[j - 2] + gmb * y[j - 1] + gmc * y[j] + gmd * y[j + 1]
        assert abs(y_interp - (3.0 * 3.7 + 2.0)) < 1e-10


class TestOfeld:
    """Test ofeld.f port — interior point computation."""

    def test_planar_no_source(self):
        """In 2D planar (ie=0), no source terms, exact compatibility."""
        from nozzle.gas import prandtl_meyer

        M_A = 2.0
        M_B = 2.0
        psi_A = prandtl_meyer(M_A, GAMMA)
        psi_B = prandtl_meyer(M_B, GAMMA)
        theta_A = 0.05   # small positive angle
        theta_B = -0.05  # small negative angle

        # a = C- upstream, b = C+ upstream (CONTUR convention)
        # a: [x, y, M, psi, theta]
        a = [1.0, 0.5, M_A, psi_A, theta_A]
        b = [1.0, 0.7, M_B, psi_B, theta_B]

        c, converged = ofeld(a, b, gamma=GAMMA, ie=0)

        # For planar, exact solution:
        # θ_C = (θ_A + θ_B + ψ_A - ψ_B)/2
        # ψ_C = (ψ_A + ψ_B + θ_A - θ_B)/2
        theta_exact = 0.5 * (theta_A + theta_B + psi_A - psi_B)
        psi_exact = 0.5 * (psi_A + psi_B + theta_A - theta_B)

        assert abs(c[4] - theta_exact) < 1e-10, (
            f"theta: got {c[4]}, expected {theta_exact}"
        )
        assert abs(c[3] - psi_exact) < 1e-10, (
            f"psi: got {c[3]}, expected {psi_exact}"
        )

    def test_axisymmetric_converges(self):
        """Axisymmetric case converges for typical upstream conditions."""
        from nozzle.gas import prandtl_meyer

        M_A = 1.5
        M_B = 1.5
        psi_A = prandtl_meyer(M_A, GAMMA)
        psi_B = prandtl_meyer(M_B, GAMMA)

        a = [1.0, 0.0, M_A, psi_A, 0.0]  # axis point
        b = [1.0, 0.05, M_B, psi_B, 0.01]

        c, converged = ofeld(a, b, gamma=GAMMA, ie=1)
        assert converged
        assert c[2] > 1.0, "Output M should be supersonic"


class TestThroatCharacteristic:
    """Test sivells_throat_characteristic_with_wo against CONTUR output."""

    def test_mach4_throat_characteristic(self):
        """Compare throat characteristic with CONTUR Mach 4 output.

        CRITICAL: The CONTUR Mach 4 test case is PLANAR (ie=0), not
        axisymmetric. Verified by compiling and running the F90 code
        with debug output: ie=0, gb=0.
        """
        # Mach 4 case: RC=6, rto=7, wo=0.97501029, ie=0 (PLANAR)
        rto = 7.0
        wo = 0.97501029320773414  # full precision from Fortran debug

        fc, result = sivells_throat_characteristic_with_wo(
            rto, wo, gamma=1.4, ie=0, n_points=21
        )

        # Check Mach at axis end (point 1 in CONTUR = our index 0)
        # CONTUR: M = 1.0639748
        assert abs(result['amn'] - 1.0639748) < 0.001, (
            f"Axis Mach: got {result['amn']}, expected ~1.0639748"
        )

        # Check wwo matches CONTUR
        assert abs(result['wwo'] - 1.0524571) < 1e-5, (
            f"wwo: got {result['wwo']}, expected ~1.0524571"
        )

        # Check that the characteristic has reasonable shape
        assert fc[1, 0] < 0.01, f"Axis y should be ~0, got {fc[1, 0]}"
        assert abs(fc[1, -1] - 1.0) < 0.01, f"Wall y should be ~1, got {fc[1, -1]}"

        # Mach should be > 1 everywhere on the characteristic
        for j in range(21):
            if fc[2, j] > 0:  # skip uninitialized
                assert fc[2, j] >= 1.0, f"M<1 at point {j}: {fc[2, j]}"

    def test_throat_characteristic_symmetry(self):
        """Throat characteristic Mach should be symmetric (same at wall and axis).

        CONTUR Mach 4 output shows M=1.0640 at both point 1 (axis) and
        point 21 (wall), with a dip to M=1.0399 in the middle.
        """
        rto = 7.0
        wo = 0.97501029320773414
        fc, result = sivells_throat_characteristic_with_wo(
            rto, wo, gamma=1.4, ie=0, n_points=21
        )
        # Wall and axis M should be close (symmetric characteristic)
        M_axis = fc[2, 0]
        M_wall = fc[2, -1]
        assert abs(M_axis - M_wall) < 0.01, (
            f"Axis M={M_axis:.6f} vs Wall M={M_wall:.6f} (should be close)"
        )

    def test_mass_flow_normalized(self):
        """Mass flow: 1 at wall (start), 0 at axis (end of integration)."""
        rto = 7.0
        wo = 0.97501029320773414

        fc, result = sivells_throat_characteristic_with_wo(
            rto, wo, gamma=1.4, ie=0, n_points=21
        )

        # CONTUR convention: march from wall (y=1) to axis (y=0)
        # Mass flow fraction = 1 - accumulated/total
        # At wall (start): 1.0 (nothing integrated yet)
        # At axis (end): 0.0 (all flow accounted for)
        assert abs(fc[5, -1] - 1.0) < 0.01, (
            f"Mass flow at wall: {fc[5, -1]}, expected ~1"
        )
        assert abs(fc[5, 0]) < 0.01, (
            f"Mass flow at axis: {fc[5, 0]}, expected ~0"
        )


class TestAxial:
    """Test sivells_axial against CONTUR Mach 4 output.

    Reference values from compiling and running the CONTUR F90 code
    with the Mach 4 test case. This is PLANAR (ie=0).
    Input: etad=8.67, rc=6.0, fmach=0, bmach=3.2, cmach=4.0, ix=0, nx=13.
    """

    @pytest.fixture
    def mach4_result(self):
        """Run sivells_axial once for all tests."""
        return sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.2, cmach=4.0,
            ie=0, n_char=41, n_axis=21, nx=13
        )

    def test_throat_parameters(self, mach4_result):
        """Throat center velocity and mass ratio."""
        r = mach4_result
        assert abs(r['wo'] - 0.97501029) < 1e-6
        assert abs(r['tk'] - 0.99937483) < 1e-6
        assert abs(r['yo'] - 0.15141471) < 1e-6

    def test_throat_characteristic_mach(self, mach4_result):
        """Throat characteristic Mach at axis."""
        r = mach4_result
        assert abs(r['wwo'] - 1.0524571) < 1e-5
        assert abs(r['wwop'] - 1.86441556) < 1e-5

    def test_emach_convergence(self, mach4_result):
        """Inflection Mach and downstream Mach converge."""
        r = mach4_result
        assert abs(r['emach'] - 1.44128) < 1e-4, (
            f"emach: got {r['emach']}, expected ~1.44128"
        )
        assert abs(r['fmach'] - 2.04174) < 1e-4, (
            f"fmach: got {r['fmach']}, expected ~2.04174"
        )
        assert r['nocon'] <= 65, f"Too many iterations: {r['nocon']}"

    def test_velocity_derivatives_at_inflection(self, mach4_result):
        """Velocity derivatives at the inflection point (from trans)."""
        r = mach4_result
        assert abs(r['wi'] - 1.05245706) < 1e-5
        assert abs(r['wip'] - 1.60685345) < 1e-4
        assert abs(r['wipp'] - (-0.52752781)) < 1e-4

    def test_velocity_at_exit(self, mach4_result):
        """Velocity and derivatives at the exit point."""
        r = mach4_result
        assert abs(r['we'] - 1.32705918) < 1e-5
        assert abs(r['wep'] - 1.08183651) < 1e-4
        assert abs(r['wepp'] - (-4.8823711)) < 1e-3

    def test_geometry_positions(self, mach4_result):
        """Key x-positions along the axis."""
        r = mach4_result
        assert abs(r['xoi'] - 0.04795502) < 1e-5, (
            f"xoi: got {r['xoi']}"
        )
        assert abs(r['xi'] - 0.94457600) < 1e-5, (
            f"xi: got {r['xi']}"
        )
        assert abs(r['xo'] - 0.89662098) < 1e-5, (
            f"xo: got {r['xo']}"
        )
        assert abs(r['xie'] - 0.19409491) < 1e-5, (
            f"xie: got {r['xie']}"
        )
        assert abs(r['xe'] - 1.13867091) < 1e-5, (
            f"xe: got {r['xe']}"
        )

    def test_polynomial_coefficients(self, mach4_result):
        """3rd-degree polynomial coefficients for velocity distribution."""
        c = mach4_result['c']
        assert abs(c[0] - 1.0524571) < 1e-5
        assert abs(c[1] - 0.31188208) < 1e-5
        assert abs(c[2] - (-9.9367340e-03)) < 1e-7
        assert abs(c[3] - (-2.7343222e-02)) < 1e-6
        assert c[4] == 0.0  # 3rd-degree: c5=0
        assert c[5] == 0.0  # c6=0

    def test_axis_endpoints(self, mach4_result):
        """Axis array at exit and inflection points."""
        axis = mach4_result['axis']
        # k=0 is exit (x=xe)
        assert abs(axis[0, 0] - 1.13867) < 1e-4
        assert abs(axis[2, 0] - 1.441279) < 1e-4
        # k=20 is inflection (x=xi)
        assert abs(axis[0, 20] - 0.94458) < 1e-4
        assert abs(axis[2, 20] - 1.063975) < 1e-4

    def test_axis_mach_monotonic(self, mach4_result):
        """Mach should decrease monotonically from exit to inflection."""
        axis = mach4_result['axis']
        mach = axis[2, :]
        for k in range(1, 21):
            assert mach[k] < mach[k - 1], (
                f"Mach not decreasing at k={k}: "
                f"M[{k-1}]={mach[k-1]:.6f}, M[{k}]={mach[k]:.6f}"
            )

    def test_axis_all_supersonic(self, mach4_result):
        """All axis points should be supersonic."""
        axis = mach4_result['axis']
        for k in range(21):
            assert axis[2, k] > 1.0, (
                f"Subsonic at k={k}: M={axis[2, k]:.6f}"
            )

    def test_axis_psi_positive(self, mach4_result):
        """Prandtl-Meyer angle should be positive at all axis points."""
        axis = mach4_result['axis']
        for k in range(21):
            assert axis[3, k] > 0, (
                f"psi <= 0 at k={k}: psi={axis[3, k]:.6f}"
            )


class TestPerfc:
    """Test sivells_perfc against CONTUR Mach 4 upstream wall output.

    Reference: 41 wall points from docs/references/external_codes/contur/docs/output.txt
    lines 387-433. PLANAR (ie=0), etad=8.67, rc=6.0, bmach=3.2, cmach=4.0.
    """

    # CONTUR Mach 4 upstream wall reference data (41 points, throat to inflection)
    REF_X = np.array([
        8.9662098e-01, 9.0214410e-01, 9.0770503e-01, 9.1329910e-01,
        9.1892322e-01, 9.2458916e-01, 9.3032042e-01, 9.3611810e-01,
        9.4200495e-01, 9.4800586e-01, 9.5415029e-01, 9.6047046e-01,
        9.6700302e-01, 9.7378809e-01, 9.8087017e-01, 9.8828909e-01,
        9.9609193e-01, 1.0043282e+00, 1.0130445e+00, 1.0222971e+00,
        1.0321348e+00, 1.0393632e+00, 1.0497442e+00, 1.0616497e+00,
        1.0746142e+00, 1.0884116e+00, 1.1029084e+00, 1.1180237e+00,
        1.1337043e+00, 1.1499141e+00, 1.1666232e+00, 1.1838094e+00,
        1.2014532e+00, 1.2195352e+00, 1.2380358e+00, 1.2569334e+00,
        1.2762043e+00, 1.2958198e+00, 1.3157496e+00, 1.3359545e+00,
        1.3563968e+00])

    REF_Y = np.array([
        1.5141471e-01, 1.5143156e-01, 1.5148237e-01, 1.5156757e-01,
        1.5168747e-01, 1.5184274e-01, 1.5203461e-01, 1.5226376e-01,
        1.5253191e-01, 1.5284126e-01, 1.5319460e-01, 1.5359544e-01,
        1.5404803e-01, 1.5455735e-01, 1.5512903e-01, 1.5576883e-01,
        1.5648352e-01, 1.5728047e-01, 1.5816753e-01, 1.5915367e-01,
        1.6024778e-01, 1.6107893e-01, 1.6230970e-01, 1.6376995e-01,
        1.6541319e-01, 1.6721595e-01, 1.6916208e-01, 1.7124033e-01,
        1.7344211e-01, 1.7575947e-01, 1.7818468e-01, 1.8071073e-01,
        1.8333074e-01, 1.8603777e-01, 1.8882487e-01, 1.9168520e-01,
        1.9461175e-01, 1.9759696e-01, 2.0063354e-01, 2.0371400e-01,
        2.0683110e-01])

    REF_M = np.array([
        1.0639748e+00, 1.0768502e+00, 1.0898310e+00, 1.1029005e+00,
        1.1159998e+00, 1.1291475e+00, 1.1424141e+00, 1.1557114e+00,
        1.1690643e+00, 1.1824876e+00, 1.1959967e+00, 1.2095971e+00,
        1.2232908e+00, 1.2370625e+00, 1.2508687e+00, 1.2646753e+00,
        1.2784811e+00, 1.2923974e+00, 1.3065794e+00, 1.3211122e+00,
        1.3361384e+00, 1.3469175e+00, 1.3620890e+00, 1.3791320e+00,
        1.3973195e+00, 1.4162911e+00, 1.4358816e+00, 1.4559761e+00,
        1.4764956e+00, 1.4973639e+00, 1.5185491e+00, 1.5400047e+00,
        1.5616653e+00, 1.5834933e+00, 1.6054394e+00, 1.6274217e+00,
        1.6494019e+00, 1.6712857e+00, 1.6930150e+00, 1.7144850e+00,
        1.7356275e+00])

    REF_ANGLE = np.array([
        0.0, 3.4856826e-01, 6.9751151e-01, 1.0465024e+00,
        1.3948738e+00, 1.7425667e+00, 2.0894719e+00, 2.4344431e+00,
        2.7778978e+00, 3.1191385e+00, 3.4577595e+00, 3.7933891e+00,
        4.1249167e+00, 4.4505075e+00, 4.7683744e+00, 5.0773237e+00,
        5.3761470e+00, 5.6649207e+00, 5.9432593e+00, 6.2108869e+00,
        6.4690204e+00, 6.6422599e+00, 6.8691230e+00, 7.1032523e+00,
        7.3304832e+00, 7.5425380e+00, 7.7354446e+00, 7.9093084e+00,
        8.0632129e+00, 8.1962384e+00, 8.3092745e+00, 8.4034115e+00,
        8.4799902e+00, 8.5404878e+00, 8.5868368e+00, 8.6209093e+00,
        8.6441045e+00, 8.6581753e+00, 8.6658795e+00, 8.6692098e+00,
        8.6700000e+00])

    REF_WALTAN = np.array([
        0.0, 6.0837388e-03, 1.2174474e-02, 1.8266945e-02,
        2.4349952e-02, 3.0422906e-02, 3.6484340e-02, 4.2514635e-02,
        4.8521488e-02, 5.4493079e-02, 6.0422659e-02, 6.6304037e-02,
        7.2118019e-02, 7.7832608e-02, 8.3416510e-02, 8.8848707e-02,
        9.4107814e-02, 9.9194958e-02, 1.0410309e-01, 1.0882702e-01,
        1.1338793e-01, 1.1645146e-01, 1.2046653e-01, 1.2461423e-01,
        1.2864375e-01, 1.3240787e-01, 1.3583529e-01, 1.3892706e-01,
        1.4166607e-01, 1.4403519e-01, 1.4604954e-01, 1.4772799e-01,
        1.4909398e-01, 1.5017351e-01, 1.5100079e-01, 1.5160908e-01,
        1.5202325e-01, 1.5227451e-01, 1.5241210e-01, 1.5247157e-01,
        1.5248569e-01])

    REF_SECD = np.array([
        1.1007297e+00, 1.0983988e+00, 1.0921931e+00, 1.0853538e+00,
        1.0767323e+00, 1.0647624e+00, 1.0489157e+00, 1.0303294e+00,
        1.0078707e+00, 9.8025099e-01, 9.4804658e-01, 9.1062188e-01,
        8.6656751e-01, 8.1591867e-01, 7.6098525e-01, 7.0383779e-01,
        6.4658824e-01, 5.9114755e-01, 5.3760645e-01, 4.8780261e-01,
        4.4067621e-01, 4.0861198e-01, 3.6889054e-01, 3.3039838e-01,
        2.9240451e-01, 2.5506924e-01, 2.2081877e-01, 1.8988463e-01,
        1.6065092e-01, 1.3354777e-01, 1.0926981e-01, 8.7674546e-02,
        6.8669625e-02, 5.2294962e-02, 3.8519233e-02, 2.6892551e-02,
        1.7189088e-02, 9.8799668e-03, 4.9370984e-03, 1.8235753e-03,
        0.0])

    @pytest.fixture
    def perfc_result(self):
        """Run sivells_axial + sivells_perfc once for all tests."""
        axial = sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.2, cmach=4.0,
            ie=0, n_char=41, n_axis=21, nx=13
        )
        return sivells_perfc(axial, gamma=1.4, ie=0)

    def test_n_wall_points(self, perfc_result):
        """41 wall points from throat to inflection."""
        assert perfc_result['n_wall'] == 41

    def test_mass_flow(self, perfc_result):
        """Mass flow from first characteristic should be ~1.0."""
        assert abs(perfc_result['mass'] - 1.0) < 1e-6

    def test_wall_x_coordinates(self, perfc_result):
        """Wall x-coordinates match CONTUR to < 1e-6."""
        np.testing.assert_allclose(
            perfc_result['wax'], self.REF_X, atol=1e-6,
            err_msg="Wall x-coordinates disagree with CONTUR"
        )

    def test_wall_y_coordinates(self, perfc_result):
        """Wall y-coordinates (from slope integration) match CONTUR to < 1e-4."""
        np.testing.assert_allclose(
            perfc_result['way'], self.REF_Y, atol=1e-4,
            err_msg="Wall y-coordinates disagree with CONTUR"
        )

    def test_wall_mach(self, perfc_result):
        """Wall Mach numbers match CONTUR to < 1e-6."""
        np.testing.assert_allclose(
            perfc_result['wmn'], self.REF_M, atol=1e-6,
            err_msg="Wall Mach numbers disagree with CONTUR"
        )

    def test_wall_angle(self, perfc_result):
        """Wall flow angles match CONTUR to < 0.01 degrees."""
        np.testing.assert_allclose(
            perfc_result['wan'], self.REF_ANGLE, atol=0.01,
            err_msg="Wall flow angles disagree with CONTUR"
        )

    def test_wall_waltan(self, perfc_result):
        """Wall tan(theta) matches CONTUR to < 2e-4.

        Tolerance reflects ofeld iterative convergence propagating through tan().
        """
        np.testing.assert_allclose(
            perfc_result['waltan'], self.REF_WALTAN, atol=2e-4,
            err_msg="Wall tan(theta) disagrees with CONTUR"
        )

    def test_wall_secd(self, perfc_result):
        """Second derivative matches CONTUR to < 0.01."""
        np.testing.assert_allclose(
            perfc_result['secd'], self.REF_SECD, atol=0.01,
            err_msg="Second derivative disagrees with CONTUR"
        )

    def test_throat_endpoint(self, perfc_result):
        """First point (throat) has correct values."""
        r = perfc_result
        assert abs(r['wax'][0] - 0.89662098) < 1e-5
        assert abs(r['way'][0] - 0.15141471) < 1e-5
        assert abs(r['wan'][0]) < 1e-10  # theta = 0 at throat

    def test_inflection_endpoint(self, perfc_result):
        """Last point (inflection) has correct values."""
        r = perfc_result
        assert abs(r['wax'][-1] - 1.3563968) < 1e-5
        assert abs(r['way'][-1] - 0.2068311) < 1e-5
        assert abs(r['wan'][-1] - 8.67) < 0.01
        assert abs(r['wmn'][-1] - 1.7356275) < 1e-5

    def test_wall_x_monotonic(self, perfc_result):
        """Wall x-coordinates should increase monotonically."""
        wax = perfc_result['wax']
        for i in range(1, len(wax)):
            assert wax[i] > wax[i - 1], (
                f"x not increasing at i={i}: x[{i-1}]={wax[i-1]:.6f}, "
                f"x[{i}]={wax[i]:.6f}"
            )

    def test_wall_y_monotonic(self, perfc_result):
        """Wall y-coordinates should increase monotonically."""
        way = perfc_result['way']
        for i in range(1, len(way)):
            assert way[i] >= way[i - 1], (
                f"y not increasing at i={i}: y[{i-1}]={way[i-1]:.6f}, "
                f"y[{i}]={way[i]:.6f}"
            )

    def test_wall_mach_monotonic(self, perfc_result):
        """Wall Mach should increase monotonically from throat to inflection."""
        wmn = perfc_result['wmn']
        for i in range(1, len(wmn)):
            assert wmn[i] > wmn[i - 1], (
                f"M not increasing at i={i}: M[{i-1}]={wmn[i-1]:.6f}, "
                f"M[{i}]={wmn[i]:.6f}"
            )

    def test_wall_angle_monotonic(self, perfc_result):
        """Wall angle should increase from 0 (throat) to eta (inflection)."""
        wan = perfc_result['wan']
        for i in range(1, len(wan)):
            assert wan[i] >= wan[i - 1], (
                f"angle not increasing at i={i}: a[{i-1}]={wan[i-1]:.4f}, "
                f"a[{i}]={wan[i]:.4f}"
            )

    def test_secd_decreasing(self, perfc_result):
        """Second derivative should decrease from throat to inflection."""
        secd = perfc_result['secd']
        assert secd[0] > secd[-1], "secd should decrease from throat to inflection"
        assert abs(secd[-1]) < 1e-10, "secd should be 0 at inflection"
        assert secd[0] > 1.0, "secd at throat should be > 1 (curvature)"
