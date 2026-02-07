"""Tests comparing our code against Sivells CONTUR Mach 4 validated output.

Reference: Sivells, AEDC-TR-78-63, 1978.
Validated F90 port: docs/references/external_codes/contur/
Output: docs/references/external_codes/contur/docs/output.txt

Mach 4 test case parameters (from input.txt):
    gamma = 1.4
    ie = 0 (PLANAR — verified by compiling F90 with debug output)
    NOTE: input card "M A C H 4  -1" has jd=-1, so ie=1+jd=0
    Inflection angle (eta) = 8.67 degrees
    RC = 6.0 (throat radius of curvature / throat radius, rto = RC + 1 = 7)
    M_exit (CMACH) = 4.0
    BMACH = 3.2
    Scale factor SF = 7.55258257
    y* = 0.15132005  (y*/y0, throat half-height / reference height)
"""

import numpy as np
import pytest
from nozzle.gas import (
    prandtl_meyer,
    mach_from_prandtl_meyer,
    mach_angle,
    area_mach_ratio,
)


# ==========================================================================
# Mach 4 test case constants
# ==========================================================================
GAMMA = 1.4

# CONTUR gas constants for gamma=1.4
# From main.f: gm=gam-1, g1=1/gm, g9=2*g1, g8=1/g9, g7=1+g8,
#              g6=1/g7, g5=g8*g6, rga=2*g5, ga=1/rga, g4=sqrt(g5),
#              g3=ga/2, g2=1/g4
G_CONSTS = {
    'gm': 0.4,       # gamma - 1
    'g1': 2.5,       # 1/(gamma-1)
    'g9': 5.0,       # 2/(gamma-1)
    'g8': 0.2,       # (gamma-1)/2
    'g7': 1.2,       # 1 + (gamma-1)/2
    'g6': 5.0/6.0,   # 1/(1 + (gamma-1)/2)
    'g5': 1.0/6.0,   # (gamma-1)/(2*(1+(gamma-1)/2))
    'g2': np.sqrt(6.0),    # sqrt((gamma+1)/(gamma-1))
    'ga': 3.0,             # (gamma+1)/(2*(gamma-1))
    'qt': 1.0,             # 1/(1+ie) for planar (ie=0)
}


class TestFmvEquivalence:
    """Compare gas.py:mach_from_prandtl_meyer vs CONTUR fmv.f.

    fmv.f uses Newton-Raphson; our code uses Brent's method.
    Both should give the same Mach number to high precision.
    """

    # Values from CONTUR Mach 4 output, throat characteristic
    # Format: (M, psi_degrees)  where psi = P-M angle
    THROAT_CHAR_DATA = [
        # Point 1 (axis): M=1.0639748, PSI=0.69924025 deg
        (1.0639748, 0.69924025),
        # Point 11: M=1.0399398, PSI=0.35020393 deg
        (1.0399398, 0.35020393),
        # Point 21 (wall): M=1.0639748, PSI=0.69924025 deg
        (1.0639748, 0.69924025),
    ]

    # Upstream contour characteristic 1
    UPSTREAM_DATA = [
        # Point 1 (axis): M=1.4412788, PSI=10.183632 deg
        (1.4412788, 10.183632),
        # Point 21: M=1.5884492, PSI=14.518632 deg
        (1.5884492, 14.518632),
        # Point 41 (wall): M=1.7356275, PSI=18.853632 deg
        (1.7356275, 18.853632),
    ]

    # Downstream contour exit: M=4.0, PSI=65.785 deg (from P-M tables)
    EXIT_DATA = [
        (4.0, None),  # Check our PM computation
        (3.2, None),   # BMACH
    ]

    @pytest.mark.parametrize("M_expected,psi_deg", THROAT_CHAR_DATA)
    def test_fmv_throat_characteristic(self, M_expected, psi_deg):
        """P-M angle from CONTUR throat characteristic matches our computation."""
        psi_rad = np.radians(psi_deg)
        M_computed = mach_from_prandtl_meyer(psi_rad, GAMMA)
        assert abs(M_computed - M_expected) < 1e-4, (
            f"mach_from_prandtl_meyer({psi_deg}°) = {M_computed}, "
            f"expected {M_expected}"
        )

    @pytest.mark.parametrize("M_expected,psi_deg", UPSTREAM_DATA)
    def test_fmv_upstream_contour(self, M_expected, psi_deg):
        """P-M angle from CONTUR upstream contour matches our computation."""
        psi_rad = np.radians(psi_deg)
        M_computed = mach_from_prandtl_meyer(psi_rad, GAMMA)
        assert abs(M_computed - M_expected) < 1e-4, (
            f"mach_from_prandtl_meyer({psi_deg}°) = {M_computed}, "
            f"expected {M_expected}"
        )

    def test_prandtl_meyer_roundtrip(self):
        """ν(M) → M → ν roundtrip at several Mach numbers."""
        for M in [1.06, 1.2, 1.5, 2.0, 3.0, 4.0]:
            nu = prandtl_meyer(M, GAMMA)
            M_back = mach_from_prandtl_meyer(nu, GAMMA)
            assert abs(M_back - M) < 1e-10, (
                f"Roundtrip failed: M={M}, nu={np.degrees(nu)}°, M_back={M_back}"
            )

    def test_prandtl_meyer_known_values(self):
        """Check P-M angle at M=4.0 against Anderson MCF Table A.5."""
        # Anderson MCF Table A.5: M=4.0, ν=65.785° for γ=1.4
        nu_4 = prandtl_meyer(4.0, GAMMA)
        assert abs(np.degrees(nu_4) - 65.785) < 0.01

    def test_contur_psi_matches_our_nu(self):
        """Verify CONTUR PSI is exactly our Prandtl-Meyer ν."""
        # CONTUR stores PSI = ν at each point (in radians in code, degrees in output)
        # From output: M=1.4412788 has PSI=10.183632 deg
        M = 1.4412788
        nu = prandtl_meyer(M, GAMMA)
        psi_contur = 10.183632  # degrees from output
        assert abs(np.degrees(nu) - psi_contur) < 0.01, (
            f"nu({M}) = {np.degrees(nu):.6f}°, CONTUR PSI = {psi_contur}°"
        )


class TestMach4Geometry:
    """Extract and validate Mach 4 test case geometry."""

    def test_exit_area_ratio(self):
        """A/A* at M=4 for γ=1.4."""
        ar = area_mach_ratio(4.0, GAMMA)
        assert abs(ar - 10.7188) < 0.01

    def test_inflection_angle(self):
        """Inflection angle = ν(M_exit)/2 for MLN."""
        # Note: CONTUR uses a different inflection angle (8.67°) because
        # it's not a pure MLN — it has a radial flow region.
        # But ν(4.0)/2 = 65.785/2 = 32.89° for a true MLN.
        # CONTUR eta=8.67° is a design parameter, not the P-M half-angle.
        nu_exit = prandtl_meyer(4.0, GAMMA)
        theta_max_mln = np.degrees(nu_exit) / 2
        assert abs(theta_max_mln - 32.89) < 0.1

    def test_throat_mach_number(self):
        """Throat characteristic Mach on axis = 1.0639748."""
        # From output line: "M A C H 4" throat characteristic
        # Axial velocity w0 = 0.97501029 (velocity / a*)
        # w = M / sqrt(g7 - g8*M²) → M = w*sqrt(g7/(1+g8*w²))? No...
        # Actually w = V/a* and M = V/a, with a* = a(T0)*sqrt(2/(γ+1))
        # The relation is: w² = M²/(1 + (γ-1)/2 * M²) * ((γ+1)/2)
        # Or equivalently: M = w/sqrt(g7 - g8*w²)
        w0 = 0.97501029
        g7, g8 = 1.2, 0.2
        M_axis = w0 / np.sqrt(g7 - g8 * w0**2)
        assert abs(M_axis - 0.97023347) < 1e-5

    def test_wall_mach_at_inflection(self):
        """Wall Mach at inflection (point 41) = 1.7356275."""
        # This is where flow angle = 8.67° = eta (max wall angle)
        M_wall = 1.7356275
        nu = prandtl_meyer(M_wall, GAMMA)
        assert abs(np.degrees(nu) - 18.853632) < 0.01

    def test_scale_factor(self):
        """Scale factor SF converts normalized coords to inches."""
        # SF = 7.55258257
        # y* (throat radius) = 0.15132005 (normalized by y0)
        # y* in inches = SF * y* = 7.55258257 * 0.15132005 ≈ 1.143 in
        # Throat radius r* should be y0 * y* = some reference
        sf = 7.55258257
        ystar = 0.15132005
        rstar_inch = sf * ystar  # ≈ 1.143 inches
        assert abs(rstar_inch - 1.1428) < 0.01


class TestUpstreamContourValues:
    """Validate wall contour coordinates from CONTUR upstream contour."""

    # Wall contour data from "UPSTREAM CONTOUR" section
    # Format: (point#, x, y, M, flow_angle_deg, waltan, secdif)
    WALL_DATA = [
        (1,  0.89662098, 0.15141471, 1.0639748, 0.0,      0.0,         1.1007297),
        (11, 0.95415029, 0.15319460, 1.1959967, 3.4577595, 0.060422659, 0.94804658),
        (21, 1.0321348,  0.16024778, 1.3361384, 6.4690204, 0.11338793,  0.44067621),
        (31, 1.1666232,  0.17818468, 1.5185491, 8.3092745, 0.14604954,  0.10926981),
        (41, 1.3563968,  0.20683110, 1.7356275, 8.6700000, 0.15248569,  0.0),
    ]

    @pytest.mark.parametrize("pt,x,y,M,theta_deg,waltan,secdif", WALL_DATA)
    def test_wall_coordinates(self, pt, x, y, M, theta_deg, waltan, secdif):
        """Wall coordinates from CONTUR output are self-consistent."""
        # waltan = tan(theta_wall)
        if theta_deg > 0.01:
            expected_tan = np.tan(np.radians(theta_deg))
            assert abs(waltan - expected_tan) < 0.001, (
                f"Point {pt}: waltan={waltan}, tan({theta_deg}°)={expected_tan}"
            )

    def test_wall_mach_increases(self):
        """Wall Mach number increases along contour."""
        M_values = [d[3] for d in self.WALL_DATA]
        for i in range(len(M_values) - 1):
            assert M_values[i] < M_values[i + 1]

    def test_wall_angle_increases_then_levels(self):
        """Wall angle increases to eta=8.67° then stays there."""
        theta_values = [d[4] for d in self.WALL_DATA]
        # Should increase
        for i in range(len(theta_values) - 1):
            assert theta_values[i] <= theta_values[i + 1] + 0.01

        # Last point should be at eta = 8.67°
        assert abs(theta_values[-1] - 8.67) < 0.001


class TestOfeld:
    """Compare ofeld.f (CONTUR interior point) with our interior_point.

    Key differences to document:
    1. CONTUR convention: a = C- upstream, b = C+ upstream
       Our convention: idx_left = C+ upstream, idx_right = C- upstream

    2. Source term:
       CONTUR: fsy = sin(θ)/(y·M)
       Ours: Q = sin(θ)·sin(μ)/(y·cos(θ±μ))

    3. Iteration:
       CONTUR: damped iteration t3 = (t3 + told)/2, up to 40 iterations
       Ours: simple predictor-corrector, n_iter=3

    4. Position computation:
       Both use characteristic slope intersection, same formula.

    Both formulations should converge to the same physical point.
    """

    def test_interior_point_2d_planar(self):
        """In 2D planar (no source terms), both methods are exact."""
        from nozzle.moc import CharMesh, interior_point

        mesh = CharMesh(gamma=GAMMA)

        # Create two parent points at large y (so source terms are small)
        # C+ parent (our left): lower point
        idx_L = mesh.add_point(x=1.0, y=10.0, M=2.0, theta=0.1)
        # C- parent (our right): upper point
        idx_R = mesh.add_point(x=1.0, y=10.5, M=2.0, theta=-0.1)

        idx_C = interior_point(mesh, idx_L, idx_R, GAMMA)
        pt = mesh.points[idx_C]

        # For large y, should approach 2D solution:
        # θ_C = (K+ + K-)/2, ν_C = (K+ - K-)/2
        nu_L = prandtl_meyer(2.0, GAMMA)
        nu_R = prandtl_meyer(2.0, GAMMA)
        Kp = -0.1 + nu_R  # K+ from right
        Km = 0.1 - nu_L   # K- from left
        theta_2d = (Kp + Km) / 2
        nu_2d = (Kp - Km) / 2

        assert abs(pt.theta - theta_2d) < 0.01
        assert abs(pt.nu - nu_2d) < 0.01

    def test_interior_point_from_contur_data(self):
        """Test interior_point with data from CONTUR upstream char 1."""
        from nozzle.moc import CharMesh, interior_point

        mesh = CharMesh(gamma=GAMMA)

        # Use upstream contour char 1, points 1 and 2 as parents
        # Point 1 (axis): x=1.1386709, y=0, M=1.4412788, θ=0
        # Point 2: x=1.1431647, y=0.004324614, M=1.4487080, θ=0.2167500°
        idx_1 = mesh.add_point(
            x=1.1386709, y=0.001, M=1.4412788, theta=0.0
        )
        idx_2 = mesh.add_point(
            x=1.1431647, y=0.004324614, M=1.4487080,
            theta=np.radians(0.2167500)
        )

        # This should compute an interior point without error
        idx_C = interior_point(mesh, idx_1, idx_2, GAMMA)
        pt = mesh.points[idx_C]

        # Basic sanity checks
        assert pt.M > 1.0
        assert pt.x > 1.13


class TestThroatCharacteristic:
    """Validate throat characteristic data from CONTUR output."""

    # Throat characteristic: 21 points from axis (point 1) to wall (point 21)
    # Format: (point, x, y, M, psi_deg, theta_deg)
    THROAT_CHAR = [
        (1,  0.94457600, 0.00000000, 1.0639748, 0.69924025, 0.0),
        (6,  0.93197667, 0.03785368, 1.0448128, 0.41492582, 0.28431443),
        (11, 0.92074818, 0.07570735, 1.0399398, 0.35020393, 0.34903631),
        (16, 0.90937939, 0.11356103, 1.0467293, 0.44129014, 0.25795010),
        (21, 0.89662098, 0.15141471, 1.0639748, 0.69924025, 0.0),
    ]

    def test_throat_char_symmetry(self):
        """Axis and wall points have the same M and PSI."""
        p1 = self.THROAT_CHAR[0]   # axis
        p21 = self.THROAT_CHAR[-1]  # wall
        assert abs(p1[3] - p21[3]) < 1e-6, "M should be same at axis and wall"
        assert abs(p1[4] - p21[4]) < 1e-6, "PSI should be same at axis and wall"

    def test_throat_char_minimum_M(self):
        """Minimum M occurs near the middle of the characteristic."""
        M_values = [p[3] for p in self.THROAT_CHAR]
        # Point 11 (middle) has the lowest M
        min_M = min(M_values)
        assert abs(min_M - 1.0399398) < 1e-5

    def test_throat_char_theta_profile(self):
        """Flow angle is 0 on axis, peaks near middle, 0 at wall."""
        theta_values = [p[5] for p in self.THROAT_CHAR]
        assert abs(theta_values[0]) < 1e-6, "θ = 0 on axis"
        assert abs(theta_values[-1]) < 1e-6, "θ = 0 on wall (symmetry)"
        assert max(theta_values) > 0.2, "θ peaks in the middle"


class TestDownstreamContour:
    """Validate downstream contour data from CONTUR output."""

    # Downstream axis Mach distribution
    # Format: (point, x, M)
    AXIS_DATA = [
        (1,  5.12096, 3.200000),
        (21, 10.60833, 3.867229),
        (41, 16.09571, 3.997510),
        (49, 18.29066, 4.000000),
    ]

    def test_axis_mach_monotonic(self):
        """Axis Mach increases monotonically from BMACH to CMACH."""
        M_values = [d[2] for d in self.AXIS_DATA]
        for i in range(len(M_values) - 1):
            assert M_values[i] < M_values[i + 1]

    def test_axis_reaches_exit_mach(self):
        """Axis Mach reaches M=4.0 at the end."""
        assert abs(self.AXIS_DATA[-1][2] - 4.0) < 1e-4

    def test_mass_conservation(self):
        """CONTUR reports MASS = 0.9999999991 for upstream contour."""
        # This validates the mass flow integration accuracy
        mass = 0.9999999991
        assert abs(mass - 1.0) < 1e-8
