"""Tests for Sivells CONTUR axisymmetric mode (ie=1).

Validates sivells_axial and sivells_perfc against CONTUR F90 output
for the Mach 4 axisymmetric test case (jd=0 -> ie=1).

Reference: CONTUR F90 compiled with gfortran, run with
docs/references/external_codes/contur/sivells/input.txt (jd=0).
Input: gamma=1.4, etad=8.67, rc=6.0, bmach=3.0, cmach=4.0,
       ie=1, m=41, n=21, nx=13.
"""

import numpy as np
import pytest
from nozzle.sivells import sivells_axial, sivells_perfc


class TestAxialAxisymmetric:
    """Test sivells_axial with ie=1 against CONTUR axisymmetric output."""

    @pytest.fixture
    def axi_result(self):
        return sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.0, cmach=4.0,
            ie=1, n_char=41, n_axis=21, nx=13
        )

    def test_throat_velocity(self, axi_result):
        """Throat center velocity wo differs from planar (0.975 vs 0.964)."""
        r = axi_result
        assert abs(r['wo'] - 0.96385164) < 1e-6

    def test_throat_mass_ratio(self, axi_result):
        """Mass ratio tk = rmass."""
        r = axi_result
        assert abs(r['tk'] - 0.99971357) < 1e-6

    def test_throat_yo(self, axi_result):
        """Throat half-height yo."""
        r = axi_result
        assert abs(r['yo'] - 0.15121903) < 1e-6

    def test_throat_characteristic_wall(self, axi_result):
        """Velocity at wall on throat characteristic."""
        r = axi_result
        assert abs(r['wwo'] - 1.0388387) < 1e-5
        assert abs(r['wwop'] - 2.59666557) < 1e-5

    def test_emach_fmach(self, axi_result):
        """Exit and downstream Mach numbers."""
        r = axi_result
        assert abs(r['emach'] - 1.66015) < 1e-4
        assert abs(r['fmach'] - 3.0821543) < 1e-4

    def test_inflection_velocity(self, axi_result):
        """Velocity and derivatives at inflection point."""
        r = axi_result
        assert abs(r['wi'] - 1.06914514) < 1e-5
        assert abs(r['wip'] - 2.24012223) < 1e-4
        assert abs(r['wipp'] - (-0.89085294)) < 1e-4

    def test_exit_velocity(self, axi_result):
        """Velocity and derivatives at exit point."""
        r = axi_result
        assert abs(r['we'] - 1.46016505) < 1e-5
        assert abs(r['wep'] - 1.45785766) < 1e-4
        assert abs(r['wepp'] - (-6.9097416)) < 1e-3

    def test_geometry_positions(self, axi_result):
        """Key x-positions along the axis."""
        r = axi_result
        assert abs(r['xoi'] - 0.04673408) < 1e-5
        assert abs(r['xi'] - 0.94011759) < 1e-5
        assert abs(r['xo'] - 0.89338351) < 1e-5
        assert abs(r['xie'] - 0.20056538) < 1e-5
        assert abs(r['xe'] - 1.14068296) < 1e-5

    def test_polynomial_coefficients(self, axi_result):
        """3rd-degree velocity distribution coefficients."""
        c = axi_result['c']
        assert abs(c[0] - 1.0691451) < 1e-5
        assert abs(c[1] - 0.44929096) < 1e-5
        assert abs(c[2] - (-1.7917934e-02)) < 1e-7
        assert abs(c[3] - (-4.0353099e-02)) < 1e-6
        assert c[4] == 0.0
        assert c[5] == 0.0

    def test_axis_endpoints(self, axi_result):
        """Axis array at exit and inflection."""
        axis = axi_result['axis']
        # k=0 is exit
        assert abs(axis[0, 0] - 1.14068) < 1e-4
        assert abs(axis[2, 0] - 1.660154) < 1e-4
        # k=20 is inflection
        assert abs(axis[0, 20] - 0.94012) < 1e-4
        assert abs(axis[2, 20] - 1.084778) < 1e-4

    def test_axis_mach_monotonic(self, axi_result):
        """Mach decreases from exit to inflection."""
        mach = axi_result['axis'][2, :]
        for k in range(1, 21):
            assert mach[k] < mach[k - 1]

    def test_axis_all_supersonic(self, axi_result):
        """All axis points supersonic."""
        for k in range(21):
            assert axi_result['axis'][2, k] > 1.0

    def test_differs_from_planar(self, axi_result):
        """Axisymmetric wo != planar wo (0.964 vs 0.975)."""
        planar = sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.2, cmach=4.0,
            ie=0, n_char=41, n_axis=21, nx=13
        )
        # wo is significantly different (axisymmetric source term matters)
        assert abs(axi_result['wo'] - planar['wo']) > 0.01


class TestPerfcAxisymmetric:
    """Test sivells_perfc with ie=1 against CONTUR axisymmetric wall output.

    Reference: 41 wall points from CONTUR F90 axisymmetric Mach 4 output.
    """

    REF_X = np.array([
        8.9338351e-01, 8.9817803e-01, 9.0313629e-01, 9.0825111e-01,
        9.1353801e-01, 9.1899646e-01, 9.2465957e-01, 9.3053668e-01,
        9.3667828e-01, 9.4311748e-01, 9.4989369e-01, 9.5707300e-01,
        9.6470038e-01, 9.7283636e-01, 9.8154483e-01, 9.9089189e-01,
        1.0009363e+00, 1.0117627e+00, 1.0234319e+00, 1.0360337e+00,
        1.0496494e+00, 1.0586537e+00, 1.0715645e+00, 1.0863627e+00,
        1.1024792e+00, 1.1196438e+00, 1.1376982e+00, 1.1565561e+00,
        1.1761470e+00, 1.1964348e+00, 1.2173753e+00, 1.2389456e+00,
        1.2611122e+00, 1.2838486e+00, 1.3071231e+00, 1.3309005e+00,
        1.3551440e+00, 1.3798149e+00, 1.4048689e+00, 1.4302636e+00,
        1.4559664e+00])

    REF_Y = np.array([
        1.5121903e-01, 1.5123170e-01, 1.5127139e-01, 1.5134059e-01,
        1.5144220e-01, 1.5157894e-01, 1.5175473e-01, 1.5197313e-01,
        1.5223987e-01, 1.5256093e-01, 1.5294304e-01, 1.5339554e-01,
        1.5392737e-01, 1.5454907e-01, 1.5527196e-01, 1.5610837e-01,
        1.5707028e-01, 1.5817310e-01, 1.5943060e-01, 1.6085912e-01,
        1.6247489e-01, 1.6357969e-01, 1.6520817e-01, 1.6713198e-01,
        1.6928744e-01, 1.7164233e-01, 1.7417403e-01, 1.7686825e-01,
        1.7971142e-01, 1.8269284e-01, 1.8580201e-01, 1.8903000e-01,
        1.9236687e-01, 1.9580533e-01, 1.9933644e-01, 2.0295157e-01,
        2.0664295e-01, 2.1040276e-01, 2.1422257e-01, 2.1809487e-01,
        2.2201404e-01])

    REF_M = np.array([
        1.0471638e+00, 1.0626027e+00, 1.0786293e+00, 1.0952300e+00,
        1.1124932e+00, 1.1303552e+00, 1.1489592e+00, 1.1682305e+00,
        1.1883589e+00, 1.2093556e+00, 1.2312390e+00, 1.2541190e+00,
        1.2779433e+00, 1.3026785e+00, 1.3283469e+00, 1.3550399e+00,
        1.3829205e+00, 1.4120873e+00, 1.4427244e+00, 1.4749409e+00,
        1.5088820e+00, 1.5308817e+00, 1.5618623e+00, 1.5966017e+00,
        1.6335898e+00, 1.6720611e+00, 1.7116110e+00, 1.7519140e+00,
        1.7928312e+00, 1.8341515e+00, 1.8758124e+00, 1.9176407e+00,
        1.9595949e+00, 2.0015440e+00, 2.0434024e+00, 2.0850968e+00,
        2.1265044e+00, 2.1675528e+00, 2.2081710e+00, 2.2482849e+00,
        2.2878437e+00])

    REF_ANGLE = np.array([
        0.0, 3.0245925e-01, 6.1441645e-01, 9.3538991e-01,
        1.2653979e+00, 1.6033225e+00, 1.9498158e+00, 2.3036495e+00,
        2.6663346e+00, 3.0362174e+00, 3.4115784e+00, 3.7917793e+00,
        4.1730231e+00, 4.5511485e+00, 4.9228566e+00, 5.2854769e+00,
        5.6368754e+00, 5.9772441e+00, 6.3029481e+00, 6.6117213e+00,
        6.9045359e+00, 7.0747554e+00, 7.2906021e+00, 7.5078996e+00,
        7.7117619e+00, 7.8949827e+00, 8.0549464e+00, 8.1929694e+00,
        8.3079571e+00, 8.4020831e+00, 8.4777598e+00, 8.5356298e+00,
        8.5800232e+00, 8.6131638e+00, 8.6358918e+00, 8.6512393e+00,
        8.6612814e+00, 8.6669028e+00, 8.6693439e+00, 8.6698209e+00,
        8.6700000e+00])

    REF_WALTAN = np.array([
        0.0, 5.2789588e-03, 1.0724001e-02, 1.6327084e-02,
        2.2088951e-02, 2.7990563e-02, 3.4043849e-02, 4.0227947e-02,
        4.6569941e-02, 5.3041650e-02, 5.9613744e-02, 6.6275817e-02,
        7.2962051e-02, 7.9600009e-02, 8.6132110e-02, 9.2511544e-02,
        9.8700684e-02, 1.0470270e-01, 1.1045311e-01, 1.1591127e-01,
        1.2109362e-01, 1.2410917e-01, 1.2793624e-01, 1.3179276e-01,
        1.3541435e-01, 1.3867221e-01, 1.4151890e-01, 1.4397695e-01,
        1.4602606e-01, 1.4770430e-01, 1.4905419e-01, 1.5008681e-01,
        1.5087917e-01, 1.5147080e-01, 1.5187660e-01, 1.5215065e-01,
        1.5232998e-01, 1.5243037e-01, 1.5247397e-01, 1.5248249e-01,
        1.5248569e-01])

    REF_SECD = np.array([
        1.1021541e+00, 1.0996321e+00, 1.0968392e+00, 1.0926953e+00,
        1.0855826e+00, 1.0751564e+00, 1.0607205e+00, 1.0426474e+00,
        1.0191646e+00, 9.8791142e-01, 9.4952199e-01, 9.0305903e-01,
        8.4722246e-01, 7.8409951e-01, 7.1749170e-01, 6.5053418e-01,
        5.8644040e-01, 5.2474031e-01, 4.6410130e-01, 4.0788671e-01,
        3.5309811e-01, 3.1909219e-01, 2.7973569e-01, 2.4342529e-01,
        2.0780700e-01, 1.7414320e-01, 1.4430635e-01, 1.1771595e-01,
        9.3849670e-02, 7.3736928e-02, 5.6290407e-02, 4.1891545e-02,
        3.0945147e-02, 2.1778656e-02, 1.4512271e-02, 9.4814587e-03,
        5.7476131e-03, 2.9135948e-03, 1.0425217e-03, 2.3058018e-04,
        0.0])

    @pytest.fixture
    def perfc_result(self):
        axial = sivells_axial(
            gamma=1.4, eta_deg=8.67, rc=6.0, bmach=3.0, cmach=4.0,
            ie=1, n_char=41, n_axis=21, nx=13
        )
        return sivells_perfc(axial, gamma=1.4, ie=1)

    def test_n_wall_points(self, perfc_result):
        assert perfc_result['n_wall'] == 41

    def test_mass_flow(self, perfc_result):
        assert abs(perfc_result['mass'] - 1.0) < 1e-6

    def test_wall_x_coordinates(self, perfc_result):
        np.testing.assert_allclose(
            perfc_result['wax'], self.REF_X, atol=1e-6,
            err_msg="Wall x-coords disagree with CONTUR (ie=1)"
        )

    def test_wall_y_coordinates(self, perfc_result):
        np.testing.assert_allclose(
            perfc_result['way'], self.REF_Y, atol=1e-4,
            err_msg="Wall y-coords disagree with CONTUR (ie=1)"
        )

    def test_wall_mach(self, perfc_result):
        np.testing.assert_allclose(
            perfc_result['wmn'], self.REF_M, atol=1e-6,
            err_msg="Wall Mach disagrees with CONTUR (ie=1)"
        )

    def test_wall_angle(self, perfc_result):
        # Axisymmetric source term makes ofeld iterate harder; tolerance
        # slightly looser than planar (0.02 vs 0.01 deg).
        np.testing.assert_allclose(
            perfc_result['wan'], self.REF_ANGLE, atol=0.02,
            err_msg="Wall flow angles disagree with CONTUR (ie=1)"
        )

    def test_wall_waltan(self, perfc_result):
        # Slightly looser than planar (3e-4 vs 2e-4) due to axi source term.
        np.testing.assert_allclose(
            perfc_result['waltan'], self.REF_WALTAN, atol=3e-4,
            err_msg="Wall tan(theta) disagrees with CONTUR (ie=1)"
        )

    def test_wall_secd(self, perfc_result):
        np.testing.assert_allclose(
            perfc_result['secd'], self.REF_SECD, atol=0.01,
            err_msg="Second derivative disagrees with CONTUR (ie=1)"
        )

    def test_wall_angle_reaches_eta(self, perfc_result):
        """Last wall point should be at the inflection angle."""
        assert abs(perfc_result['wan'][-1] - 8.67) < 1e-6

    def test_wall_mach_monotonic(self, perfc_result):
        """Wall Mach should increase monotonically."""
        m = perfc_result['wmn']
        for i in range(1, len(m)):
            assert m[i] > m[i - 1], (
                f"Non-monotonic Mach at i={i}: M[{i-1}]={m[i-1]:.6f}, "
                f"M[{i}]={m[i]:.6f}"
            )

    def test_wall_y_monotonic(self, perfc_result):
        """Wall y should increase monotonically (expanding nozzle)."""
        y = perfc_result['way']
        for i in range(1, len(y)):
            assert y[i] >= y[i - 1]


class TestSivellsNozzleAxisymmetric:
    """Test the high-level sivells_nozzle wrapper with ie=1."""

    def test_contour_returns_array(self):
        """sivells_nozzle with ie=1 returns valid contour."""
        from nozzle.contours import sivells_nozzle
        x, y = sivells_nozzle(4.0, gamma=1.4, rc=6.0,
                              inflection_angle_deg=8.67, ie=1,
                              n_char=41, n_axis=21, nx=13)
        assert len(x) == 41
        assert len(y) == 41
        # Throat at x=0, y=1 in r*-normalized coords
        assert abs(x[0]) < 0.5
        assert abs(y[0] - 1.0) < 0.05

    def test_contour_monotonic_x(self):
        """x increases monotonically."""
        from nozzle.contours import sivells_nozzle
        x, y = sivells_nozzle(4.0, gamma=1.4, rc=6.0,
                              inflection_angle_deg=8.67, ie=1,
                              n_char=41, n_axis=21, nx=13)
        for i in range(1, len(x)):
            assert x[i] > x[i - 1]

    def test_contour_monotonic_y(self):
        """y increases monotonically (diverging nozzle)."""
        from nozzle.contours import sivells_nozzle
        x, y = sivells_nozzle(4.0, gamma=1.4, rc=6.0,
                              inflection_angle_deg=8.67, ie=1,
                              n_char=41, n_axis=21, nx=13)
        for i in range(1, len(y)):
            assert y[i] >= y[i - 1]

    def test_contour_differs_from_planar(self):
        """Axisymmetric contour is different from planar."""
        from nozzle.contours import sivells_nozzle
        x_axi, y_axi = sivells_nozzle(4.0, gamma=1.4, rc=6.0,
                                       inflection_angle_deg=8.67,
                                       ie=1, n_char=41, n_axis=21, nx=13)
        x_pla, y_pla = sivells_nozzle(4.0, gamma=1.4, rc=6.0,
                                       inflection_angle_deg=8.67,
                                       ie=0, n_char=41, n_axis=21, nx=13)
        # Same number of points but different coordinates
        assert len(x_axi) == len(x_pla)
        # The contours should differ significantly
        assert np.max(np.abs(x_axi - x_pla)) > 0.01
