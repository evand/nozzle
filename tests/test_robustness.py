"""Robustness and property tests across the design space.

Tests that the solver handles high Mach numbers, low area ratios,
and produces physically consistent results.
"""

import numpy as np
import pytest
from nozzle.moc import design_mln, CharMesh
from nozzle.contours import (
    minimum_length_nozzle,
    rao_parabolic_nozzle,
    conical_nozzle,
)
from nozzle.gas import (
    area_mach_ratio,
    thrust_coefficient_ideal,
    mach_from_area_ratio,
    prandtl_meyer,
)
from nozzle.analysis import moc_performance, conical_performance, rao_performance


# ---------------------------------------------------------------------------
# Helpers for isolating the real MOC mesh from synthetic exit points
# ---------------------------------------------------------------------------

def _real_wall_points(mesh):
    """Wall points from the MOC computation, excluding synthetic exit.

    design_mln appends one synthetic wall point at (x_exit, y_exit)
    with M=M_exit, θ=0.  It's the only wall point with theta==0 and
    is always last in x-order.  Everything before it is from the
    actual characteristic-wall intersection.
    """
    pts = [p for p in mesh.points if p.is_wall]
    pts.sort(key=lambda p: p.x)
    # Drop the synthetic exit (last point, has θ=0 exactly)
    if len(pts) > 1 and pts[-1].theta == 0.0:
        pts = pts[:-1]
    return pts


def _real_axis_points(mesh):
    """Axis points from the MOC computation, excluding synthetic exit.

    Real axis points are produced by the axis_point unit process and
    have right_idx set (the C- parent above).  Synthetic exit-plane
    axis points are injected with right_idx=None.
    """
    return [p for p in mesh.points
            if p.is_axis and p.right_idx is not None]


def _real_mesh(mesh):
    """Return a new CharMesh containing only computed points (no synthetics).

    Synthetic exit-plane points have: left_idx=None, right_idx=None,
    is_wall=False, and x equal to the maximum x in the mesh.
    The single synthetic wall exit point has theta=0 and is_wall=True.
    """
    x_exit = max(p.x for p in mesh.points)
    real = CharMesh(gamma=mesh.gamma)
    for p in mesh.points:
        is_synth_exit_wall = (
            p.is_wall and abs(p.x - x_exit) < 1e-10 and p.theta == 0.0
            and p.left_idx is not None
        )
        is_synth_exit_interior = (
            abs(p.x - x_exit) < 1e-10
            and p.left_idx is None and p.right_idx is None
        )
        if is_synth_exit_wall or is_synth_exit_interior:
            continue
        real.points.append(p)
    return real


# ---------------------------------------------------------------------------
# MLN physics invariants
#
# These define what a correct axisymmetric MLN must produce.
# They are derived from first principles and published references,
# independent of the implementation.  Several FAIL against the
# current solver, which uses a 2D-planar expansion fan instead
# of the correct axisymmetric approach with a transonic kernel.
# ---------------------------------------------------------------------------

class TestMLNPhysicsInvariants:
    """Physics invariants for a correct axisymmetric minimum-length nozzle.

    An MLN is designed so the exit flow is uniform and parallel at a
    specified Mach number.  The wall contour has two sections:

      1. Expansion: wall angle increases from 0 to θ_max = ν(M_exit)/2
         (Anderson MCF Eq. 11.33).
      2. Straightening: wall absorbs incoming C- characteristics,
         angle decreases from θ_max back to 0 (parallel exit flow).

    The tests below check properties that follow from these requirements
    and from basic conservation laws.

    References
    ----------
    [1] Anderson, *Modern Compressible Flow*, 3rd ed., §11.11
    [2] Zucrow & Hoffman, *Gas Dynamics* Vol. 2, Ch. 11-12
    [3] Hall, *QJMAM* 15(4), 1962 — transonic kernel
    [4] Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed., §3.4
    """

    N_CHARS = 30  # mesh resolution for all tests

    # ------------------------------------------------------------------
    # 1. Axis Mach number must increase through the nozzle
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_axis_mach_reaches_exit(self, M_exit):
        """Axis Mach should approach M_exit at the nozzle exit.

        In a correct MLN the axis Mach increases from near-sonic at
        the throat to M_exit at the exit plane.  Quantitatively, the
        rightmost real axis point should reach ≥80% of ν(M_exit) in
        Prandtl-Meyer space.  (We compare ν rather than M because ν
        is the natural expansion variable in MOC.)

        Physics: Zucrow & Hoffman §11.5.3 — the axis point unit
        process propagates K+ along C- to the symmetry line.  Each
        successive axis point must have higher M than the last.
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        real_axis = _real_axis_points(mesh)
        assert len(real_axis) >= 2, "Need at least 2 real axis points"

        M_max_axis = max(p.M for p in real_axis)
        nu_max = prandtl_meyer(M_max_axis, 1.4)
        nu_exit = prandtl_meyer(M_exit, 1.4)
        assert nu_max > 0.8 * nu_exit, (
            f"Best axis ν = {np.degrees(nu_max):.1f}° but need ≥80% of "
            f"ν_exit = {np.degrees(nu_exit):.1f}° "
            f"(axis M = {M_max_axis:.2f}, target M = {M_exit})"
        )

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_axis_mach_monotonic(self, M_exit):
        """Axis Mach should increase monotonically downstream.

        In a diverging supersonic nozzle the Mach number on the
        symmetry axis can only increase (the cross-sectional area
        is increasing).  Non-monotonic axis M indicates the source
        term is overwhelming the expansion — a sign of mesh trouble.

        Physics: Anderson MCF §3.4 — area-Mach relation in a
        diverging duct.
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        real_axis = _real_axis_points(mesh)
        real_axis.sort(key=lambda p: p.x)
        Ms = [p.M for p in real_axis]
        for i in range(1, len(Ms)):
            assert Ms[i] >= Ms[i - 1] - 1e-6, (
                f"Axis M decreased: M[{i-1}]={Ms[i-1]:.4f} → "
                f"M[{i}]={Ms[i]:.4f} at x={real_axis[i].x:.4f}"
            )

    # ------------------------------------------------------------------
    # 2. Wall contour shape
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_wall_theta_spans_full_range(self, M_exit):
        """Wall angle should span from θ_max down to near 0.

        The MLN wall starts at the throat with slope θ_max = ν(M)/2
        and must straighten to θ ≈ 0 at the exit so the exit flow
        is parallel.

        Ref: Anderson MCF §11.11 — "The last wall point has θ = 0,
        corresponding to uniform, parallel exit flow."
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        wall = _real_wall_points(mesh)
        thetas_deg = [np.degrees(p.theta) for p in wall]
        theta_max_expected = np.degrees(prandtl_meyer(M_exit, 1.4) / 2)

        # The maximum θ should be near θ_max
        assert max(thetas_deg) == pytest.approx(theta_max_expected, abs=2.0), (
            f"Max wall θ = {max(thetas_deg):.1f}° vs "
            f"expected θ_max = {theta_max_expected:.1f}°"
        )
        # The minimum θ should approach 0
        assert min(thetas_deg) < 3.0, (
            f"Min wall θ = {min(thetas_deg):.1f}° — should approach 0° "
            f"(parallel exit flow)"
        )

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_wall_y_monotonic(self, M_exit):
        """Wall y must increase monotonically (nozzle diverges).

        A diverging supersonic nozzle expands outward.  The wall
        radius y should increase from 1.0 (throat) to √(A/A*) (exit).
        Any y decrease means the wall is converging — physically wrong
        for the divergent section.

        Ref: basic nozzle geometry; Sutton & Biblarz Fig. 3-9.
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        x, y = mesh.get_wall_contour()
        dy = np.diff(y)
        violations = np.where(dy < -1e-10)[0]
        assert len(violations) == 0, (
            f"Wall y decreases at {len(violations)} points; "
            f"worst: dy={dy[violations[0]]:.6f} at "
            f"x={x[violations[0]]:.4f}"
        )

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_wall_y_reaches_exit_naturally(self, M_exit):
        """The real wall contour should reach ≥90% of the exit radius.

        Without synthetic exit points, the wall should expand to
        y ≈ √(A/A*) through the actual MOC construction.  If the
        wall only reaches y ≈ 1.2 regardless of M_exit, the mesh
        is not producing real expansion.

        Ref: continuity for axisymmetric flow — exit area must equal
        A/A* times the throat area.
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        wall = _real_wall_points(mesh)
        y_max = max(p.y for p in wall)
        y_exit = np.sqrt(area_mach_ratio(M_exit, 1.4))
        assert y_max > 0.9 * y_exit, (
            f"Real wall y_max = {y_max:.3f}, need ≥90% of "
            f"y_exit = {y_exit:.3f} ({y_max / y_exit:.0%})"
        )

    # ------------------------------------------------------------------
    # 3. Nozzle length
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_nozzle_length_reasonable(self, M_exit):
        """Nozzle length should be a reasonable fraction of L_conical_15°.

        The MLN is typically comparable in length to a conical nozzle
        with the same exit area.  As a coarse bound: the MLN should
        be at least 40% of L_conical_15° (shorter than a Rao 60% bell
        makes no physical sense).

        L_conical_15° = (y_exit − 1) / tan(15°) is the standard
        reference length for bell nozzle sizing.

        Ref: Sutton & Biblarz §3.4, Table 3-3.
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        wall = _real_wall_points(mesh)
        L = max(p.x for p in wall) - min(p.x for p in wall)

        y_exit = np.sqrt(area_mach_ratio(M_exit, 1.4))
        L_conical = (y_exit - 1.0) / np.tan(np.radians(15))
        ratio = L / L_conical
        assert ratio > 0.4, (
            f"Nozzle L/r* = {L:.3f}, only {ratio:.0%} of "
            f"L_conical_15° = {L_conical:.3f}"
        )

    # ------------------------------------------------------------------
    # 4. Cf from real mesh (not synthetic points)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_cf_from_real_mesh(self, M_exit):
        """Cf from actual MOC points should match 1D ideal to within 5%.

        The MLN guarantees uniform parallel exit flow, so Cf should
        equal the 1D ideal value.  If we only see Cf ≈ ideal when
        using synthetic exit points, the mesh itself is wrong.

        We allow 5% tolerance because the real mesh exit plane may
        not be perfectly aligned (the tolerance accounts for mesh
        resolution, not physics error).

        Ref: Sutton & Biblarz Eq. 3-30; Anderson MCF Eq. 12.30.
        """
        mesh = design_mln(M_exit, n_chars=self.N_CHARS, gamma=1.4)
        real = _real_mesh(mesh)

        result = moc_performance(real, gamma=1.4)
        Cf_ideal = thrust_coefficient_ideal(M_exit, gamma=1.4)
        assert result['Cf'] == pytest.approx(Cf_ideal, rel=0.05), (
            f"Real-mesh Cf = {result['Cf']:.4f} vs "
            f"ideal = {Cf_ideal:.4f} "
            f"({result['Cf'] / Cf_ideal:.1%})"
        )

    # ------------------------------------------------------------------
    # 5. Mesh convergence (coarse sanity check)
    # ------------------------------------------------------------------

    def test_wall_converges_with_resolution(self):
        """Doubling n_chars should not drastically change the wall shape.

        The wall y_exit from n=15 and n=30 should agree to within 10%.
        This catches algorithms that produce qualitatively different
        results at different resolutions.
        """
        M_exit = 2.0
        wall_15 = _real_wall_points(
            design_mln(M_exit, n_chars=15, gamma=1.4)
        )
        wall_30 = _real_wall_points(
            design_mln(M_exit, n_chars=30, gamma=1.4)
        )
        y_15 = max(p.y for p in wall_15)
        y_30 = max(p.y for p in wall_30)
        assert y_15 == pytest.approx(y_30, rel=0.10), (
            f"Wall y_max: n=15 → {y_15:.3f}, n=30 → {y_30:.3f}"
        )


# ---------------------------------------------------------------------------
# MLN high-Mach robustness (should still complete without error)
# ---------------------------------------------------------------------------

class TestMLNHighMach:
    """MLN should complete without error for high exit Mach numbers."""

    @pytest.mark.parametrize("M_exit", [3.0, 3.2, 4.0, 5.0, 7.0])
    def test_completes(self, M_exit):
        mesh = design_mln(M_exit, n_chars=30, gamma=1.4)
        x, y = mesh.get_wall_contour()
        assert len(x) >= 3

    @pytest.mark.parametrize("M_exit", [3.0, 3.2, 4.0, 5.0, 7.0])
    def test_wall_x_monotonic(self, M_exit):
        mesh = design_mln(M_exit, n_chars=30, gamma=1.4)
        x, y = mesh.get_wall_contour()
        assert np.all(np.diff(x) >= -1e-10)

    @pytest.mark.parametrize("M_exit", [3.0, 3.2, 4.0, 5.0, 7.0])
    def test_exit_radius(self, M_exit):
        """Exit radius within 30% of sqrt(A/A*) — coarse mesh tolerance."""
        mesh = design_mln(M_exit, n_chars=30, gamma=1.4)
        x, y = mesh.get_wall_contour()
        y_expected = np.sqrt(area_mach_ratio(M_exit, 1.4))
        assert y[-1] == pytest.approx(y_expected, rel=0.30)


# ---------------------------------------------------------------------------
# MLN performance cross-validation (uses synthetic points — kept for now)
# ---------------------------------------------------------------------------

class TestMLNPerformanceCf:
    """MLN Cf should match 1D ideal to within 1%.

    NOTE: These tests currently pass because moc_performance integrates
    over synthetic exit-plane points injected by design_mln.  The real
    validation is TestMLNPhysicsInvariants.test_cf_from_real_mesh above.
    """

    @pytest.mark.parametrize("M_exit", [2.0, 3.0])
    def test_cf_matches_ideal(self, M_exit):
        x_wall, y_wall, mesh = minimum_length_nozzle(M_exit, n_chars=30)
        result = moc_performance(mesh, gamma=1.4)
        Cf_ideal = thrust_coefficient_ideal(M_exit, gamma=1.4)
        assert result['Cf'] == pytest.approx(Cf_ideal, rel=0.01)

    def test_performance_ordering_ar10(self):
        """At AR=10: conical Cf < rao Cf < MLN Cf."""
        ar = 10.0
        M_exit = mach_from_area_ratio(ar, gamma=1.4)

        conical = conical_performance(15, ar, gamma=1.4)
        rao = rao_performance(ar, bell_fraction=0.8, gamma=1.4)

        x_wall, y_wall, mesh = minimum_length_nozzle(M_exit, n_chars=30)
        mln = moc_performance(mesh, gamma=1.4)

        assert conical['Cf'] < rao['Cf']
        assert rao['Cf'] < mln['Cf']


# ---------------------------------------------------------------------------
# Rao low area-ratio robustness
# ---------------------------------------------------------------------------

class TestRaoLowAR:
    """Rao parabolic nozzle should produce valid contours at low AR."""

    @pytest.mark.parametrize("ar", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_x_monotonic(self, ar):
        x, y, _, _ = rao_parabolic_nozzle(ar, 0.8)
        assert np.all(np.diff(x) > 0)

    @pytest.mark.parametrize("ar", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_y_monotonic(self, ar):
        x, y, _, _ = rao_parabolic_nozzle(ar, 0.8)
        assert np.all(np.diff(y) >= 0)

    @pytest.mark.parametrize("ar", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_exit_radius(self, ar):
        """Exit y within 5% of sqrt(AR)."""
        x, y, _, _ = rao_parabolic_nozzle(ar, 0.8)
        assert y[-1] == pytest.approx(np.sqrt(ar), rel=0.05)

    @pytest.mark.parametrize("ar", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_curvature_sign_changes(self, ar):
        """Bell contour should have a small number of curvature sign changes (1-4)."""
        x, y, _, _ = rao_parabolic_nozzle(ar, 0.8, n_points=500)
        # Compute second derivative (curvature proxy)
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        # Count sign changes in curvature
        sign_changes = np.sum(np.abs(np.diff(np.sign(d2y))) > 0)
        assert sign_changes <= 4

    def test_rejects_low_ar(self):
        with pytest.raises(ValueError, match="too small"):
            rao_parabolic_nozzle(1.2, 0.8)


# ---------------------------------------------------------------------------
# Contour property checks across all types
# ---------------------------------------------------------------------------

class TestContourProperties:
    """Universal properties that all contour types should satisfy."""

    def test_conical_starts_at_throat(self):
        x, y = conical_nozzle(15, 10)
        assert x[0] == pytest.approx(0.0, abs=1e-10)
        assert y[0] == pytest.approx(1.0, abs=1e-10)

    def test_conical_exit_radius(self):
        ar = 10.0
        x, y = conical_nozzle(15, ar)
        assert y[-1] == pytest.approx(np.sqrt(ar), rel=0.01)

    def test_rao_starts_at_throat(self):
        x, y, _, _ = rao_parabolic_nozzle(4.0, 0.8)
        assert x[0] == pytest.approx(0.0, abs=0.05)
        assert y[0] == pytest.approx(1.0, abs=0.05)

    def test_rao_exit_radius(self):
        ar = 10.0
        x, y, _, _ = rao_parabolic_nozzle(ar, 0.8)
        assert y[-1] == pytest.approx(np.sqrt(ar), rel=0.01)

    def test_mln_starts_at_throat(self):
        x, y, _ = minimum_length_nozzle(2.0, n_chars=20)
        assert x[0] == pytest.approx(0.0, abs=0.05)
        assert y[0] == pytest.approx(1.0, abs=0.05)

    def test_mln_exit_radius(self):
        ar = area_mach_ratio(2.0, 1.4)
        x, y, _ = minimum_length_nozzle(2.0, n_chars=20)
        assert y[-1] == pytest.approx(np.sqrt(ar), rel=0.05)

    def test_mln_x_starts_at_zero(self):
        x, y, _ = minimum_length_nozzle(2.0, n_chars=20)
        assert x[0] == pytest.approx(0.0, abs=0.05)
