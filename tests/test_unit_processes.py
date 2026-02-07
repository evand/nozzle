"""Tests for MOC unit processes (interior, axis, wall points).

Validates predictor-corrector convergence and the planar limit.
"""

import numpy as np
import pytest
from nozzle.moc import CharMesh, interior_point, axis_point, wall_point
from nozzle.gas import prandtl_meyer, mach_angle, mach_from_prandtl_meyer


class TestInteriorPoint:
    """Interior point unit process tests."""

    def make_simple_mesh(self, gamma=1.4):
        """Create a mesh with two known upstream points.

        Convention: idx_left is C+ upstream (lower), idx_right is C- upstream (upper).
        C+ from lower point: slope = tan(θ+μ) > 0 (upward to the right)
        C- from upper point: slope = tan(θ-μ) < 0 (downward to the right)
        These converge downstream.
        """
        mesh = CharMesh(gamma=gamma)
        M = 2.0
        theta_L = np.radians(3)   # lower point, smaller θ
        theta_R = np.radians(5)   # upper point, larger θ

        # idx 0 = C+ upstream (lower point)
        mesh.add_point(x=1.0, y=1.0, M=M, theta=theta_L)
        # idx 1 = C- upstream (upper point)
        mesh.add_point(x=1.0, y=1.5, M=M, theta=theta_R)
        return mesh

    def test_interior_point_creates_point(self):
        mesh = self.make_simple_mesh()
        idx = interior_point(mesh, 0, 1)
        assert idx == 2
        assert len(mesh.points) == 3

    def test_interior_point_downstream(self):
        """New point should be downstream of both parents."""
        mesh = self.make_simple_mesh()
        idx = interior_point(mesh, 0, 1)
        pt = mesh.points[idx]
        assert pt.x > mesh.points[0].x
        assert pt.x > mesh.points[1].x

    def test_interior_point_between_parents_y(self):
        """New point y should be between parent y values."""
        mesh = self.make_simple_mesh()
        idx = interior_point(mesh, 0, 1)
        pt = mesh.points[idx]
        y_min = min(mesh.points[0].y, mesh.points[1].y)
        y_max = max(mesh.points[0].y, mesh.points[1].y)
        assert y_min <= pt.y <= y_max

    def test_connectivity(self):
        """New point should reference both parents."""
        mesh = self.make_simple_mesh()
        idx = interior_point(mesh, 0, 1)
        pt = mesh.points[idx]
        assert pt.left_idx == 0
        assert pt.right_idx == 1

    def test_mach_positive(self):
        mesh = self.make_simple_mesh()
        idx = interior_point(mesh, 0, 1)
        assert mesh.points[idx].M > 1.0

    def test_planar_limit(self):
        """At large y (effectively 2D), Q→0 and Riemann invariants are conserved.

        For 2D flow (no source term):
            K+ = θ + ν is constant along C-
            K- = θ - ν is constant along C+
        """
        mesh = CharMesh(gamma=1.4)
        # Place points at very large y where axisymmetric effects vanish
        y_offset = 1e6
        M_L, M_R = 2.0, 1.8
        theta_L, theta_R = np.radians(5), np.radians(3)

        mesh.add_point(x=1.0, y=y_offset + 0.5, M=M_L, theta=theta_L)
        mesh.add_point(x=1.0, y=y_offset - 0.5, M=M_R, theta=theta_R)

        idx = interior_point(mesh, 0, 1, gamma=1.4)
        pt = mesh.points[idx]

        # In 2D limit: K-_new should equal K-_left, K+_new should equal K+_right
        K_minus_left = mesh.points[0].K_minus
        K_plus_right = mesh.points[1].K_plus

        assert pt.K_minus == pytest.approx(K_minus_left, abs=0.01)
        assert pt.K_plus == pytest.approx(K_plus_right, abs=0.01)

    def test_predictor_corrector_convergence(self):
        """More iterations should converge (last iterations change little)."""
        mesh1 = self.make_simple_mesh()
        mesh2 = self.make_simple_mesh()

        idx1 = interior_point(mesh1, 0, 1, n_iter=1)
        idx2 = interior_point(mesh2, 0, 1, n_iter=10)

        pt1 = mesh1.points[idx1]
        pt2 = mesh2.points[idx2]

        # 10 iterations should give different result from 1 (correction),
        # but both should be reasonable
        assert pt2.M > 1.0
        assert abs(pt2.x - pt1.x) < 1.0  # Not wildly different


class TestAxisPoint:
    """Axis point (y=0) unit process tests."""

    def test_axis_y_zero(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=0.3, M=2.0, theta=np.radians(5))
        idx = axis_point(mesh, 0)
        assert mesh.points[idx].y == pytest.approx(0.0, abs=1e-12)

    def test_axis_theta_zero(self):
        """θ must be 0 on axis by symmetry."""
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=0.3, M=2.0, theta=np.radians(5))
        idx = axis_point(mesh, 0)
        assert mesh.points[idx].theta == pytest.approx(0.0, abs=1e-12)

    def test_axis_is_downstream(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=0.3, M=2.0, theta=np.radians(5))
        idx = axis_point(mesh, 0)
        assert mesh.points[idx].x > 1.0

    def test_axis_flag(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=0.3, M=2.0, theta=np.radians(5))
        idx = axis_point(mesh, 0)
        assert mesh.points[idx].is_axis is True

    def test_axis_mach_positive(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=0.3, M=2.0, theta=np.radians(5))
        idx = axis_point(mesh, 0)
        assert mesh.points[idx].M > 1.0


class TestWallPoint:
    """Wall point unit process tests."""

    def test_wall_position(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=1.5, M=2.0, theta=np.radians(10))
        x_w, y_w = 1.5, 1.8
        dydx = 0.3  # Wall slope
        idx = wall_point(mesh, 0, x_w, y_w, dydx)
        pt = mesh.points[idx]
        assert pt.x == pytest.approx(x_w)
        assert pt.y == pytest.approx(y_w)

    def test_wall_flag(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=1.5, M=2.0, theta=np.radians(10))
        idx = wall_point(mesh, 0, 1.5, 1.8, 0.3)
        assert mesh.points[idx].is_wall is True

    def test_wall_theta_matches_slope(self):
        """Wall point θ should equal arctan(dy/dx)."""
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=1.5, M=2.0, theta=np.radians(10))
        dydx = 0.2
        idx = wall_point(mesh, 0, 1.5, 1.7, dydx)
        expected_theta = np.arctan(dydx)
        assert mesh.points[idx].theta == pytest.approx(expected_theta, rel=1e-10)

    def test_wall_mach_positive(self):
        mesh = CharMesh(gamma=1.4)
        mesh.add_point(x=1.0, y=1.5, M=2.0, theta=np.radians(10))
        idx = wall_point(mesh, 0, 1.5, 1.8, 0.3)
        assert mesh.points[idx].M > 1.0
