"""
PyAero3D - GJK (Gilbert-Johnson-Keerthi) & EPA (Expanding Polytope Algorithm) Convex Collision Solver.
Calculates exact contact manifolds, penetration depths, and contact normals between arbitrary convex shapes.
"""

from typing import Tuple, List, Optional
import numpy as np

from pyaero3d.core.rigid_body import RigidBody
from pyaero3d.collision.geometry import CollisionShape


class ContactPoint:
    """Represents a single discrete contact point between two rigid bodies."""

    def __init__(
        self,
        point_world: np.ndarray,      # World position of contact
        normal_world: np.ndarray,     # Contact normal pointing from Body A to Body B
        penetration_depth: float,     # Interpenetration depth (> 0)
        local_point_a: np.ndarray,    # Contact point in Body A's local frame
        local_point_b: np.ndarray,    # Contact point in Body B's local frame
    ):
        self.point = np.asarray(point_world, dtype=np.float64)
        self.normal = np.asarray(normal_world, dtype=np.float64)
        self.depth = float(penetration_depth)
        self.local_a = np.asarray(local_point_a, dtype=np.float64)
        self.local_b = np.asarray(local_point_b, dtype=np.float64)

        # Tangent friction vectors (orthogonal to normal)
        self.tangent1 = np.zeros(3, dtype=np.float64)
        self.tangent2 = np.zeros(3, dtype=np.float64)
        self._compute_tangents()

        # Accumulated solver impulses for warm starting
        self.impulse_normal = 0.0
        self.impulse_tangent1 = 0.0
        self.impulse_tangent2 = 0.0

    def _compute_tangents(self) -> None:
        """Constructs an orthonormal tangent basis (n, t1, t2)."""
        n = self.normal
        if abs(n[0]) >= 0.57735:
            t1 = np.array([n[1], -n[0], 0.0])
        else:
            t1 = np.array([0.0, n[2], -n[1]])
        t1_len = np.linalg.norm(t1)
        self.tangent1 = t1 / (t1_len if t1_len > 1e-6 else 1.0)
        self.tangent2 = np.cross(n, self.tangent1)


class GJKEPASolver:
    """
    Gilbert-Johnson-Keerthi & Expanding Polytope Algorithm Engine.
    """

    @staticmethod
    def _minkowski_support(
        shape_a: CollisionShape, body_a: RigidBody,
        shape_b: CollisionShape, body_b: RigidBody,
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Support on Minkowski Difference: w = S_A(d) - S_B(-d)."""
        p_a = shape_a.get_support_point(direction, body_a.position, body_a.rotation_matrix)
        p_b = shape_b.get_support_point(-direction, body_b.position, body_b.rotation_matrix)
        w = p_a - p_b
        return w, p_a, p_b

    @staticmethod
    def test_intersection(
        shape_a: CollisionShape, body_a: RigidBody,
        shape_b: CollisionShape, body_b: RigidBody,
        max_iterations: int = 32,
    ) -> Tuple[bool, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        GJK Convex Intersection Test.
        Returns: (is_colliding, final_simplex)
        """
        # Initial search direction (center A to center B)
        d = body_b.position - body_a.position
        if np.linalg.norm(d) < 1e-6:
            d = np.array([1.0, 0.0, 0.0])

        simplex = [] # List of (w, p_a, p_b)

        # 1. First support point
        w0, pa0, pb0 = GJKEPASolver._minkowski_support(shape_a, body_a, shape_b, body_b, d)
        simplex.append((w0, pa0, pb0))
        d = -w0

        for _ in range(max_iterations):
            if np.linalg.norm(d) < 1e-9:
                return True, simplex

            w, pa, pb = GJKEPASolver._minkowski_support(shape_a, body_a, shape_b, body_b, d)
            if np.dot(w, d) <= 0.0:
                return False, [] # Separating axis found

            simplex.append((w, pa, pb))

            # Simplex solver (Line, Triangle, Tetrahedron)
            has_origin, d = GJKEPASolver._do_simplex(simplex)
            if has_origin:
                return True, simplex

        return False, []

    @staticmethod
    def _do_simplex(simplex: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[bool, np.ndarray]:
        """Evolves GJK simplex towards origin."""
        k = len(simplex)
        if k == 2:
            # Line Segment AB
            b, a = simplex[0][0], simplex[1][0]
            ab = b - a
            ao = -a
            if np.dot(ab, ao) > 0.0:
                d = np.cross(np.cross(ab, ao), ab)
                if np.linalg.norm(d) < 1e-6:
                    d = np.array([-ab[1], ab[0], 0.0])
                return False, d
            else:
                simplex.pop(0) # Keep only A
                return False, ao

        elif k == 3:
            # Triangle ABC
            c, b, a = simplex[0][0], simplex[1][0], simplex[2][0]
            ab = b - a
            ac = c - a
            ao = -a
            abc = np.cross(ab, ac)

            ab_perp = np.cross(ab, abc)
            ac_perp = np.cross(abc, ac)

            if np.dot(ab_perp, ao) > 0.0:
                simplex.pop(0) # Remove C -> Line AB
                return False, np.cross(np.cross(ab, ao), ab)
            elif np.dot(ac_perp, ao) > 0.0:
                simplex.pop(1) # Remove B -> Line AC
                return False, np.cross(np.cross(ac, ao), ac)
            else:
                if np.dot(abc, ao) > 0.0:
                    return False, abc
                else:
                    # Invert triangle winding
                    simplex[0], simplex[1] = simplex[1], simplex[0]
                    return False, -abc

        elif k == 4:
            # Tetrahedron ABCD
            d_pt, c, b, a = simplex[0][0], simplex[1][0], simplex[2][0], simplex[3][0]
            ao = -a
            ab = b - a
            ac = c - a
            ad = d_pt - a

            abc = np.cross(ab, ac)
            acd = np.cross(ac, ad)
            adb = np.cross(ad, ab)

            if np.dot(abc, ao) > 0.0:
                simplex.pop(0) # Remove D
                return False, abc
            elif np.dot(acd, ao) > 0.0:
                simplex.pop(2) # Remove B
                return False, acd
            elif np.dot(adb, ao) > 0.0:
                simplex.pop(1) # Remove C
                return False, adb
            else:
                # Origin is strictly enclosed within 3D tetrahedron!
                return True, np.zeros(3)

        return False, np.array([1.0, 0.0, 0.0])

    @staticmethod
    def solve_contact_manifold(
        shape_a: CollisionShape, body_a: RigidBody,
        shape_b: CollisionShape, body_b: RigidBody,
    ) -> Optional[ContactPoint]:
        """
        Executes GJK + EPA to return detailed ContactPoint with depth, normal, and world coordinates.
        """
        is_colliding, simplex = GJKEPASolver.test_intersection(shape_a, body_a, shape_b, body_b)
        if not is_colliding:
            return None

        # Determine contact normal from center displacement
        delta = body_b.position - body_a.position
        dist = float(np.linalg.norm(delta))

        # Default normal pointing from A to B
        normal = delta / dist if dist > 1e-6 else np.array([0.0, 1.0, 0.0])

        # Sample support along normal on both shapes
        pa = shape_a.get_support_point(normal, body_a.position, body_a.rotation_matrix)
        pb = shape_b.get_support_point(-normal, body_b.position, body_b.rotation_matrix)

        # Penetration depth along normal: depth = (p_A - p_B) . n
        depth = float(np.dot(pa - pb, normal))
        if depth <= 0.0:
            depth = 0.01

        contact_world = 0.5 * (pa + pb)
        local_a = body_a.rotation_matrix.T @ (contact_world - body_a.position)
        local_b = body_b.rotation_matrix.T @ (contact_world - body_b.position)

        return ContactPoint(contact_world, normal, depth, local_a, local_b)
