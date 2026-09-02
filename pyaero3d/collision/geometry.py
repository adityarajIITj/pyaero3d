"""
PyAero3D - Collision Geometry Primitives & Support Function Mapping.
Supports GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm) Convex Narrowphase.
"""

from typing import Tuple, List, Optional
import numpy as np


class CollisionShape:
    """Base class for all 3D collision geometries."""

    def get_support_point(self, direction_world: np.ndarray, body_pos: np.ndarray, body_rot: np.ndarray) -> np.ndarray:
        """
        Computes Minkowski support point S(d) = argmax_(v in Shape) (v . d) in world coordinates.
        """
        raise NotImplementedError

    def compute_aabb(self, body_pos: np.ndarray, body_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Axis-Aligned Bounding Box (AABB_min, AABB_max) in world coordinates."""
        raise NotImplementedError


class SphereShape(CollisionShape):
    """Solid Sphere Geometry."""

    def __init__(self, radius: float = 0.5):
        self.radius = float(radius)

    def get_support_point(self, direction_world: np.ndarray, body_pos: np.ndarray, body_rot: np.ndarray) -> np.ndarray:
        dir_len = np.linalg.norm(direction_world)
        if dir_len < 1e-9:
            return body_pos + np.array([self.radius, 0.0, 0.0])
        return body_pos + (direction_world / dir_len) * self.radius

    def compute_aabb(self, body_pos: np.ndarray, body_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r_vec = np.array([self.radius, self.radius, self.radius])
        return body_pos - r_vec, body_pos + r_vec


class BoxShape(CollisionShape):
    """Oriented Bounding Box (OBB) Geometry."""

    def __init__(self, half_extents: np.ndarray):
        self.half_extents = np.asarray(half_extents, dtype=np.float64)

    def get_support_point(self, direction_world: np.ndarray, body_pos: np.ndarray, body_rot: np.ndarray) -> np.ndarray:
        # Transform search direction into body local space: d_local = R^T * d_world
        d_local = body_rot.T @ direction_world
        # Extreme vertex in local space
        s_local = np.sign(d_local) * self.half_extents
        # Keep zero components at 0.0 to prevent fictitious corner lever arms
        s_local[d_local == 0.0] = 0.0
        # Transform back to world space: s_world = pos + R * s_local
        return body_pos + (body_rot @ s_local)

    def compute_aabb(self, body_pos: np.ndarray, body_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Maximum extents from world rotation matrix
        hx, hy, hz = self.half_extents
        # aabb_half = sum(|R_ij| * h_j)
        aabb_half = np.abs(body_rot) @ self.half_extents
        return body_pos - aabb_half, body_pos + aabb_half


class CapsuleShape(CollisionShape):
    """Solid Capsule Geometry (Cylinder aligned with body Y-axis + 2 Hemispheres)."""

    def __init__(self, radius: float = 0.3, half_height: float = 0.8):
        self.radius = float(radius)
        self.half_height = float(half_height)

    def get_support_point(self, direction_world: np.ndarray, body_pos: np.ndarray, body_rot: np.ndarray) -> np.ndarray:
        d_local = body_rot.T @ direction_world
        dir_len = np.linalg.norm(d_local)
        norm_d = d_local / dir_len if dir_len > 1e-9 else np.array([0.0, 1.0, 0.0])

        # Capsule segment endpoints in body frame (+Y and -Y)
        p_seg = np.array([0.0, self.half_height if d_local[1] >= 0.0 else -self.half_height, 0.0])
        s_local = p_seg + norm_d * self.radius
        return body_pos + (body_rot @ s_local)

    def compute_aabb(self, body_pos: np.ndarray, body_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        up_world = body_rot @ np.array([0.0, self.half_height, 0.0])
        aabb_half = np.abs(up_world) + self.radius
        return body_pos - aabb_half, body_pos + aabb_half


class ConvexHullShape(CollisionShape):
    """Arbitrary 3D Convex Polyhedron / Point Cloud Geometry."""

    def __init__(self, vertices: np.ndarray):
        self.vertices_local = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))

    def get_support_point(self, direction_world: np.ndarray, body_pos: np.ndarray, body_rot: np.ndarray) -> np.ndarray:
        d_local = body_rot.T @ direction_world
        # Vectorized dot product across all vertices
        dots = self.vertices_local @ d_local
        best_idx = np.argmax(dots)
        return body_pos + (body_rot @ self.vertices_local[best_idx])

    def compute_aabb(self, body_pos: np.ndarray, body_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        verts_world = body_pos + (self.vertices_local @ body_rot.T)
        return np.min(verts_world, axis=0), np.max(verts_world, axis=0)
