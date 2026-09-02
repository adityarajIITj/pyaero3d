"""
PyAero3D - Spatial Quaternion & Direction Cosine Matrix (DCM) Mathematics.
Eliminates Euler angle singularities (Gimbal Lock) across all 3D rotations.
"""

from typing import Tuple
import numpy as np


class SpatialQuaternion:
    """
    Pure Spatial Unit Quaternion Operations [w, x, y, z].
    """

    @staticmethod
    def identity() -> np.ndarray:
        """Returns identity quaternion [1, 0, 0, 0]."""
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Safely normalizes quaternion to unit length."""
        norm_sq = float(np.dot(q, q))
        if norm_sq < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / np.sqrt(norm_sq)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """Constructs quaternion from unit axis and rotation angle in radians."""
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            return SpatialQuaternion.identity()
        u = axis / axis_norm
        half = 0.5 * angle_rad
        s = np.sin(half)
        return np.array([np.cos(half), u[0] * s, u[1] * s, u[2] * s], dtype=np.float64)

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product of two quaternions: q_out = q1 * q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rodrigues formula quaternion vector rotation: v' = q * [0, v] * q^-1.
        """
        w = q[0]
        u = q[1:4]
        # v' = v + 2*w*(u x v) + 2*(u x (u x v))
        uv = np.cross(u, v)
        uuv = np.cross(u, uv)
        return v + 2.0 * (w * uv + uuv)

    @staticmethod
    def to_dcm(q: np.ndarray) -> np.ndarray:
        """
        Converts unit quaternion to 3x3 Direction Cosine Matrix (DCM / Rotation Matrix).
        """
        q = SpatialQuaternion.normalize(q)
        w, x, y, z = q

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float64)
        return R

    @staticmethod
    def integrate_quaternion(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """
        First-order symplectic quaternion time-step integration:
        q(t + dt) = q(t) + 0.5 * dt * q(t) * [0, omega]
        """
        omega_norm = float(np.linalg.norm(omega_body))
        if omega_norm < 1e-12:
            return SpatialQuaternion.normalize(q)

        # Exponential map for exact SO(3) rotational integration
        half_angle = 0.5 * omega_norm * dt
        axis = omega_body / omega_norm
        s = np.sin(half_angle)
        dq = np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)

        return SpatialQuaternion.normalize(SpatialQuaternion.multiply(q, dq))
