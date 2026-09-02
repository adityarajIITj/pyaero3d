"""
PyAero3D - General-Purpose 6-DOF Rigid Body Dynamics & Spatial Kinematics.
Full 3x3 Moment of Inertia Tensor, Newton-Euler Equations of Motion, and Spatial Quaternions.
"""

from typing import Optional, List, Tuple
import numpy as np

from pyaero3d.core.quaternion_math import SpatialQuaternion


class RigidBody:
    """
    Universal 6-DOF Rigid Body Entity for General-Purpose Physics Simulation.
    """

    def __init__(
        self,
        mass: float = 1.0,
        inertia_tensor: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
        is_static: bool = False,
        restitution: float = 0.25,
        friction_static: float = 0.65,
        friction_kinetic: float = 0.55,
        linear_damping: float = 0.01,
        angular_damping: float = 0.02,
        name: str = "RigidBody",
    ):
        self.name = name
        self.is_static = is_static

        # Mass & Inverse Mass
        if is_static or mass <= 0.0:
            self.mass = float("inf")
            self.inv_mass = 0.0
        else:
            self.mass = float(mass)
            self.inv_mass = 1.0 / self.mass

        # 3x3 Body Moment of Inertia Tensor
        if inertia_tensor is not None and not is_static:
            self.inertia_body = np.asarray(inertia_tensor, dtype=np.float64).reshape((3, 3))
            self.inv_inertia_body = np.linalg.inv(self.inertia_body)
        elif not is_static:
            # Default unit box/sphere inertia (0.4 * m * r^2 * I)
            default_i = 0.4 * self.mass * (0.5 ** 2)
            self.inertia_body = np.eye(3, dtype=np.float64) * default_i
            self.inv_inertia_body = np.eye(3, dtype=np.float64) * (1.0 / default_i)
        else:
            self.inertia_body = np.eye(3, dtype=np.float64) * 1e30
            self.inv_inertia_body = np.zeros((3, 3), dtype=np.float64)

        # Spatial State Vectors
        self.position = np.asarray(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.asarray(velocity if velocity is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.quaternion = np.asarray(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.quaternion = SpatialQuaternion.normalize(self.quaternion)
        self.angular_velocity = np.asarray(angular_velocity if angular_velocity is not None else [0.0, 0.0, 0.0], dtype=np.float64)

        # Force & Torque Accumulators
        self.force_accum = np.zeros(3, dtype=np.float64)
        self.torque_accum = np.zeros(3, dtype=np.float64)

        # Physical Material Properties
        self.restitution = float(restitution)
        self.friction_static = float(friction_static)
        self.friction_kinetic = float(friction_kinetic)
        self.linear_damping = float(linear_damping)
        self.angular_damping = float(angular_damping)

        # Collision Shape Reference (assigned externally)
        self.collision_shape = None

        # Cached World Inverse Inertia Tensor
        self.inv_inertia_world = np.zeros((3, 3), dtype=np.float64)
        self.rotation_matrix = np.eye(3, dtype=np.float64)
        self.update_world_inertia()

    def update_world_inertia(self) -> None:
        """
        Updates world-space inverse inertia tensor: I_world^-1 = R * I_body^-1 * R^T.
        """
        if self.is_static:
            self.inv_inertia_world[:] = 0.0
            self.rotation_matrix = np.eye(3, dtype=np.float64)
            return

        self.rotation_matrix = SpatialQuaternion.to_dcm(self.quaternion)
        R = self.rotation_matrix
        self.inv_inertia_world = R @ self.inv_inertia_body @ R.T

    def apply_force(self, force_vector: np.ndarray) -> None:
        """Applies a force vector at the center of mass."""
        if not self.is_static:
            self.force_accum += force_vector

    def apply_force_at_world_point(self, force_vector: np.ndarray, world_point: np.ndarray) -> None:
        """
        Applies a force at an arbitrary world position point.
        Generates both linear force and torque: tau = (p - x_cm) x F.
        """
        if self.is_static:
            return
        self.force_accum += force_vector
        r = world_point - self.position
        self.torque_accum += np.cross(r, force_vector)

    def apply_force_at_body_point(self, force_vector: np.ndarray, body_point: np.ndarray) -> None:
        """Applies a force at an offset in the body's local coordinate frame."""
        if self.is_static:
            return
        r_world = self.rotation_matrix @ body_point
        self.force_accum += force_vector
        self.torque_accum += np.cross(r_world, force_vector)

    def apply_torque(self, torque_vector: np.ndarray) -> None:
        """Applies pure torque vector to body."""
        if not self.is_static:
            self.torque_accum += torque_vector

    def apply_impulse(self, impulse_vector: np.ndarray) -> None:
        """Applies instantaneous linear impulse: delta_v = J / m."""
        if not self.is_static:
            self.velocity += impulse_vector * self.inv_mass

    def apply_impulse_at_world_point(self, impulse_vector: np.ndarray, world_point: np.ndarray) -> None:
        """
        Applies instantaneous impulse at contact point:
        delta_v = J / m
        delta_omega = I_world^-1 * (r x J)
        """
        if self.is_static:
            return
        self.velocity += impulse_vector * self.inv_mass
        r = world_point - self.position
        self.angular_velocity += self.inv_inertia_world @ np.cross(r, impulse_vector)

    def get_point_velocity(self, world_point: np.ndarray) -> np.ndarray:
        """Returns instantaneous velocity of a world point attached to body: v_p = v_cm + omega x r."""
        if self.is_static:
            return np.zeros(3, dtype=np.float64)
        r = world_point - self.position
        return self.velocity + np.cross(self.angular_velocity, r)

    def integrate_velocities(self, dt: float) -> None:
        """
        Integrates linear and angular accelerations into velocities:
        v(t+dt) = v(t) + (F / m) * dt
        omega(t+dt) = omega(t) + I_world^-1 * (tau - omega x (I_world * omega)) * dt
        """
        if self.is_static:
            self.force_accum[:] = 0.0
            self.torque_accum[:] = 0.0
            return

        # Linear acceleration
        self.velocity += (self.force_accum * self.inv_mass) * dt
        self.velocity *= np.exp(-self.linear_damping * dt)

        # Rotational Euler dynamics: I * d_omega/dt + omega x (I * omega) = tau
        # Gyroscopic precession torque: tau_gyro = omega x (I_world * omega)
        I_w = np.linalg.inv(self.inv_inertia_world + 1e-12 * np.eye(3))
        L_rot = I_w @ self.angular_velocity
        tau_gyro = np.cross(self.angular_velocity, L_rot)
        net_torque = self.torque_accum - tau_gyro

        self.angular_velocity += (self.inv_inertia_world @ net_torque) * dt
        self.angular_velocity *= np.exp(-self.angular_damping * dt)

        # Clear accumulators
        self.force_accum[:] = 0.0
        self.torque_accum[:] = 0.0

    def integrate_positions(self, dt: float) -> None:
        """
        Integrates velocities into positions and quaternion orientation:
        x(t+dt) = x(t) + v * dt
        q(t+dt) = integrate_quaternion(q, omega, dt)
        """
        if self.is_static:
            return

        self.position += self.velocity * dt
        self.quaternion = SpatialQuaternion.integrate_quaternion(self.quaternion, self.angular_velocity, dt)
        self.update_world_inertia()

    @staticmethod
    def create_box_inertia(mass: float, half_extents: np.ndarray) -> np.ndarray:
        """Solid rectangular cuboid moment of inertia tensor."""
        hx, hy, hz = half_extents
        lx, ly, lz = 2.0 * hx, 2.0 * hy, 2.0 * hz
        ixx = (1.0 / 12.0) * mass * (ly * ly + lz * lz)
        iyy = (1.0 / 12.0) * mass * (lx * lx + lz * lz)
        izz = (1.0 / 12.0) * mass * (lx * lx + ly * ly)
        return np.diag([ixx, iyy, izz])

    @staticmethod
    def create_sphere_inertia(mass: float, radius: float) -> np.ndarray:
        """Solid sphere moment of inertia tensor: (2/5) * m * r^2 * I."""
        i = 0.4 * mass * (radius ** 2)
        return np.diag([i, i, i])

    @staticmethod
    def create_cylinder_inertia(mass: float, radius: float, height: float, axis: int = 1) -> np.ndarray:
        """Solid cylinder moment of inertia tensor."""
        i_long = 0.5 * mass * (radius ** 2)
        i_trans = (1.0 / 12.0) * mass * (3.0 * (radius ** 2) + height ** 2)
        diag = [i_trans, i_trans, i_trans]
        diag[axis] = i_long
        return np.diag(diag)
