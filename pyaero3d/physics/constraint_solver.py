"""
PyAero3D - Sequential Impulse & Projected Gauss-Seidel (PGS) Constraint Solver.
Solves rigid contact manifolds, non-penetration, Coulomb friction, and multi-body articulated joints.
"""

from typing import List, Tuple, Optional
import numpy as np

from pyaero3d.core.rigid_body import RigidBody
from pyaero3d.collision.gjk_epa import ContactPoint


class ContactConstraint:
    """
    Projected Gauss-Seidel Non-Penetration & Coulomb Friction Constraint between 2 bodies.
    """

    def __init__(self, body_a: RigidBody, body_b: RigidBody, contact: ContactPoint):
        self.bodyA = body_a
        self.bodyB = body_b
        self.contact = contact

        # Geometric lever arms
        self.rA = contact.point - body_a.position
        self.rB = contact.point - body_b.position

        # Effective normal mass K_n
        rnA = np.cross(self.rA, contact.normal)
        rnB = np.cross(self.rB, contact.normal)
        k_normal = (body_a.inv_mass + body_b.inv_mass +
                    float(rnA @ (body_a.inv_inertia_world @ rnA)) +
                    float(rnB @ (body_b.inv_inertia_world @ rnB)))
        self.mass_normal = 1.0 / max(1e-9, k_normal)

        # Effective tangent masses K_t1, K_t2
        rt1A = np.cross(self.rA, contact.tangent1)
        rt1B = np.cross(self.rB, contact.tangent1)
        k_t1 = (body_a.inv_mass + body_b.inv_mass +
                float(rt1A @ (body_a.inv_inertia_world @ rt1A)) +
                float(rt1B @ (body_b.inv_inertia_world @ rt1B)))
        self.mass_tangent1 = 1.0 / max(1e-9, k_t1)

        rt2A = np.cross(self.rA, contact.tangent2)
        rt2B = np.cross(self.rB, contact.tangent2)
        k_t2 = (body_a.inv_mass + body_b.inv_mass +
                float(rt2A @ (body_a.inv_inertia_world @ rt2A)) +
                float(rt2B @ (body_b.inv_inertia_world @ rt2B)))
        self.mass_tangent2 = 1.0 / max(1e-9, k_t2)

        # Friction coefficient (geometric mean)
        self.friction = float(np.sqrt(body_a.friction_kinetic * body_b.friction_kinetic))
        # Restitution coefficient
        self.restitution = float(max(body_a.restitution, body_b.restitution))

        # Accumulated impulses
        self.accum_normal = 0.0
        self.accum_t1 = 0.0
        self.accum_t2 = 0.0

        # Baumgarte velocity bias
        self.bias = 0.0

    def pre_step(self, dt: float) -> None:
        """Computes Baumgarte position correction bias and restitution velocity."""
        # Relative velocity of Body B with respect to Body A: v_rel = vB - vA
        vA = self.bodyA.get_point_velocity(self.contact.point)
        vB = self.bodyB.get_point_velocity(self.contact.point)
        v_rel = vB - vA
        v_norm = float(np.dot(v_rel, self.contact.normal))

        # Position correction (Baumgarte stabilization)
        slop = 0.005 # 5mm allowable penetration
        beta = 0.20
        penetration = max(0.0, self.contact.depth - slop)
        pos_bias = (beta / dt) * penetration

        # Restitution velocity threshold (avoid jitter at rest)
        rest_bias = 0.0
        if v_norm < -0.5:
            rest_bias = self.restitution * v_norm

        self.bias = pos_bias - rest_bias

    def solve_velocity_constraint(self) -> None:
        """Solves normal non-penetration and tangential friction impulses iteratively."""
        vA = self.bodyA.get_point_velocity(self.contact.point)
        vB = self.bodyB.get_point_velocity(self.contact.point)
        v_rel = vB - vA

        # 1. Friction Tangent 1
        vt1 = float(np.dot(v_rel, self.contact.tangent1))
        d_lambda_t1 = self.mass_tangent1 * (-vt1)
        max_fric = self.friction * self.accum_normal
        old_t1 = self.accum_t1
        self.accum_t1 = float(np.clip(old_t1 + d_lambda_t1, -max_fric, max_fric))
        d_lambda_t1 = self.accum_t1 - old_t1

        impulse_t1 = d_lambda_t1 * self.contact.tangent1
        self.bodyA.apply_impulse_at_world_point(-impulse_t1, self.contact.point)
        self.bodyB.apply_impulse_at_world_point(impulse_t1, self.contact.point)

        # 2. Friction Tangent 2
        vA = self.bodyA.get_point_velocity(self.contact.point)
        vB = self.bodyB.get_point_velocity(self.contact.point)
        v_rel = vB - vA
        vt2 = float(np.dot(v_rel, self.contact.tangent2))
        d_lambda_t2 = self.mass_tangent2 * (-vt2)
        old_t2 = self.accum_t2
        self.accum_t2 = float(np.clip(old_t2 + d_lambda_t2, -max_fric, max_fric))
        d_lambda_t2 = self.accum_t2 - old_t2

        impulse_t2 = d_lambda_t2 * self.contact.tangent2
        self.bodyA.apply_impulse_at_world_point(-impulse_t2, self.contact.point)
        self.bodyB.apply_impulse_at_world_point(impulse_t2, self.contact.point)

        # 3. Normal Non-Penetration Constraint (lambda >= 0 pushes B in +normal and A in -normal)
        vA = self.bodyA.get_point_velocity(self.contact.point)
        vB = self.bodyB.get_point_velocity(self.contact.point)
        v_rel = vB - vA
        vn = float(np.dot(v_rel, self.contact.normal))

        d_lambda_n = self.mass_normal * (-(vn - self.bias))
        old_n = self.accum_normal
        self.accum_normal = max(0.0, old_n + d_lambda_n)
        d_lambda_n = self.accum_normal - old_n

        impulse_n = d_lambda_n * self.contact.normal
        self.bodyA.apply_impulse_at_world_point(-impulse_n, self.contact.point)
        self.bodyB.apply_impulse_at_world_point(impulse_n, self.contact.point)


class BallSocketJoint:
    """3-DOF Spherical / Ball-and-Socket Joint constraining two anchor points together."""

    def __init__(self, body_a: RigidBody, body_b: RigidBody, anchor_world: np.ndarray):
        self.bodyA = body_a
        self.bodyB = body_b
        self.localA = body_a.rotation_matrix.T @ (anchor_world - body_a.position)
        self.localB = body_b.rotation_matrix.T @ (anchor_world - body_b.position)
        self.accum_impulse = np.zeros(3, dtype=np.float64)

    def solve(self, dt: float) -> None:
        pA = self.bodyA.position + (self.bodyA.rotation_matrix @ self.localA)
        pB = self.bodyB.position + (self.bodyB.rotation_matrix @ self.localB)
        vA = self.bodyA.get_point_velocity(pA)
        vB = self.bodyB.get_point_velocity(pB)

        # Positional Baumgarte error
        pos_err = pB - pA
        bias = (0.20 / dt) * pos_err
        v_rel = (vB - vA) + bias

        rA = pA - self.bodyA.position
        rB = pB - self.bodyB.position

        # Skew-symmetric cross product matrices
        skew_A = np.array([
            [0.0, -rA[2], rA[1]],
            [rA[2], 0.0, -rA[0]],
            [-rA[1], rA[0], 0.0]
        ])
        skew_B = np.array([
            [0.0, -rB[2], rB[1]],
            [rB[2], 0.0, -rB[0]],
            [-rB[1], rB[0], 0.0]
        ])

        # Exact 3x3 K matrix
        k_mat = (self.bodyA.inv_mass + self.bodyB.inv_mass) * np.eye(3)
        if not self.bodyA.is_static:
            k_mat -= skew_A @ self.bodyA.inv_inertia_world @ skew_A
        if not self.bodyB.is_static:
            k_mat -= skew_B @ self.bodyB.inv_inertia_world @ skew_B

        inv_k = np.linalg.inv(k_mat + 1e-9 * np.eye(3))
        d_impulse = -inv_k @ v_rel
        self.accum_impulse += d_impulse

        self.bodyA.apply_impulse_at_world_point(-d_impulse, pA)
        self.bodyB.apply_impulse_at_world_point(d_impulse, pB)


class SpringDamperJoint:
    """Nonlinear Elastic Spring-Damper Constraint."""

    def __init__(self, body_a: RigidBody, body_b: RigidBody, local_a: np.ndarray, local_b: np.ndarray, rest_length: float, k_spring: float, c_damping: float):
        self.bodyA = body_a
        self.bodyB = body_b
        self.localA = np.asarray(local_a, dtype=np.float64)
        self.localB = np.asarray(local_b, dtype=np.float64)
        self.rest_len = float(rest_length)
        self.k = float(k_spring)
        self.c = float(c_damping)

    def update_forces(self) -> None:
        pA = self.bodyA.position + (self.bodyA.rotation_matrix @ self.localA)
        pB = self.bodyB.position + (self.bodyB.rotation_matrix @ self.localB)
        delta = pB - pA
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return

        direction = delta / dist
        displacement = dist - self.rest_len

        vA = self.bodyA.get_point_velocity(pA)
        vB = self.bodyB.get_point_velocity(pB)
        v_rel_proj = float(np.dot(vB - vA, direction))

        # Spring-damper force magnitude
        f_mag = self.k * displacement + self.c * v_rel_proj
        force_vec = f_mag * direction

        self.bodyA.apply_force_at_world_point(force_vec, pA)
        self.bodyB.apply_force_at_world_point(-force_vec, pB)
