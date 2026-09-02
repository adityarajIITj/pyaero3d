"""
PyAero3D - General-Purpose 3D Physics World Container.
Orchestrates rigid-body dynamics, broadphase/narrowphase collision, PGS constraint solver, and numerical integrators.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from pyaero3d.core.rigid_body import RigidBody
from pyaero3d.collision.geometry import CollisionShape
from pyaero3d.collision.gjk_epa import GJKEPASolver, ContactPoint
from pyaero3d.physics.constraint_solver import ContactConstraint, BallSocketJoint, SpringDamperJoint
from pyaero3d.physics.force_generators import ForceGenerator, UniformGravity


class PhysicsWorld:
    """
    General-Purpose 3D Physics Simulation World.
    """

    def __init__(self, solver_iterations: int = 10):
        self.bodies: List[RigidBody] = []
        self.force_generators: List[ForceGenerator] = []
        self.joints: List[Any] = []
        self.contact_constraints: List[ContactConstraint] = []
        self.solver_iterations = int(solver_iterations)

        # Add default Earth surface gravity
        self.default_gravity = UniformGravity()
        self.force_generators.append(self.default_gravity)

        self.time = 0.0
        self.step_count = 0

    def add_body(self, body: RigidBody, shape: Optional[CollisionShape] = None) -> RigidBody:
        """Adds a rigid body to the physics world."""
        if shape is not None:
            body.collision_shape = shape
        self.bodies.append(body)
        return body

    def remove_body(self, body: RigidBody) -> None:
        """Removes a rigid body from the physics world."""
        if body in self.bodies:
            self.bodies.remove(body)

    def add_force_generator(self, generator: ForceGenerator) -> None:
        """Registers a global or local force field generator."""
        self.force_generators.append(generator)

    def add_joint(self, joint: Any) -> None:
        """Registers a multi-body articulated joint constraint."""
        self.joints.append(joint)

    def step(self, dt: float) -> None:
        """
        Advances the 3D physics simulation by time step dt.
        """
        # 1. Apply all force generators (Gravity, Buoyancy, Aero, user forces)
        for fg in self.force_generators:
            fg.apply_forces(self.bodies, dt)

        # 2. Update Spring-Damper Joints
        for joint in self.joints:
            if isinstance(joint, SpringDamperJoint):
                joint.update_forces()

        # 3. Integrate Unconstrained Velocities
        for body in self.bodies:
            body.integrate_velocities(dt)

        # 4. Collision Detection (Broadphase + Narrowphase GJK/EPA)
        self.contact_constraints.clear()
        num_bodies = len(self.bodies)

        for i in range(num_bodies):
            bodyA = self.bodies[i]
            if bodyA.collision_shape is None:
                continue

            aabb_min_a, aabb_max_a = bodyA.collision_shape.compute_aabb(bodyA.position, bodyA.rotation_matrix)

            for j in range(i + 1, num_bodies):
                bodyB = self.bodies[j]
                if bodyB.collision_shape is None:
                    continue

                if bodyA.is_static and bodyB.is_static:
                    continue # Skip static-static pairs

                # Broadphase AABB overlap test
                aabb_min_b, aabb_max_b = bodyB.collision_shape.compute_aabb(bodyB.position, bodyB.rotation_matrix)
                if (aabb_max_a[0] < aabb_min_b[0] or aabb_min_a[0] > aabb_max_b[0] or
                    aabb_max_a[1] < aabb_min_b[1] or aabb_min_a[1] > aabb_max_b[1] or
                    aabb_max_a[2] < aabb_min_b[2] or aabb_min_a[2] > aabb_max_b[2]):
                    continue # No AABB overlap

                # Narrowphase GJK + EPA
                contact = GJKEPASolver.solve_contact_manifold(
                    bodyA.collision_shape, bodyA,
                    bodyB.collision_shape, bodyB
                )
                if contact is not None:
                    c_constraint = ContactConstraint(bodyA, bodyB, contact)
                    c_constraint.pre_step(dt)
                    self.contact_constraints.append(c_constraint)

        # 5. Projected Gauss-Seidel Constraint Solver Iterations
        for _ in range(self.solver_iterations):
            # Solve contacts
            for cc in self.contact_constraints:
                cc.solve_velocity_constraint()

            # Solve joints (Ball-Socket, etc.)
            for joint in self.joints:
                if hasattr(joint, "solve"):
                    joint.solve(dt)

        # 6. Integrate Positions & Orientations
        for body in self.bodies:
            body.integrate_positions(dt)

        self.time += dt
        self.step_count += 1
