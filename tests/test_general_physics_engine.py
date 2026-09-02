"""
Unit tests for General-Purpose 3D Physics Engine:
RigidBody Kinematics, GJK/EPA Collisions, Constraint Solver, Joints, and Force Fields.
"""

import numpy as np
import pytest

from pyaero3d.core.rigid_body import RigidBody
from pyaero3d.collision.geometry import SphereShape, BoxShape, CapsuleShape, ConvexHullShape
from pyaero3d.collision.gjk_epa import GJKEPASolver, ContactPoint
from pyaero3d.physics.constraint_solver import ContactConstraint, BallSocketJoint, SpringDamperJoint
from pyaero3d.physics.force_generators import UniformGravity, FluidBuoyancyField, BlastExplosionForce
from pyaero3d.core.physics_world import PhysicsWorld


def test_rigid_body_spatial_kinematics():
    mass = 10.0
    half_extents = np.array([1.0, 1.0, 1.0])
    inertia = RigidBody.create_box_inertia(mass, half_extents)

    body = RigidBody(mass=mass, inertia_tensor=inertia, position=np.array([0.0, 5.0, 0.0]))

    # Apply off-center force -> must create torque
    f_vec = np.array([100.0, 0.0, 0.0])
    p_world = np.array([0.0, 6.0, 0.0]) # 1m above CG (+Y)
    body.apply_force_at_world_point(f_vec, p_world)

    # Torque = r x F = [0, 1, 0] x [100, 0, 0] = [0, 0, -100]
    assert abs(body.torque_accum[2] - (-100.0)) < 1e-6

    # Step velocities
    body.integrate_velocities(dt=0.01)
    assert body.velocity[0] > 0.0
    assert body.angular_velocity[2] < 0.0


def test_gjk_epa_sphere_and_box_collision():
    # 1. Sphere-Sphere Overlap
    s1 = SphereShape(radius=1.0)
    s2 = SphereShape(radius=1.0)
    b1 = RigidBody(position=np.array([0.0, 0.0, 0.0]))
    b2 = RigidBody(position=np.array([1.5, 0.0, 0.0]))

    is_colliding, _ = GJKEPASolver.test_intersection(s1, b1, s2, b2)
    assert is_colliding is True

    # 2. Sphere-Sphere Separated
    b2_far = RigidBody(position=np.array([3.0, 0.0, 0.0]))
    is_colliding_far, _ = GJKEPASolver.test_intersection(s1, b1, s2, b2_far)
    assert is_colliding_far is False

    # 3. Box-Box Overlap
    box1 = BoxShape(np.array([1.0, 1.0, 1.0]))
    box2 = BoxShape(np.array([1.0, 1.0, 1.0]))
    b_box1 = RigidBody(position=np.array([0.0, 0.0, 0.0]))
    b_box2 = RigidBody(position=np.array([0.0, 1.8, 0.0]))

    contact = GJKEPASolver.solve_contact_manifold(box1, b_box1, box2, b_box2)
    assert contact is not None
    assert contact.depth > 0.0


def test_physics_world_ground_contact_and_settling():
    world = PhysicsWorld(solver_iterations=15)

    # Static ground plane box
    ground = RigidBody(is_static=True, position=np.array([0.0, 0.0, 0.0]))
    ground_shape = BoxShape(np.array([50.0, 1.0, 50.0]))
    world.add_body(ground, ground_shape)

    # Dynamic dropping sphere
    sphere = RigidBody(mass=5.0, position=np.array([0.0, 3.0, 0.0]), restitution=0.1)
    sphere_shape = SphereShape(radius=0.5)
    world.add_body(sphere, sphere_shape)

    # Simulate 1.5 seconds (150 steps at dt = 0.01)
    for _ in range(150):
        world.step(0.01)

    # Sphere should settle resting on ground surface (y = ground_top + radius = 1.0 + 0.5 = 1.5m)
    assert abs(sphere.position[1] - 1.5) < 0.15
    assert abs(sphere.velocity[1]) < 0.20 # Settled velocity ~ 0


def test_ball_socket_and_spring_damper_joints():
    world = PhysicsWorld()

    # 1. Ball-Socket Joint between static anchor and pendulum body
    anchor = RigidBody(is_static=True, position=np.array([0.0, 5.0, 0.0]))
    bob = RigidBody(mass=2.0, position=np.array([2.0, 5.0, 0.0]))
    world.add_body(anchor)
    world.add_body(bob)

    joint = BallSocketJoint(anchor, bob, anchor_world=np.array([0.0, 5.0, 0.0]))
    world.add_joint(joint)

    # Step simulation: bob swings under gravity while distance remains ~ 2m
    for _ in range(50):
        world.step(0.01)

    dist = np.linalg.norm(bob.position - anchor.position)
    assert abs(dist - 2.0) < 0.25

    # 2. Spring-Damper Joint
    b_top = RigidBody(is_static=True, position=np.array([0.0, 10.0, 0.0]))
    b_weight = RigidBody(mass=5.0, position=np.array([0.0, 8.0, 0.0]))
    spring = SpringDamperJoint(b_top, b_weight, [0, 0, 0], [0, 0, 0], rest_length=1.5, k_spring=200.0, c_damping=10.0)

    # Test spring force generation when stretched
    spring.update_forces()
    # Upward restoring force on weight
    assert b_weight.force_accum[1] > 0.0


def test_fluid_buoyancy_equilibrium():
    world = PhysicsWorld()
    # Remove default gravity generator to test pure buoyancy and custom gravity
    world.force_generators.clear()
    world.add_force_generator(UniformGravity(np.array([0.0, -9.81, 0.0])))

    # Wooden floating sphere (mass = 200 kg, radius = 0.5m, volume = 0.523 m^3, density = 382 kg/m^3 < 1000 kg/m^3)
    float_sphere = RigidBody(mass=200.0, position=np.array([0.0, -0.2, 0.0]), linear_damping=0.1)
    sphere_shape = SphereShape(radius=0.5)
    world.add_body(float_sphere, sphere_shape)

    buoyancy = FluidBuoyancyField(fluid_surface_y=0.0, fluid_density=1000.0)
    world.add_force_generator(buoyancy)

    # Step simulation until equilibrium
    for _ in range(200):
        world.step(0.01)

    # Buoyant upward force must hold sphere near surface
    assert -0.5 < float_sphere.position[1] < 0.3
