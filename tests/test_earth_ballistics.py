"""
Unit tests for EarthGravityModel, EarthAirDragModel, ProjectileBallisticsEngine, and SurfaceContactFrictionModel.
"""

import numpy as np
import pytest

from pyaero3d.physics.earth_ballistics import (
    EarthGravityModel,
    EarthAirDragModel,
    ProjectileBallisticsEngine,
    SurfaceContactFrictionModel,
)


def test_earth_gravity_altitude_decay_and_latitude():
    # 1. Sea Level Gravity
    g0 = EarthGravityModel.get_gravity_at_altitude(0.0)
    assert abs(g0 - 9.814) < 0.05

    # 2. Low Earth Orbit (400 km altitude)
    g_leo = EarthGravityModel.get_gravity_at_altitude(400000.0)
    assert 8.5 < g_leo < 8.8
    assert g_leo < g0

    # 3. Somigliana Latitude Dependency
    g_equator = EarthGravityModel.get_somigliana_gravity(0.0)
    g_pole = EarthGravityModel.get_somigliana_gravity(90.0)
    assert abs(g_equator - 9.7803) < 0.01
    assert abs(g_pole - 9.8322) < 0.01
    assert g_pole > g_equator # Gravity is stronger at poles


def test_earth_aerodynamic_drag_vectors():
    vel = np.array([100.0, 50.0, -120.0]) # 3D flight velocity
    alt = 1500.0 # 1500m MSL
    cd = 0.35
    area = 2.0

    f_drag, diag = EarthAirDragModel.compute_aerodynamic_drag_vector(
        vel, alt, cd_base=cd, ref_area_m2=area
    )

    # Drag must directly oppose velocity vector: dot(f_drag, vel) < 0
    assert np.dot(f_drag, vel) < 0.0

    # Normalized direction check
    v_norm = vel / np.linalg.norm(vel)
    f_norm = f_drag / np.linalg.norm(f_drag)
    assert np.allclose(f_norm, -v_norm, atol=1e-6)

    # Dynamic pressure check
    speed = np.linalg.norm(vel)
    rho = diag["air_density"]
    expected_q = 0.5 * rho * (speed ** 2)
    assert abs(diag["dynamic_pressure_pa"] - expected_q) < 1e-3


def test_projectile_terminal_velocity_and_ballistics():
    mass = 80.0 # 80kg skydiver/projectile
    alt = 0.0
    cd = 1.0
    area = 0.7

    v_term = ProjectileBallisticsEngine.get_terminal_velocity(mass, alt, cd, area)
    # Human terminal velocity at sea level is ~ 45 - 55 m/s (~120 mph)
    assert 40.0 < v_term < 55.0

    # Test acceleration at terminal velocity in vertical freefall
    pos = np.array([0.0, 0.0, 0.0])
    vel_term = np.array([0.0, -v_term, 0.0])

    a_net, diag = ProjectileBallisticsEngine.evaluate_ballistic_acceleration(
        pos, vel_term, mass, cd=cd, ref_area_m2=area
    )

    # At terminal velocity, upward drag cancels downward gravity -> net vertical acceleration ~ 0
    assert abs(a_net[1]) < 0.15


def test_surface_contact_normal_and_coulomb_friction_vectors():
    pos = np.array([0.0, 10.0, 0.0])
    # Sliding on 30-degree incline slope
    normal = np.array([0.5, 0.866, 0.0]) # 30 deg slope
    normal = normal / np.linalg.norm(normal)
    vel_sliding = np.array([10.0, -5.77, 0.0]) # Sliding along slope
    mass = 100.0

    f_norm, f_fric, diag = SurfaceContactFrictionModel.compute_surface_forces(
        pos, vel_sliding, mass, normal, mu_kinetic=0.40
    )

    # Normal force must point along normal vector
    assert np.dot(f_norm, normal) > 0.0

    # Friction force must oppose sliding velocity: dot(f_fric, vel_sliding) < 0
    assert np.dot(f_fric, vel_sliding) < 0.0

    # Friction magnitude must obey Coulomb law: ||F_fric|| = mu * ||F_N||
    f_n_mag = np.linalg.norm(f_norm)
    f_f_mag = np.linalg.norm(f_fric)
    assert abs(f_f_mag - 0.40 * f_n_mag) < 1e-4
