"""
Unit tests for Advanced Physical Solvers:
Airfoil Cp Pressure, Orbital Hohmann Transfer, Chaotic Double Pendulum, and Lorentz Fields.
"""

import numpy as np
import pytest

from pyaero3d.physics.advanced_solvers import (
    NACA4AirfoilSolver, OrbitalMechanicsSolver,
    ChaoticDoublePendulumSolver, LorentzParticleSolver
)


def test_naca_airfoil_coordinates_and_pressure():
    # 1. Geometry generation for symmetric NACA 0012
    xu, yu, xl, yl = NACA4AirfoilSolver.generate_airfoil_coordinates(m_camber=0.0, p_camber_pos=0.0, t_thickness=0.12)
    assert len(xu) == 100
    assert abs(xu[0] - 0.0) < 1e-3
    assert abs(xu[-1] - 1.0) < 1e-3
    # Max thickness ~ 12% chord (yu - yl ~ 0.12)
    max_t = np.max(yu - yl)
    assert abs(max_t - 0.12) < 0.015

    # 2. Pressure distribution for cambered NACA 2412 at AoA = 4 deg
    x, cp_u, cp_l = NACA4AirfoilSolver.compute_pressure_distribution(alpha_deg=4.0, m_camber=0.02, t_thickness=0.12)
    assert len(x) == 100
    # Upper surface should experience suction (Cp < 0) near leading edge
    assert np.min(cp_u) < 0.0
    # Lower surface has higher pressure than upper surface (Cp_lower > Cp_upper)
    assert np.mean(cp_l) > np.mean(cp_u)


def test_orbital_hohmann_transfer_calculation():
    # LEO (400km) to GEO (35,786km) Hohmann transfer
    res = OrbitalMechanicsSolver.calculate_hohmann_transfer(r1_alt_km=400.0, r2_alt_km=35786.0)
    
    # Check LEO circular speed ~ 7.67 km/s
    assert abs(res["v1_mps"] - 7670.0) < 100.0
    # Check GEO circular speed ~ 3.07 km/s
    assert abs(res["v2_mps"] - 3075.0) < 100.0
    # Check total delta-v ~ 3.85 km/s
    assert abs(res["dv_total_mps"] - 3855.0) < 150.0
    # Check transfer time ~ 5.25 hours
    assert abs(res["transfer_time_hours"] - 5.25) < 0.3


def test_chaotic_double_pendulum_rk4():
    solver = ChaoticDoublePendulumSolver(l1=1.0, l2=1.0, m1=1.0, m2=1.0)
    init_state = np.array([np.pi / 2.0, 0.0, np.pi / 2.0, 0.0])

    # Step double pendulum
    next_state = solver.rk4_step(init_state, dt=0.01)
    assert len(next_state) == 4
    # Angular accelerations must start moving pendulum under gravity
    assert abs(next_state[1]) > 0.0 # omega 1
    assert abs(next_state[3]) > 0.0 # omega 2

    # Cartesian positions
    x1, y1, x2, y2 = solver.get_cartesian_positions(next_state)
    assert -2.0 <= x2 <= 2.0
    assert -2.0 <= y2 <= 2.0


def test_lorentz_particle_boris_integrator():
    # Uniform magnetic field B along Z-axis -> circular cyclotron gyromotion
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([100.0, 0.0, 0.0]) # Initial velocity in X
    E = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])      # 1 Tesla along Z

    dt = 0.001
    # Step through 500 Boris steps
    for _ in range(500):
        pos, vel = LorentzParticleSolver.step_boris(pos, vel, q=1.0, m=1.0, E=E, B=B, dt=dt)

    # Pure magnetic field does NO work on particle: speed must remain exactly 100 m/s
    current_speed = np.linalg.norm(vel)
    assert abs(current_speed - 100.0) < 1e-4
