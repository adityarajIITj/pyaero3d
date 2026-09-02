"""
Unit tests for FlightDynamicsSolver, SpatialQuaternion, and StateBuffer.
"""

import numpy as np
import pytest

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer
from pyaero3d.core.quaternion_math import SpatialQuaternion
from pyaero3d.physics.flight_dynamics import FlightDynamicsSolver


def test_spatial_quaternion_dcm_and_rotations():
    # 90-degree pitch up around Z axis
    axis = np.array([0.0, 0.0, 1.0])
    q = SpatialQuaternion.from_axis_angle(axis, np.radians(90.0))

    R = SpatialQuaternion.to_dcm(q)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9 # Valid SO(3) matrix

    # Rotate unit vector along +X: should point along +Y
    v_in = np.array([1.0, 0.0, 0.0])
    v_rot = SpatialQuaternion.rotate_vector(q, v_in)
    assert abs(v_rot[0] - 0.0) < 1e-6
    assert abs(v_rot[1] - 1.0) < 1e-6


def test_fixed_wing_lift_and_thrust_generation():
    buffer = StateBuffer(max_entities=10)
    idx = buffer.allocate_entity(
        entity_type=EntityType.FIXED_WING_JET,
        mass=12000.0,
        position=np.array([0.0, 2000.0, 0.0]),
        velocity=np.array([0.0, 0.0, 220.0]), # 220 m/s cruise
        radius=4.0,
        cd=0.025,
        area=28.0,
    )
    buffer.data[idx, StateIdx.THROTTLE] = 0.80

    f_tot, tau_body = FlightDynamicsSolver.evaluate_entity_dynamics(buffer.data[idx], dt=0.001)

    # Must produce forward thrust (Z) and upward lift (Y)
    assert f_tot[2] > 0.0
    assert f_tot[1] > -12000.0 * 9.81 # Lift must oppose gravity


def test_rocket_mass_depletion_and_thrust():
    buffer = StateBuffer(max_entities=10)
    idx = buffer.allocate_entity(
        entity_type=EntityType.MULTI_STAGE_ROCKET,
        mass=25000.0,
        position=np.array([0.0, 5000.0, 0.0]),
        velocity=np.array([0.0, 300.0, 0.0]),
        radius=2.0,
        fuel_mass=18000.0,
    )
    buffer.data[idx, StateIdx.THROTTLE] = 1.0

    m_init = buffer.data[idx, StateIdx.MASS]
    f_tot, _ = FlightDynamicsSolver.evaluate_entity_dynamics(buffer.data[idx], dt=0.1)

    # Thrust must be strongly positive along vertical launch axis (Y)
    assert f_tot[1] > 0.0
    # Mass must deplete
    assert buffer.data[idx, StateIdx.MASS] < m_init
