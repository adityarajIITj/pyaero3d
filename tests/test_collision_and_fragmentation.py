"""
Unit tests for MountainCollisionEngine and Class 11 FragmentationEngine.
"""

import numpy as np
import pytest

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer
from pyaero3d.render.terrain_gen import MountainTerrainGenerator
from pyaero3d.physics.mountain_collision import MountainCollisionEngine
from pyaero3d.physics.fragmentation import FragmentationEngine


def test_mountain_collision_and_ground_support():
    terrain = MountainTerrainGenerator(grid_resolution=65, world_size_m=2000.0, max_height_m=1000.0)
    collision = MountainCollisionEngine(terrain, crash_speed_threshold=35.0)

    buffer = StateBuffer(max_entities=10)
    # Place entity slightly below terrain surface
    gh = terrain.get_height(100.0, 100.0)
    idx = buffer.allocate_entity(
        entity_type=EntityType.FIXED_WING_JET,
        mass=5000.0,
        position=np.array([100.0, gh - 0.2, 100.0]),
        velocity=np.array([0.0, -1.0, 15.0]), # Taxiing/landing speed
        radius=2.0,
    )

    is_ground, is_crash = collision.resolve_entity_collision(buffer.data[idx], dt=0.001)

    assert is_ground is True
    assert is_crash is False
    # Upward normal force must be positive
    assert buffer.data[idx, StateIdx.FY] > 0.0


def test_class11_fragmentation_strict_momentum_conservation():
    buffer = StateBuffer(max_entities=100)

    # Test across 25 independent random parent explosion states
    for trial in range(25):
        parent_mass = np.random.uniform(500.0, 25000.0)
        parent_vel = np.random.uniform(-150.0, 150.0, size=3)
        parent_pos = np.random.uniform(-1000.0, 1000.0, size=3)

        p_idx = buffer.allocate_entity(
            entity_type=EntityType.FIXED_WING_JET,
            mass=parent_mass,
            position=parent_pos,
            velocity=parent_vel,
        )

        initial_momentum = parent_mass * parent_vel

        # Trigger Class 11 breakup
        shards = FragmentationEngine.explode_entity(
            buffer, p_idx, num_shards=12, dispersion_energy_j=350000.0
        )

        assert len(shards) == 12

        # Compute sum of debris shards linear momentum
        total_shard_momentum = np.zeros(3, dtype=np.float64)
        for s in shards:
            m_s = buffer.data[s, StateIdx.MASS]
            v_s = buffer.data[s, StateIdx.VX:StateIdx.VZ + 1]
            total_shard_momentum += m_s * v_s

        # Relative residual momentum error must be at machine floating-point precision (< 1e-12)
        residual = np.linalg.norm(total_shard_momentum - initial_momentum)
        rel_error = residual / max(1.0, float(np.linalg.norm(initial_momentum)))
        assert rel_error < 1e-12

        # Clean up buffer
        for s in shards:
            buffer.free_entity(s)
