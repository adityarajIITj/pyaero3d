"""
Unit tests for MountainTerrainGenerator and StandardAtmosphere.
"""

import numpy as np
import pytest

from pyaero3d.render.terrain_gen import MountainTerrainGenerator
from pyaero3d.physics.atmosphere import StandardAtmosphere


def test_mountain_terrain_generation_and_queries():
    terrain = MountainTerrainGenerator(grid_resolution=129, world_size_m=4000.0, max_height_m=1800.0)

    # 1. Runway corridor at center (0, 0) should be relatively flat
    h_center = terrain.get_height(0.0, 0.0)
    assert 0.0 <= h_center <= 1800.0

    # 2. Mountain summit queries
    h_ridge = terrain.get_height(1000.0, 1000.0)
    assert 0.0 <= h_ridge <= 1800.0

    # 3. Surface normal vector length must be unit normalized
    normal = terrain.get_surface_normal(500.0, -500.0)
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-5
    # Normal Y-component must point upwards
    assert normal[1] > 0.0


def test_standard_atmosphere_sea_level_and_stratosphere():
    # Sea Level (0 m)
    T0, P0, rho0, a0, mu0 = StandardAtmosphere.get_properties(0.0)
    assert abs(T0 - 288.15) < 0.1
    assert abs(P0 - 101325.0) < 1.0
    assert abs(rho0 - 1.225) < 0.01
    assert abs(a0 - 340.29) < 0.5

    # Tropopause (11,000 m)
    T11, P11, rho11, _, _ = StandardAtmosphere.get_properties(11000.0)
    assert abs(T11 - 216.65) < 0.2
    assert P11 < P0 * 0.25 # Pressure must drop significantly
    assert rho11 < rho0 * 0.35

    # Stratosphere (25,000 m)
    T25, P25, rho25, _, _ = StandardAtmosphere.get_properties(25000.0)
    assert P25 < P11
    assert rho25 < rho11
