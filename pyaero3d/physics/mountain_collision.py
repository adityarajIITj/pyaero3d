"""
PyAero3D - Mountain Terrain Collision, Landing Gear Shock Struts, and Impact Detection.
"""

from typing import Tuple, List, Optional
import numpy as np

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.quaternion_math import SpatialQuaternion
from pyaero3d.render.terrain_gen import MountainTerrainGenerator


class MountainCollisionEngine:
    """
    Handles terrain penetration constraints, landing gear shock absorption,
    and high-speed kinetic mountain crashes.
    """

    def __init__(self, terrain: MountainTerrainGenerator, crash_speed_threshold: float = 38.0):
        self.terrain = terrain
        self.v_crash = crash_speed_threshold
        # Spring-damper shock strut stiffness & damping
        self.k_strut = 180000.0  # N/m
        self.c_strut = 12000.0   # N*s/m

    def resolve_entity_collision(
        self,
        state_row: np.ndarray,
        dt: float,
    ) -> Tuple[bool, bool]:
        """
        Tests and resolves collision against continuous mountain terrain.
        Returns: (is_on_ground, is_critical_crash)
        """
        px = float(state_row[StateIdx.PX])
        py = float(state_row[StateIdx.PY])
        pz = float(state_row[StateIdx.PZ])
        radius = float(state_row[StateIdx.RADIUS])
        vel = state_row[StateIdx.VX:StateIdx.VZ + 1]

        # Query terrain elevation and surface normal at (px, pz)
        ground_h = self.terrain.get_height(px, pz)
        normal = self.terrain.get_surface_normal(px, pz)

        penetration = (ground_h + radius) - py

        if penetration > 0.0:
            # Entity is in contact with terrain
            impact_speed = float(np.linalg.norm(vel))

            # Check if this is a high-speed destructive crash into mountain cliff
            v_normal = float(np.dot(vel, normal))
            if (impact_speed > self.v_crash or v_normal < -25.0) and int(state_row[StateIdx.ENTITY_TYPE]) != EntityType.DEBRIS_FRAGMENT:
                # Critical impact -> Trigger Class 11 kinetic breakup!
                return True, True

            # Normal reaction force via spring-damper shock strut model
            # delta = penetration, delta_dot = -v_normal
            delta_dot = -v_normal
            f_n_mag = max(0.0, self.k_strut * penetration + self.c_strut * delta_dot)
            f_normal = f_n_mag * normal

            # Tangential friction (Coulomb model)
            v_tangent = vel - v_normal * normal
            v_tangent_mag = float(np.linalg.norm(v_tangent))
            mu_fric = state_row[StateIdx.SURFACE_FRICTION] if state_row[StateIdx.SURFACE_FRICTION] > 0.01 else 0.65

            if v_tangent_mag > 1e-3:
                f_fric = -min(f_n_mag * mu_fric, (v_tangent_mag / dt) * state_row[StateIdx.MASS]) * (v_tangent / v_tangent_mag)
            else:
                f_fric = np.zeros(3)

            # Apply forces directly to state row
            state_row[StateIdx.FX:StateIdx.FZ + 1] += (f_normal + f_fric)

            # Enforce hard non-penetration position correction
            state_row[StateIdx.PY] = ground_h + radius
            state_row[StateIdx.ON_GROUND] = 1.0

            # Restitution velocity damping
            restitution = 0.15
            if v_normal < 0.0:
                vel_new = vel - (1.0 + restitution) * v_normal * normal
                state_row[StateIdx.VX:StateIdx.VZ + 1] = vel_new

            return True, False

        state_row[StateIdx.ON_GROUND] = 0.0
        return False, False
