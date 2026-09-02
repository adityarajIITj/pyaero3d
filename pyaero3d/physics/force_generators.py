"""
PyAero3D - Universal Physics Force Generators & Environmental Field Subsystems.
Supports Gravity, Fluid Buoyancy, Aerodynamic Surfaces, N-Body Gravity, and Explosions.
"""

from typing import List, Tuple, Optional
import numpy as np

from pyaero3d.core.types import G_GRAVITATIONAL, STANDARD_GRAVITY
from pyaero3d.core.rigid_body import RigidBody
from pyaero3d.physics.atmosphere import StandardAtmosphere


class ForceGenerator:
    """Base class for all persistent physical force generators."""

    def apply_forces(self, bodies: List[RigidBody], dt: float) -> None:
        raise NotImplementedError


class UniformGravity(ForceGenerator):
    """Uniform gravitational acceleration field (e.g. Earth surface g = -9.80665 m/s^2 along Y)."""

    def __init__(self, gravity_vector: Optional[np.ndarray] = None):
        self.gravity = np.asarray(
            gravity_vector if gravity_vector is not None else [0.0, -STANDARD_GRAVITY, 0.0],
            dtype=np.float64,
        )

    def apply_forces(self, bodies: List[RigidBody], dt: float) -> None:
        for b in bodies:
            if not b.is_static:
                b.apply_force(b.mass * self.gravity)


class PointGravityField(ForceGenerator):
    """Spherical central body gravitational field: F = -G * M * m / r^2."""

    def __init__(self, center_position: np.ndarray, central_mass: float):
        self.center = np.asarray(center_position, dtype=np.float64)
        self.mass = float(central_mass)

    def apply_forces(self, bodies: List[RigidBody], dt: float) -> None:
        for b in bodies:
            if not b.is_static:
                delta = self.center - b.position
                dist = float(np.linalg.norm(delta))
                if dist > 1.0:
                    dir_vec = delta / dist
                    f_mag = (G_GRAVITATIONAL * self.mass * b.mass) / (dist * dist)
                    b.apply_force(f_mag * dir_vec)


class FluidBuoyancyField(ForceGenerator):
    """
    Archimedes principle fluid buoyancy and hydrodynamic viscous drag.
    F_b = rho_fluid * V_submerged * g
    """

    def __init__(self, fluid_surface_y: float = 0.0, fluid_density: float = 1000.0, fluid_drag_cd: float = 0.80):
        self.surface_y = float(fluid_surface_y)
        self.density = float(fluid_density)
        self.cd = float(fluid_drag_cd)

    def apply_forces(self, bodies: List[RigidBody], dt: float) -> None:
        for b in bodies:
            if b.is_static:
                continue

            # Estimate submerged depth from body bounding radius
            radius = 0.5
            if b.collision_shape is not None and hasattr(b.collision_shape, "radius"):
                radius = b.collision_shape.radius
            elif b.collision_shape is not None and hasattr(b.collision_shape, "half_extents"):
                radius = float(np.mean(b.collision_shape.half_extents))

            submerged_h = self.surface_y - (b.position[1] - radius)
            if submerged_h > 0.0:
                sub_fraction = min(1.0, submerged_h / (2.0 * radius))
                volume = (4.0 / 3.0) * np.pi * (radius ** 3)
                v_sub = volume * sub_fraction

                # Upward buoyant force
                f_buoyancy = np.array([0.0, self.density * v_sub * STANDARD_GRAVITY, 0.0])
                b.apply_force(f_buoyancy)

                # Fluid viscous drag opposing velocity
                speed = float(np.linalg.norm(b.velocity))
                if speed > 1e-4:
                    area = np.pi * (radius ** 2) * sub_fraction
                    f_drag = -0.5 * self.density * (speed ** 2) * self.cd * area * (b.velocity / speed)
                    b.apply_force(f_drag)


class BlastExplosionForce(ForceGenerator):
    """
    Instantaneous radial blast shockwave and impulse field.
    """

    def __init__(self, epicenter_xyz: np.ndarray, energy_joules: float = 1000000.0, blast_radius: float = 35.0):
        self.epicenter = np.asarray(epicenter_xyz, dtype=np.float64)
        self.energy = float(energy_joules)
        self.radius = float(blast_radius)
        self.has_fired = False

    def apply_forces(self, bodies: List[RigidBody], dt: float) -> None:
        if self.has_fired:
            return

        for b in bodies:
            if b.is_static:
                continue

            delta = b.position - self.epicenter
            dist = float(np.linalg.norm(delta))
            if 0.001 < dist <= self.radius:
                dir_vec = delta / dist
                # Inverse square blast pressure falloff
                falloff = (1.0 - dist / self.radius) ** 2.0
                impulse_mag = np.sqrt(2.0 * self.energy * b.mass * falloff) / max(1.0, dist)
                b.apply_impulse(impulse_mag * dir_vec + np.array([0.0, impulse_mag * 0.3, 0.0]))

        self.has_fired = True
