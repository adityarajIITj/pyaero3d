"""
PyAero3D - Earth Gravity, Aerodynamic Drag, Projectile Ballistics & Surface Vector Physics.
Provides first-principles physics formulations for Earth atmosphere, altitude-dependent gravity g(h),
compressible quadratic air drag, Magnus spin forces, ballistic trajectories, and surface friction vectors.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from pyaero3d.core.types import (
    G_GRAVITATIONAL,
    EARTH_MASS,
    EARTH_RADIUS,
    STANDARD_GRAVITY,
    SEA_LEVEL_DENSITY,
)
from pyaero3d.physics.atmosphere import StandardAtmosphere


class EarthGravityModel:
    """
    High-Precision Earth Gravitational Field Model.
    Calculates altitude-decay gravity g(h) and Somigliana latitude-dependent gravity g(phi).
    """

    @staticmethod
    def get_gravity_at_altitude(altitude_m: float) -> float:
        """
        Calculates altitude-decay gravitational acceleration g(h) in m/s^2.
        g(h) = g0 * (R_earth / (R_earth + h))^2 = G * M_earth / (R_earth + h)^2
        """
        r = EARTH_RADIUS + max(-1000.0, altitude_m)
        return float(G_GRAVITATIONAL * EARTH_MASS / (r * r))

    @staticmethod
    def get_gravity(altitude_m: float) -> float:
        """Alias for get_gravity_at_altitude."""
        return EarthGravityModel.get_gravity_at_altitude(altitude_m)

    @staticmethod
    def get_gravity_vector(position_xyz: np.ndarray) -> np.ndarray:
        """
        Calculates 3D gravitational acceleration vector pointing toward Earth center.
        In global Y-up coordinate frame (Y = Altitude above sea level).
        """
        alt = float(position_xyz[1])
        g_mag = EarthGravityModel.get_gravity_at_altitude(alt)
        return np.array([0.0, -g_mag, 0.0], dtype=np.float64)

    @staticmethod
    def get_somigliana_gravity(latitude_deg: float, altitude_m: float = 0.0) -> float:
        """
        International Gravity Formula (WGS-84 / Somigliana equation).
        Accounts for Earth oblateness and centrifugal rotation reduction at equator.
        """
        lat_rad = np.radians(latitude_deg)
        sin_lat = np.sin(lat_rad)
        sin2_lat = sin_lat * sin_lat

        # WGS-84 constants
        g_e = 9.7803267714   # Equatorial gravity (m/s^2)
        k = 0.00193185138639 # Somigliana formula constant
        e2 = 0.00669437999014 # First eccentricity squared

        # Surface gravity at latitude phi
        g_surface = g_e * (1.0 + k * sin2_lat) / np.sqrt(1.0 - e2 * sin2_lat)

        # Free-air correction for altitude h: delta_g = -2 * (g_surface / R_earth) * h
        r = EARTH_RADIUS + altitude_m
        g_h = g_surface * ((EARTH_RADIUS / r) ** 2)
        return float(g_h)


class EarthAirDragModel:
    """
    Compressible Earth Aerodynamic Air Drag & Skin Friction Vector Model.
    """

    @staticmethod
    def compute_aerodynamic_drag_vector(
        velocity_xyz: np.ndarray,
        altitude_m: float,
        cd_base: float = 0.30,
        ref_area_m2: float = 1.0,
        wind_xyz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Computes 3D aerodynamic drag force vector F_drag opposing relative wind vector.
        F_drag = -0.5 * rho(h) * ||v_rel|| * C_D(M) * A * v_rel
        """
        if wind_xyz is not None:
            v_rel = velocity_xyz - wind_xyz
        else:
            v_rel = velocity_xyz

        speed = float(np.linalg.norm(v_rel))
        if speed < 1e-5:
            return np.zeros(3, dtype=np.float64), {
                "dynamic_pressure_pa": 0.0,
                "mach_number": 0.0,
                "drag_coefficient": cd_base,
                "air_density": SEA_LEVEL_DENSITY,
                "drag_magnitude_n": 0.0,
            }

        # 1. Environmental Air Density & Speed of Sound
        _, _, rho, a_sound, mu_visc = StandardAtmosphere.get_properties(altitude_m)

        # 2. Dynamic Pressure: q = 0.5 * rho * v^2
        q_dyn = 0.5 * rho * (speed ** 2)

        # 3. Mach Number: M = v / a
        mach = speed / max(1.0, a_sound)

        # 4. Mach Drag Divergence Scaling (Compressible Wave Drag Rise)
        if mach < 0.80:
            # Subsonic: Prandtl-Glauert compressibility correction
            cd_mach = cd_base / max(0.2, np.sqrt(1.0 - (mach ** 2) * 0.95))
        elif mach < 1.20:
            # Transonic shockwave formation peak
            transonic_peak = (mach - 0.80) / 0.40 # 0.0 to 1.0
            cd_mach = cd_base * (1.0 + 2.5 * np.sin(transonic_peak * np.pi * 0.5))
        else:
            # Supersonic decay towards wave drag asymptote: C_D ~ 1 / sqrt(M^2 - 1)
            cd_mach = cd_base * (1.8 / np.sqrt(max(0.1, mach ** 2 - 1.0)) + 0.8)

        # 5. Drag Force Vector: F_drag = -q * C_D * A * (v_rel / ||v_rel||)
        drag_mag = q_dyn * cd_mach * ref_area_m2
        v_unit = v_rel / speed
        f_drag = -drag_mag * v_unit

        diagnostics = {
            "dynamic_pressure_pa": q_dyn,
            "mach_number": mach,
            "drag_coefficient": cd_mach,
            "air_density": rho,
            "drag_magnitude_n": drag_mag,
        }
        return f_drag, diagnostics

    @staticmethod
    def compute_magnus_spin_force(
        velocity_xyz: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
        altitude_m: float,
        radius_m: float = 0.1,
    ) -> np.ndarray:
        """
        Computes Magnus effect spin force vector: F_magnus = (4/3) * pi * rho * r^3 * (omega x v)
        """
        _, _, rho, _, _ = StandardAtmosphere.get_properties(altitude_m)
        # Cross product: omega x v
        spin_cross_v = np.cross(angular_velocity_rad_s, velocity_xyz)
        c_m = (4.0 / 3.0) * np.pi * rho * (radius_m ** 3)
        return c_m * spin_cross_v


class ProjectileBallisticsEngine:
    """
    Complete 3D Earth Projectile Motion & Ballistic Trajectory Solver.
    Integrates gravity g(h), atmospheric air drag, and terminal velocity.
    """

    @staticmethod
    def get_terminal_velocity(
        mass_kg: float,
        altitude_m: float,
        cd: float = 0.47, # Sphere Cd
        ref_area_m2: float = 0.05,
    ) -> float:
        """
        Analytical terminal velocity in Earth atmosphere:
        v_terminal = sqrt(2 * m * g(h) / (rho(h) * C_D * A))
        """
        g_h = EarthGravityModel.get_gravity_at_altitude(altitude_m)
        _, _, rho_h, _, _ = StandardAtmosphere.get_properties(altitude_m)
        v_term = np.sqrt((2.0 * mass_kg * g_h) / max(1e-6, rho_h * cd * ref_area_m2))
        return float(v_term)

    @staticmethod
    def evaluate_ballistic_acceleration(
        position_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        mass_kg: float,
        cd: float = 0.35,
        ref_area_m2: float = 0.1,
        wind_xyz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes total instantaneous ballistic acceleration vector a_net = g(h) + a_drag.
        """
        alt = float(position_xyz[1])
        # 1. Earth Gravity Acceleration Vector
        g_vec = EarthGravityModel.get_gravity_vector(position_xyz)

        # 2. Aerodynamic Drag Force Vector
        f_drag, diag = EarthAirDragModel.compute_aerodynamic_drag_vector(
            velocity_xyz, alt, cd_base=cd, ref_area_m2=ref_area_m2, wind_xyz=wind_xyz
        )

        # 3. Acceleration: a = g + (F_drag / m)
        a_drag = f_drag / max(1e-4, mass_kg)
        a_net = g_vec + a_drag

        diag["gravity_acceleration_mps2"] = np.linalg.norm(g_vec)
        diag["drag_acceleration_mps2"] = np.linalg.norm(a_drag)
        diag["net_acceleration_mps2"] = np.linalg.norm(a_net)
        diag["terminal_velocity_mps"] = ProjectileBallisticsEngine.get_terminal_velocity(
            mass_kg, alt, cd, ref_area_m2
        )
        return a_net, diag


class SurfaceContactFrictionModel:
    """
    Surface Vector Normal Reaction & Anisotropic Coulomb Friction Model.
    Decomposes velocity into normal and tangential surface vectors.
    """

    @staticmethod
    def compute_surface_forces(
        position_xyz: np.ndarray,
        velocity_xyz: np.ndarray,
        mass_kg: float,
        surface_normal: np.ndarray,
        mu_static: float = 0.70,
        mu_kinetic: float = 0.60,
        restitution: float = 0.20,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes normal constraint reaction vector and Coulomb surface friction vector.
        Returns: (F_normal_vector, F_friction_vector, diagnostics)
        """
        norm_unit = surface_normal / max(1e-6, np.linalg.norm(surface_normal))

        # Gravity at location
        g_vec = EarthGravityModel.get_gravity_vector(position_xyz)
        f_gravity = mass_kg * g_vec

        # Normal component of gravity acting into surface: F_N = - (F_gravity . n) * n
        f_g_dot_n = float(np.dot(f_gravity, norm_unit))
        f_n_mag = max(0.0, -f_g_dot_n)
        f_normal = f_n_mag * norm_unit

        # Decompose velocity into normal and tangent vectors
        v_normal_scalar = float(np.dot(velocity_xyz, norm_unit))
        v_normal_vec = v_normal_scalar * norm_unit
        v_tangent_vec = velocity_xyz - v_normal_vec
        v_tangent_speed = float(np.linalg.norm(v_tangent_vec))

        # Coulomb Friction opposing tangential velocity vector: F_fric = - mu * F_N * (v_tangent / ||v_tangent||)
        if v_tangent_speed > 1e-3:
            mu_curr = mu_kinetic
            f_fric = -mu_curr * f_n_mag * (v_tangent_vec / v_tangent_speed)
        else:
            # Static stick zone
            f_fric = -min(mu_static * f_n_mag, mass_kg * (v_tangent_speed / 0.001)) * (v_tangent_vec / (v_tangent_speed + 1e-6))

        diagnostics = {
            "normal_force_n": f_n_mag,
            "friction_force_n": float(np.linalg.norm(f_fric)),
            "tangent_speed_mps": v_tangent_speed,
            "normal_velocity_mps": v_normal_scalar,
        }
        return f_normal, f_fric, diagnostics
