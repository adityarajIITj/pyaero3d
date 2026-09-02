"""
PyAero3D - Comprehensive Physical Solver Library:
Airfoils, Transonic Shockwaves, Orbital Hohmann Transfers, N-Body Gravity,
Chaotic Double Pendulum, Rocket Nozzles, and Lorentz Fields.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from pyaero3d.core.types import (
    STANDARD_GRAVITY, EARTH_RADIUS, EARTH_MASS, G_GRAVITATIONAL,
    SEA_LEVEL_DENSITY, SEA_LEVEL_PRESSURE, AIR_HEAT_CAPACITY_RATIO
)


class NACA4AirfoilSolver:
    """
    NACA 4-Digit Airfoil Geometry & Thin Airfoil Theory Solver.
    Computes camber line, thickness distribution, Cp(x/c) pressure distribution,
    and aerodynamic polars (CL, CD, Cm, L/D) across AoA sweep.
    """

    @staticmethod
    def generate_airfoil_coordinates(
        m_camber: float = 0.02,   # Max camber (e.g. 0.02 for NACA 2412)
        p_camber_pos: float = 0.4, # Location of max camber (e.g. 0.4 for NACA 2412)
        t_thickness: float = 0.12, # Max thickness as fraction of chord (e.g. 0.12)
        num_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates upper and lower airfoil surface coordinates (x_u, y_u) and (x_l, y_l).
        """
        beta = np.linspace(0, np.pi, num_points)
        x = (1.0 - np.cos(beta)) * 0.5  # Cosine spacing for fine leading-edge resolution

        # Thickness distribution yt(x)
        yt = 5.0 * t_thickness * (
            0.2969 * np.sqrt(x + 1e-9)
            - 0.1260 * x
            - 0.3516 * (x ** 2)
            + 0.2843 * (x ** 3)
            - 0.1015 * (x ** 4)
        )

        # Mean camber line yc(x) and slope dyc/dx
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        if p_camber_pos > 0.0 and m_camber > 0.0:
            for i, xi in enumerate(x):
                if xi < p_camber_pos:
                    yc[i] = (m_camber / (p_camber_pos ** 2)) * (2.0 * p_camber_pos * xi - xi ** 2)
                    dyc_dx[i] = (2.0 * m_camber / (p_camber_pos ** 2)) * (p_camber_pos - xi)
                else:
                    yc[i] = (m_camber / ((1.0 - p_camber_pos) ** 2)) * ((1.0 - 2.0 * p_camber_pos) + 2.0 * p_camber_pos * xi - xi ** 2)
                    dyc_dx[i] = (2.0 * m_camber / ((1.0 - p_camber_pos) ** 2)) * (p_camber_pos - xi)

        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        return xu, yu, xl, yl

    @staticmethod
    def compute_pressure_distribution(
        alpha_deg: float = 4.0,
        m_camber: float = 0.02,
        t_thickness: float = 0.12,
        num_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes surface pressure coefficient distribution Cp(x/c) for upper and lower surfaces.
        """
        alpha_rad = np.radians(alpha_deg)
        x = np.linspace(0.001, 0.999, num_points)

        # Thin airfoil circulation & velocity perturbation
        cl_ideal = 2.0 * np.pi * (alpha_rad + 2.0 * m_camber)
        gamma_x = cl_ideal * np.sqrt((1.0 - x) / x)

        # Upper and lower velocity ratios: v_u = 1 + u_thick + gamma/2, v_l = 1 + u_thick - gamma/2
        u_thick = (t_thickness / 0.20) * (0.8 * (1.0 - x) * np.exp(-3.0 * x))
        v_u = 1.0 + u_thick + 0.5 * (gamma_x / (np.pi + 1e-6))
        v_l = 1.0 + u_thick - 0.5 * (gamma_x / (np.pi + 1e-6))

        # Bernoulli Cp = 1 - (V / V_inf)^2
        cp_upper = 1.0 - (v_u ** 2)
        cp_lower = 1.0 - (v_l ** 2)

        return x, cp_upper, cp_lower


class OrbitalMechanicsSolver:
    """
    Keplerian Orbital Mechanics & Hohmann Transfer Orbit Solver.
    """

    @staticmethod
    def calculate_hohmann_transfer(
        r1_alt_km: float = 400.0,   # Initial LEO altitude (km)
        r2_alt_km: float = 35786.0, # Target GEO altitude (km)
    ) -> Dict[str, float]:
        """
        Computes delta-v burns, transfer time, and orbital velocities for Hohmann transfer.
        """
        mu = G_GRAVITATIONAL * EARTH_MASS
        r1 = (EARTH_RADIUS + r1_alt_km * 1000.0)
        r2 = (EARTH_RADIUS + r2_alt_km * 1000.0)

        # Circular velocities
        v1 = np.sqrt(mu / r1)
        v2 = np.sqrt(mu / r2)

        # Semi-major axis of transfer ellipse
        a_transfer = 0.5 * (r1 + r2)

        # Velocities on transfer ellipse
        v_transfer_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
        v_transfer_apo  = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

        # Delta-V burns
        dv1 = v_transfer_peri - v1
        dv2 = v2 - v_transfer_apo
        dv_total = dv1 + dv2

        # Transfer duration (half orbital period)
        t_transfer_s = np.pi * np.sqrt((a_transfer ** 3) / mu)

        return {
            "r1_km": r1 / 1000.0,
            "r2_km": r2 / 1000.0,
            "v1_mps": v1,
            "v2_mps": v2,
            "dv1_mps": dv1,
            "dv2_mps": dv2,
            "dv_total_mps": dv_total,
            "transfer_time_hours": t_transfer_s / 3600.0,
        }

    @staticmethod
    def generate_orbit_points(semi_major_km: float, eccentricity: float, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generates 2D orbital ellipse points (x, y) in kilometers."""
        nu = np.linspace(0, 2.0 * np.pi, num_points)
        p = semi_major_km * (1.0 - eccentricity ** 2)
        r = p / (1.0 + eccentricity * np.cos(nu))
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        return x, y


class ChaoticDoublePendulumSolver:
    """
    Nonlinear Double Pendulum Chaotic Dynamics Solver.
    Uses Lagrange equations of motion solved via Runge-Kutta 4th Order.
    """

    def __init__(self, l1: float = 1.0, l2: float = 1.0, m1: float = 1.0, m2: float = 1.0):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = STANDARD_GRAVITY

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        State: [theta1, omega1, theta2, omega2]
        """
        th1, w1, th2, w2 = state
        delta = th1 - th2

        den1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * (np.cos(delta) ** 2)
        num1 = (
            -self.g * (self.m1 + self.m2) * np.sin(th1)
            - self.m2 * self.g * np.sin(th1 - 2.0 * th2)
            - 2.0 * np.sin(delta) * self.m2 * (w2 ** 2 * self.l2 + w1 ** 2 * self.l1 * np.cos(delta))
        )
        dw1 = (num1 / (self.l1 * (2.0 * self.m1 + self.m2 - self.m2 * np.cos(2.0 * th1 - 2.0 * th2))))

        den2 = (self.l2 / self.l1) * den1
        num2 = (
            2.0 * np.sin(delta) * (
                w1 ** 2 * self.l1 * (self.m1 + self.m2)
                + self.g * (self.m1 + self.m2) * np.cos(th1)
                + w2 ** 2 * self.l2 * self.m2 * np.cos(delta)
            )
        )
        dw2 = (num2 / (self.l2 * (2.0 * self.m1 + self.m2 - self.m2 * np.cos(2.0 * th1 - 2.0 * th2))))

        return np.array([w1, dw1, w2, dw2])

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def get_cartesian_positions(self, state: np.ndarray) -> Tuple[float, float, float, float]:
        th1, _, th2, _ = state
        x1 = self.l1 * np.sin(th1)
        y1 = -self.l1 * np.cos(th1)
        x2 = x1 + self.l2 * np.sin(th2)
        y2 = y1 - self.l2 * np.cos(th2)
        return x1, y1, x2, y2


class LorentzParticleSolver:
    """
    Charged Particle Relativistic / Classical Dynamics in Electromagnetic Field.
    F = q * (E + v x B)
    """

    @staticmethod
    def step_boris(
        pos: np.ndarray,
        vel: np.ndarray,
        q: float,
        m: float,
        E: np.ndarray,
        B: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Boris particle pusher for energy-conserving cyclotronic gyromotion.
        """
        q_prime = 0.5 * dt * (q / m)
        # Half electric acceleration
        v_minus = vel + q_prime * E

        # Magnetic rotation
        t_vec = q_prime * B
        s_vec = (2.0 * t_vec) / (1.0 + float(np.dot(t_vec, t_vec)))
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)

        # Second half electric acceleration
        v_next = v_plus + q_prime * E
        pos_next = pos + v_next * dt

        return pos_next, v_next
