"""
PyAero3D - Multi-Domain Aerospace Flight Dynamics Solver.
Solves 6-DOF aerodynamics for Fixed-Wing Jets, Quadrotor Drones, Cargo Parachutes, and Multi-Stage Rockets.
"""

from typing import Tuple, Dict, Any
import numpy as np

from pyaero3d.core.types import StateIdx, EntityType, STANDARD_GRAVITY
from pyaero3d.core.quaternion_math import SpatialQuaternion
from pyaero3d.physics.atmosphere import StandardAtmosphere


class FlightDynamicsSolver:
    """
    Unified 6-DOF Flight Dynamics & Propulsion Solver.
    """

    @staticmethod
    def evaluate_entity_dynamics(state_row: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates total forces (FX, FY, FZ in global coordinates) and body torques (TX, TY, TZ).
        """
        ent_type = int(state_row[StateIdx.ENTITY_TYPE])
        pos = state_row[StateIdx.PX:StateIdx.PZ + 1]
        vel = state_row[StateIdx.VX:StateIdx.VZ + 1]
        quat = state_row[StateIdx.QW:StateIdx.QZ + 1]
        omega = state_row[StateIdx.WX:StateIdx.WZ + 1]
        throttle = state_row[StateIdx.THROTTLE]
        mass = state_row[StateIdx.MASS]

        # Environmental atmosphere properties at current altitude (pos[1] = altitude Y)
        alt = max(0.0, float(pos[1]))
        _, _, rho, a_sound, _ = StandardAtmosphere.get_properties(alt)

        speed = float(np.linalg.norm(vel))
        q_dyn = 0.5 * rho * (speed ** 2)  # Dynamic pressure in Pa

        # Gravity force: Y-down in global frame
        f_gravity = np.array([0.0, -mass * STANDARD_GRAVITY, 0.0], dtype=np.float64)

        if ent_type == EntityType.FIXED_WING_JET:
            return FlightDynamicsSolver._solve_fixed_wing(state_row, vel, quat, omega, q_dyn, rho, f_gravity, dt)
        elif ent_type == EntityType.QUADROTOR_DRONE:
            return FlightDynamicsSolver._solve_quadrotor(state_row, vel, quat, omega, q_dyn, rho, f_gravity, dt)
        elif ent_type == EntityType.CARGO_PARACHUTE:
            return FlightDynamicsSolver._solve_cargo_drop(state_row, vel, quat, q_dyn, f_gravity, dt)
        elif ent_type == EntityType.MULTI_STAGE_ROCKET:
            return FlightDynamicsSolver._solve_rocket(state_row, vel, quat, omega, q_dyn, alt, f_gravity, dt)
        else:
            # Free particle / fragment debris
            area = max(0.01, state_row[StateIdx.AREA])
            cd = max(0.1, state_row[StateIdx.CD])
            f_drag = -0.5 * rho * speed * cd * area * vel if speed > 1e-4 else np.zeros(3)
            return f_gravity + f_drag, np.zeros(3)

    @staticmethod
    def _solve_fixed_wing(
        state_row: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        q_dyn: float,
        rho: float,
        f_grav: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fixed-Wing Jet 6-DOF aerodynamic forces and moments."""
        speed = float(np.linalg.norm(vel))

        # Fast inlined quaternion body transformation
        w, x, y, z = quat
        # Forward axis (+Z in body):
        fwd_x = 2.0 * (x * z + w * y)
        fwd_y = 2.0 * (y * z - w * x)
        fwd_z = 1.0 - 2.0 * (x * x + y * y)

        # Up axis (+Y in body):
        up_x = 2.0 * (x * y - w * z)
        up_y = 1.0 - 2.0 * (x * x + z * z)
        up_z = 2.0 * (y * z + w * x)

        # Right axis (+X in body):
        rgt_x = 1.0 - 2.0 * (y * y + z * z)
        rgt_y = 2.0 * (x * y + w * z)
        rgt_z = 2.0 * (x * z - w * y)

        # Forward, lateral, downward velocity components
        u = vel[0] * fwd_x + vel[1] * fwd_y + vel[2] * fwd_z
        v = vel[0] * rgt_x + vel[1] * rgt_y + vel[2] * rgt_z
        w_down = -(vel[0] * up_x + vel[1] * up_y + vel[2] * up_z)

        # Angle of Attack (alpha) and Sideslip Angle (beta)
        alpha = np.arctan2(w_down, max(u, 1.0))
        beta = np.arcsin(np.clip(v / max(speed, 1.0), -1.0, 1.0))

        # Aerodynamic parameters (High-performance jet)
        S = state_row[StateIdx.AREA] if state_row[StateIdx.AREA] > 0.1 else 28.0
        b = 10.5
        c_bar = 2.8
        AR = (b ** 2) / S
        e_oswald = 0.82

        # 1. Lift Coefficient with stall model: C_L(alpha)
        cl_alpha = 4.8
        cl0 = 0.15
        stall_angle = 0.27925 # 16 deg

        if abs(alpha) < stall_angle:
            cl = cl0 + cl_alpha * alpha
        else:
            cl = np.sign(alpha) * (cl0 + cl_alpha * stall_angle) * np.cos(abs(alpha) - stall_angle)

        elev = state_row[StateIdx.CTRL_ELEVATOR]
        cl += 0.45 * elev

        # 2. Drag Coefficient: C_D = C_D0 + C_L^2 / (pi * e * AR)
        cd0 = state_row[StateIdx.CD] if state_row[StateIdx.CD] > 0.01 else 0.024
        cd_induced = (cl ** 2) / (np.pi * e_oswald * AR + 1e-6)
        cd = cd0 + cd_induced

        # 3. Aerodynamic Forces in Wind/Stability Axis
        lift_mag = q_dyn * S * cl
        drag_mag = q_dyn * S * cd
        side_mag = -q_dyn * S * (0.35 * beta - 0.20 * state_row[StateIdx.CTRL_RUDDER])

        # Body forces
        fb_x = side_mag
        fb_y = lift_mag * np.cos(alpha) - drag_mag * np.sin(alpha)
        fb_z = -lift_mag * np.sin(alpha) - drag_mag * np.cos(alpha)

        # 4. Engine Jet Thrust (aligned with body +Z)
        throttle = state_row[StateIdx.THROTTLE]
        max_thrust = 85000.0
        fb_z += throttle * max_thrust

        # Transform body forces to global frame
        fg_x = fb_x * rgt_x + fb_y * up_x + fb_z * fwd_x
        fg_y = fb_x * rgt_y + fb_y * up_y + fb_z * fwd_y
        fg_z = fb_x * rgt_z + fb_y * up_z + fb_z * fwd_z

        f_tot_global = f_grav + np.array([fg_x, fg_y, fg_z], dtype=np.float64)

        # 5. Control Moments & Damping Torques
        ail = state_row[StateIdx.CTRL_AILERON]
        rud = state_row[StateIdx.CTRL_RUDDER]

        tau_pitch = q_dyn * S * c_bar * (-0.25 * alpha - 0.80 * elev - 2.5 * (omega[0] * c_bar / max(speed, 1.0)))
        tau_roll = q_dyn * S * b * (0.28 * ail - 0.08 * beta - 1.8 * (omega[2] * b / max(speed, 1.0)))
        tau_yaw = q_dyn * S * b * (0.18 * rud + 0.15 * beta - 1.2 * (omega[1] * b / max(speed, 1.0)))

        tau_body = np.array([tau_pitch, tau_yaw, tau_roll], dtype=np.float64)
        return f_tot_global, tau_body

    @staticmethod
    def _solve_quadrotor(
        state_row: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        q_dyn: float,
        rho: float,
        f_grav: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """6-DOF Quadrotor Drone Dynamics with motor thrust allocation and gyro torques."""
        R = SpatialQuaternion.to_dcm(quat)
        mass = state_row[StateIdx.MASS]

        # Hover thrust equilibrium: 4 motors total = mass * g
        throttle = max(0.0, min(1.0, state_row[StateIdx.THROTTLE]))
        max_total_thrust = mass * STANDARD_GRAVITY * 2.5 # 2.5:1 Thrust-to-weight ratio
        total_thrust = throttle * max_total_thrust

        # Thrust acts along body +Y axis (rotor plane)
        f_thrust_body = np.array([0.0, total_thrust, 0.0], dtype=np.float64)

        # Body aerodynamic drag
        speed = float(np.linalg.norm(vel))
        cd_drone = 0.45
        area_drone = 0.08
        f_drag_global = -0.5 * rho * speed * cd_drone * area_drone * vel if speed > 1e-4 else np.zeros(3)

        f_tot_global = f_grav + f_drag_global + (R @ f_thrust_body)

        # Control Torques (Attitude commands)
        pitch_cmd = state_row[StateIdx.CTRL_ELEVATOR]
        roll_cmd = state_row[StateIdx.CTRL_AILERON]
        yaw_cmd = state_row[StateIdx.CTRL_RUDDER]

        # Proportional-Derivative Attitude Control Moments
        arm_len = 0.18 # 18cm motor arm length
        k_p = 4.5
        k_d = 0.35

        tau_x = (pitch_cmd * k_p - omega[0] * k_d) * mass * arm_len
        tau_z = (roll_cmd * k_p - omega[2] * k_d) * mass * arm_len
        tau_y = (yaw_cmd * 2.0 - omega[1] * 0.25) * mass * arm_len

        # Rotor gyroscopic precession torque
        i_rotor = 1.2e-5
        w_rotor_avg = np.sqrt(total_thrust / (4.0 * 1.5e-6 + 1e-9))
        tau_gyro = np.array([
            i_rotor * w_rotor_avg * omega[2],
            0.0,
            -i_rotor * w_rotor_avg * omega[0],
        ])

        tau_body = np.array([tau_x, tau_y, tau_z], dtype=np.float64) + tau_gyro
        return f_tot_global, tau_body

    @staticmethod
    def _solve_cargo_drop(
        state_row: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        q_dyn: float,
        f_grav: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cargo parachute descent dynamics."""
        speed = float(np.linalg.norm(vel))
        area = max(1.0, state_row[StateIdx.AREA])
        cd = 1.75 # High drag parachute canopy

        f_drag = -q_dyn * area * cd * (vel / max(speed, 1e-3)) if speed > 1e-4 else np.zeros(3)
        f_tot = f_grav + f_drag
        tau = -0.5 * state_row[StateIdx.WX:StateIdx.WZ + 1] # Rotational damping
        return f_tot, tau

    @staticmethod
    def _solve_rocket(
        state_row: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        q_dyn: float,
        alt: float,
        f_grav: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-stage rocket dynamics with altitude-dependent thrust and propellant depletion."""
        R = SpatialQuaternion.to_dcm(quat)
        throttle = state_row[StateIdx.THROTTLE]
        fuel = state_row[StateIdx.FUEL_MASS]

        thrust_sl = 320000.0 # 320 kN Sea-Level thrust
        thrust_vac = 380000.0 # 380 kN Vacuum thrust
        isp_sl = 285.0
        isp_vac = 320.0

        # Altitude expansion factor (0.0 SL -> 1.0 Vacuum)
        alt_factor = np.clip(alt / 45000.0, 0.0, 1.0)
        thrust_curr = (thrust_sl * (1.0 - alt_factor) + thrust_vac * alt_factor) * throttle
        isp_curr = isp_sl * (1.0 - alt_factor) + isp_vac * alt_factor

        if fuel > 0.0 and throttle > 0.01:
            # Mass burn: m_dot = Thrust / (g0 * Isp)
            m_dot = thrust_curr / (STANDARD_GRAVITY * isp_curr)
            dm = min(fuel, m_dot * dt)
            state_row[StateIdx.FUEL_MASS] -= dm
            state_row[StateIdx.MASS] = max(500.0, state_row[StateIdx.MASS] - dm)
        else:
            thrust_curr = 0.0

        # Thrust Vector Control (TVC) gimbal deflection
        gimbal_y = state_row[StateIdx.CTRL_ELEVATOR] * np.radians(6.0)
        gimbal_x = state_row[StateIdx.CTRL_AILERON] * np.radians(6.0)

        # Rocket body +Y is long axis
        f_thrust_body = np.array([
            thrust_curr * np.sin(gimbal_x),
            thrust_curr * np.cos(gimbal_x) * np.cos(gimbal_y),
            thrust_curr * np.sin(gimbal_y),
        ])

        # Aerodynamic supersonic wave drag
        speed = float(np.linalg.norm(vel))
        area = state_row[StateIdx.AREA] if state_row[StateIdx.AREA] > 0.1 else 2.5
        cd_rocket = 0.28
        f_drag = -q_dyn * area * cd_rocket * (vel / max(speed, 1e-3)) if speed > 1e-4 else np.zeros(3)

        f_tot = f_grav + f_drag + (R @ f_thrust_body)

        # TVC Gimbal Torque
        engine_arm = 6.5 # 6.5m distance from CG to engine nozzle
        tau_body = np.array([
            thrust_curr * np.sin(gimbal_y) * engine_arm,
            0.0,
            -thrust_curr * np.sin(gimbal_x) * engine_arm,
        ]) - 0.4 * omega # Aero damping

        return f_tot, tau_body
