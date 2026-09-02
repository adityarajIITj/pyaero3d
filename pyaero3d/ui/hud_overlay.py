"""
PyAero3D - Production Glass Cockpit HUD, Digital Speedometer, Metric Options & Interactive User Guide.
"""

from typing import Dict, Any, Optional
import numpy as np
from panda3d.core import TextNode, NodePath
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectFrame import DirectFrame

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.quaternion_math import SpatialQuaternion


class FlightHUDOverlay:
    """
    Production-Grade Glass Cockpit HUD & Diagnostic Telemetry Overlay.
    Includes Digital Speedometer, Metric/Imperial toggles, Interactive Help Guide, and File controls.
    """

    def __init__(self, parent_node: Optional[NodePath] = None):
        self.use_metric = False  # False = Imperial (kt, ft, fpm), True = Metric (km/h, m, m/s)
        self.show_help = False
        self.show_menu = False

        # 1. Top Header Banner
        self.txt_title = OnscreenText(
            text="PYAERO3D // GENERAL-PURPOSE 3D MULTI-PHYSICS SIMULATOR",
            pos=(-0.95, 0.93),
            scale=0.040,
            fg=(1.0, 1.0, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.8),
            align=TextNode.ALeft,
            mayChange=False,
        )

        # 2. Interactive Navigation / Top Menu Bar
        self.txt_nav_bar = OnscreenText(
            text="[SPAWN: 1 Jet | 2 Drone | 3 Cargo | 4 Rocket | 5 Artillery | 6 Glider | 7 Satellite | 8 Sphere]  [CAM: Tab / Right-Click Orbit]",
            pos=(-0.95, 0.88),
            scale=0.028,
            fg=(0.55, 0.85, 1.0, 0.95),
            shadow=(0.0, 0.0, 0.0, 0.8),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # 2b. Live Scenario Header & Parameter Knobs Bar
        self.txt_scenario_header = OnscreenText(
            text="ACTIVE 3D PRESET: [SCENARIO]",
            pos=(-0.95, 0.83),
            scale=0.032,
            fg=(1.00, 0.85, 0.20, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ALeft,
            mayChange=True,
        )

        self.txt_param_bar = OnscreenText(
            text="[ [ / ] ] Mass (m) | [ - / = ] Drag (Cd) | [ ; / ' ] Thrust (T) | [ , / . ] Pitch Angle",
            pos=(-0.95, 0.78),
            scale=0.026,
            fg=(0.85, 0.90, 0.95, 0.9),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # 3. Top Heading Compass Ribbon (000° to 360°)
        self.txt_heading = OnscreenText(
            text="HDG: 360° [ N ]",
            pos=(0.0, 0.93),
            scale=0.042,
            fg=(0.10, 1.00, 0.50, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ACenter,
            mayChange=True,
        )

        # 4. Left Primary Flight Speedometer Gauge
        self.txt_left_speed = OnscreenText(
            text="IAS: 000 KT\nSPD: 0.0 m/s\nMACH: 0.00\nq: 0 Pa\nTHR: 0%\nG-LOAD: +1.0 G",
            pos=(-0.95, 0.25),
            scale=0.038,
            fg=(0.10, 1.00, 0.40, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # 5. Right Altimeter & Vertical Speed Tape
        self.txt_right_alt = OnscreenText(
            text="ALT MSL: 0 FT\nALT AGL: 0 FT\nVSI: +0 FPM\nPITCH: +0.0°\nROLL: +0.0°\nSTATUS: AIRBORNE",
            pos=(0.95, 0.25),
            scale=0.038,
            fg=(0.10, 1.00, 0.40, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ARight,
            mayChange=True,
        )

        # 6. Central Pitch Ladder / Artificial Horizon Reticle
        self.txt_pitch_ladder = OnscreenText(
            text="---[  +00  ]---",
            pos=(0.0, 0.0),
            scale=0.042,
            fg=(0.10, 1.00, 0.50, 0.75),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ACenter,
            mayChange=True,
        )

        # 7. Bottom Diagnostics & Engine Stats
        self.txt_bottom_diag = OnscreenText(
            text="PHYSICS: 1000.0 HZ | SOLVER: PGS 10 ITERS | GJK/EPA: ACTIVE | BODIES: 1",
            pos=(0.0, -0.92),
            scale=0.034,
            fg=(1.0, 0.90, 0.20, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            align=TextNode.ACenter,
            mayChange=True,
        )

        # 8. Interactive On-Screen Help Modal Dialog
        self.txt_help_modal = OnscreenText(
            text="",
            pos=(0.0, 0.35),
            scale=0.030,
            fg=(1.0, 1.0, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.95),
            align=TextNode.ACenter,
            mayChange=True,
        )

    def toggle_unit_system(self) -> bool:
        """Toggles between Metric (km/h, m, m/s) and Imperial (kt, ft, fpm) unit systems."""
        self.use_metric = not self.use_metric
        unit_str = "Metric" if self.use_metric else "Imperial"
        self.txt_nav_bar.setText(
            f"[FILE: F1 Reset | F2 Save]   [UNITS: U ({unit_str})]   [HELP: H]   [SPAWN: 1 Jet | 2 Drone | 3 Cargo | 4 Rocket]   [CAM: Tab]"
        )
        return self.use_metric

    def toggle_help_guide(self) -> bool:
        """Toggles on-screen interactive user guide modal."""
        self.show_help = not self.show_help
        if self.show_help:
            help_content = (
                "=================================================================================\n"
                "                         PYAERO3D USER GUIDE & FLIGHT REFERENCE MANUAL\n"
                "=================================================================================\n"
                "[W / S] Elevator Pitch Down / Pitch Up   |   [A / D] Aileron Roll Left / Roll Right\n"
                "[Q / E] Rudder Yaw Left / Yaw Right       |   [Shift / Ctrl] Throttle Up / Down\n"
                "[Space] Wheel Brakes / Coulomb Friction  |   [Tab] Cycle Camera (Chase / Cockpit / Orbit)\n"
                "[1] Spawn Jet   [2] Spawn Drone   [3] Cargo Drop   [4] Launch Rocket   [F] Kinetic Breakup\n"
                "[U] Toggle Metric/Imperial Units         |   [F1] Reset World   [F2] Quick-Save State\n"
                "---------------------------------------------------------------------------------\n"
                "PHYSICS: 1000Hz Leapfrog Integration | US 1976 Atmosphere | GJK/EPA Rigid Collisions\n"
                "=================================================================================\n"
                "                                  [ Press 'H' to Close Guide ]"
            )
            self.txt_help_modal.setText(help_content)
        else:
            self.txt_help_modal.setText("")
    def update(
        self,
        state_snapshot: np.ndarray,
        controlled_idx: int,
        ground_height_m: float,
        physics_hz: float,
        dt: float,
    ) -> None:
        """Standard update wrapper for HUD telemetry."""
        target_row = state_snapshot[controlled_idx] if (0 <= controlled_idx < state_snapshot.shape[0] and state_snapshot[controlled_idx, StateIdx.ACTIVE] > 0.5) else None
        active_count = int(np.sum(state_snapshot[:, StateIdx.ACTIVE] > 0.5))
        self.update_telemetry(
            state_row=target_row,
            ground_h=ground_height_m,
            camera_mode_name="CHASE",
            physics_hz=physics_hz,
            total_active=active_count,
        )

    def update_telemetry(
        self,
        state_row: Optional[Any],
        ground_h: float,
        camera_mode_name: str,
        physics_hz: float,
        total_active: int,
    ) -> None:
        """
        Updates on-screen speedometer gauges and altitude readouts in chosen unit system.
        """
        if state_row is None:
            self.txt_left_speed.setText("NO ACTIVE TARGET")
            self.txt_right_alt.setText("SPAWN: [1] JET  [2] DRONE  [3] CARGO  [4] ROCKET")
            return

        pos = state_row[StateIdx.PX:StateIdx.PZ + 1]
        vel = state_row[StateIdx.VX:StateIdx.VZ + 1]
        thr = state_row[StateIdx.THROTTLE]
        quat = state_row[StateIdx.QW:StateIdx.QZ + 1]
        ent_type = int(state_row[StateIdx.ENTITY_TYPE])

        speed_mps = float(np.linalg.norm(vel))
        mach = speed_mps / 340.29
        alt_msl_m = float(pos[1])
        alt_agl_m = max(0.0, alt_msl_m - ground_h)
        vsi_mps = float(vel[1])

        # Dynamic Pressure q = 0.5 * rho * v^2
        q_pa = 0.5 * 1.225 * (speed_mps ** 2)
        g_load = 1.0 + (vsi_mps / 9.81) * 0.05

        # Extract Euler Pitch & Roll from Quaternion DCM
        R = SpatialQuaternion.to_dcm(quat)
        pitch_deg = np.degrees(np.arcsin(-np.clip(R[1, 2], -1.0, 1.0)))
        roll_deg  = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
        heading_deg = np.degrees(np.arctan2(R[0, 2], R[2, 2])) % 360.0

        # Compass cardinal direction
        cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        c_idx = int((heading_deg + 22.5) / 45.0) % 8
        cardinal_str = cardinals[c_idx]

        self.txt_heading.setText(f"HDG: {int(heading_deg):03d}° [ {cardinal_str} ]")

        # Speedometer formatting (Metric vs Imperial)
        if self.use_metric:
            speed_val = speed_mps * 3.6  # km/h
            speed_unit = "KM/H"
            alt_msl_val = alt_msl_m
            alt_agl_val = alt_agl_m
            alt_unit = "M"
            vsi_val = vsi_mps
            vsi_unit = "M/S"
        else:
            speed_val = speed_mps * 1.94384  # Knots
            speed_unit = "KT"
            alt_msl_val = alt_msl_m * 3.28084  # Feet
            alt_agl_val = alt_agl_m * 3.28084
            alt_unit = "FT"
            vsi_val = vsi_mps * 196.85  # Feet per minute
            vsi_unit = "FPM"

        self.txt_left_speed.setText(
            f"IAS: {int(speed_val):03d} {speed_unit}\n"
            f"SPD: {speed_mps:.1f} m/s\n"
            f"MACH: {mach:.2f}\n"
            f"q: {int(q_pa):,d} Pa\n"
            f"THR: {int(thr * 100)}%\n"
            f"G-LOAD: {g_load:+.2f} G"
        )

        on_ground_str = "GROUND CONTACT" if state_row[StateIdx.ON_GROUND] > 0.5 else "AIRBORNE"
        self.txt_right_alt.setText(
            f"ALT MSL: {int(alt_msl_val):,d} {alt_unit}\n"
            f"ALT AGL: {int(alt_agl_val):,d} {alt_unit}\n"
            f"VSI: {int(vsi_val):+d} {vsi_unit}\n"
            f"PITCH: {pitch_deg:+.1f}°\n"
            f"ROLL: {roll_deg:+.1f}°\n"
            f"STATUS: {on_ground_str}"
        )

        pitch_sign = "+" if pitch_deg >= 0 else "-"
        abs_pitch = int(abs(pitch_deg))
        if ent_type in (EntityType.FIXED_WING_JET, EntityType.AIRFOIL_GLIDER):
            self.txt_pitch_ladder.setText(f"---[  {pitch_sign}{abs_pitch:02d}°  ]---")
        else:
            self.txt_pitch_ladder.setText("")

        type_names = {
            EntityType.FIXED_WING_JET: "FIGHTER JET",
            EntityType.QUADROTOR_DRONE: "QUADROTOR DRONE",
            EntityType.CARGO_PARACHUTE: "CARGO PARACHUTE",
            EntityType.MULTI_STAGE_ROCKET: "ROCKET LAUNCHER",
            EntityType.DEBRIS_FRAGMENT: "DEBRIS FRAGMENT",
        }
        type_str = type_names.get(ent_type, "VEHICLE")

        self.txt_bottom_diag.setText(
            f"VEHICLE: {type_str} | CAM: {camera_mode_name} | "
            f"PHYSICS: {physics_hz:.1f} HZ | SOLVER: PGS 10 ITERS | GJK/EPA: ACTIVE | BODIES: {total_active}"
        )
