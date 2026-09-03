"""
PyAero3D - 3D Panda3D Mountain Simulation & Interactive Multi-Scenario Sandbox Application.
Integrates Free View 3D Camera, Real-Time Articulated Physical Solvers, In-Viewport Controls, and Trajectory Ribbons.
"""

import sys
import numpy as np
from typing import Optional, Dict, Any, List

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    load_prc_file_data,
    Vec3,
    WindowProperties,
    DirectionalLight,
    AmbientLight,
    Fog,
    NodePath,
    LQuaternionf,
    Point3,
)

# Panda3D Window & Graphics Configuration
load_prc_file_data("", """
    window-title PyAero3D // Interactive 3D Multi-Physics & Aerospace Sandbox
    win-size 1600 900
    sync-video #t
    show-frame-rate-meter #t
    textures-power-2 none
    gl-version 3 2
""")

from pyaero3d.core.types import StateIdx, EntityType, STANDARD_GRAVITY
from pyaero3d.core.state import StateBuffer
from pyaero3d.render.terrain_gen import MountainTerrainGenerator
from pyaero3d.render.sky_dome import EnvironmentGeometryBuilder
from pyaero3d.render.vehicle_models import VehicleModelBuilder
from pyaero3d.render.mesh_primitives import MeshPrimitiveBuilder
from pyaero3d.render.camera_controller import FlightCameraController, CameraMode
from pyaero3d.render.spatial_references import SpatialReferenceBuilder, Dynamic3DTrajectoryRibbon
from pyaero3d.ui.hud_overlay import FlightHUDOverlay
from pyaero3d.ui.control_panel_3d import InViewportControlPanel3D
from pyaero3d.controls.flight_yoke import FlightYokeController
from pyaero3d.controls.god_hand import GodHandController
from pyaero3d.physics.engine_thread import PhysicsEngineThread
from pyaero3d.physics.fragmentation import FragmentationEngine
from pyaero3d.physics.mountain_collision import MountainCollisionEngine
from pyaero3d.physics.advanced_solvers import (
    ChaoticDoublePendulumSolver,
    LorentzParticleSolver,
    OrbitalMechanicsSolver,
)


class PyAero3DSimulatorApp(ShowBase):
    """
    Main 3D Desktop Simulation Application orchestrating Physics, Rendering, and Controls.
    """

    def __init__(
        self,
        scenario_idx: int = 1,
        v0: Optional[float] = None,
        theta: Optional[float] = None,
        mass: Optional[float] = None,
        cd: Optional[float] = None,
        area: Optional[float] = None,
        wind: Optional[float] = None,
        thrust: Optional[float] = None,
    ):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.05, 0.05, 0.08, 1.0)

        # Scenario & Physical Parameters
        self.scenario_idx = scenario_idx
        self.curr_v0 = v0 if v0 is not None else 220.0
        self.curr_theta = theta if theta is not None else (45.0 if scenario_idx == 0 else (60.0 if scenario_idx == 5 else 5.0))
        self.curr_mass = mass if mass is not None else (15.0 if scenario_idx == 0 else (12000.0 if scenario_idx == 1 else 10.0))
        self.curr_cd = cd if cd is not None else (0.30 if scenario_idx == 0 else 0.024)
        self.curr_area = area if area is not None else 28.0
        self.curr_wind = wind if wind is not None else 0.0
        self.curr_thrust = thrust if thrust is not None else (85000.0 if scenario_idx == 1 else 0.0)

        self.is_paused = False

        # Clean Flat Grey Studio Canvas World
        self.setBackgroundColor(0.12, 0.13, 0.16, 1.0)

        # Configure Camera Lens FOV
        self.camLens.setFov(75)
        self.camLens.setNearFar(0.5, 40000.0)

        # Subtle Studio Depth Cue
        self.fog = Fog("StudioDepthCue")
        self.fog.setColor(0.12, 0.13, 0.16)
        self.fog.setLinearRange(2500.0, 25000.0)
        self.render.setFog(self.fog)

        # 1. State Buffer Tensor
        self.state_buffer = StateBuffer(max_entities=256)

        # 2. Continuous Flat Physics World
        print("[PyAero3D] Initializing Flat Grey Canvas World...")
        self.terrain_gen = MountainTerrainGenerator(world_size_m=25000.0, is_flat=True)

        # Flat Grey Floor Canvas
        self.floor_canvas = SpatialReferenceBuilder.create_flat_canvas(size=30000.0, color=(0.17, 0.18, 0.21, 1.0))
        self.floor_canvas.reparentTo(self.render)
        self.floor_canvas.setPos(0.0, 0.0, 0.0)

        # 3. Clean Infinite CAD Metric Grid (Dual-tone major/minor grid lines)
        self.grid_np = SpatialReferenceBuilder.create_cad_grid(size=8000.0, major_step=100.0, minor_step=20.0, elevation=0.02)
        self.grid_np.reparentTo(self.render)
        self.grid_np.setPos(0.0, 0.0, 0.0)

        # Origin Cartesian Coordinate Axes (Red=X, Green=Y Up, Blue=Z Forward)
        self.axes_np = SpatialReferenceBuilder.create_coordinate_axes(length=18.0)
        self.axes_np.reparentTo(self.render)
        self.axes_np.setPos(0.0, 0.0, 0.04)

        self.trajectory_ribbon = Dynamic3DTrajectoryRibbon(self.render, max_points=700, color=(0.15, 0.85, 1.0, 0.95))

        # 4. High-Contrast Studio Lighting
        self._setup_lighting()

        # 5. Collision & Fragmentation Engines (Flat World Ground Response)
        self.collision_engine = MountainCollisionEngine(self.terrain_gen)
        self.fragmentation_engine = FragmentationEngine

        # 6. 1000Hz Background Physics Engine Thread
        self.physics_thread = PhysicsEngineThread(
            state_buffer=self.state_buffer,
            collision_engine=self.collision_engine,
            target_hz=1000.0,
        )

        # 7. Dynamic Camera Controller (Supports True Chase & Side Profile Trajectory)
        self.cam_controller = FlightCameraController(self.camera, base_app=self)

        # 8. Glass Cockpit Telemetry HUD & Input Yoke
        self.hud = FlightHUDOverlay()
        self.yoke = FlightYokeController(self, self.state_buffer)

        # 9. Visual Actor Nodes Mapping
        self.actor_nodes: Dict[int, NodePath] = {}
        self.scenario_props: List[NodePath] = []
        self.current_controlled_idx = 0

        # Dedicated Scenario Physical Solvers
        self.pendulum_solver = ChaoticDoublePendulumSolver(l1=3.5, l2=3.5, m1=self.curr_mass, m2=self.curr_mass)
        self.pendulum_state = np.array([np.radians(60.0), 0.0, np.radians(90.0), 0.0], dtype=np.float64)
        self.pendulum_nodes: Dict[str, NodePath] = {}
        self.pendulum_pivot_p3 = Point3(0.0, 0.0, 16.0)

        # Spring Oscillator state
        self.spring_y = 0.0
        self.spring_vy = 0.0
        self.spring_mesh_np: Optional[NodePath] = None
        self.spring_mass_np: Optional[NodePath] = None

        # 10. Interactive 3D "God Hand" Mouse Manipulation
        self.god_hand = GodHandController(self, self.state_buffer, self.render)

        # 11. Interactive In-Viewport 3D Control Panel
        self.control_panel = InViewportControlPanel3D(
            base_app=self,
            on_change_scenario=self.switch_scenario,
            on_tweak_mass=self.tweak_mass,
            on_tweak_cd=self.tweak_cd,
            on_tweak_thrust=self.tweak_thrust,
            on_tweak_angle=self.tweak_angle,
            on_launch_reset=self.launch_or_reset,
            on_toggle_pause=self.toggle_pause,
            on_step_physics=self.step_physics,
            on_change_cam_mode=self.set_camera_mode,
            on_toggle_axes=self.toggle_axes,
            on_toggle_grid=self.toggle_grid,
            on_toggle_trail=self.toggle_trail,
            on_spawn_object=self.spawn_sandbox_object,
        )

        # Bind parameter keyboard hotkeys
        self._bind_parameter_hotkeys()

        # Start Physics Engine Thread
        self.physics_thread.start()

        # Spawn Initial Active Scenario Preset
        self._spawn_preset_scenario(self.scenario_idx)

        # Register Main Render Frame Task
        self.taskMgr.add(self._render_frame_update, "PyAero3D_RenderUpdate")

    def _setup_lighting(self) -> None:
        """Sets up high-contrast 3-point studio lighting for clean CAD visualization."""
        # Key Light (Upper front-left)
        dlight = DirectionalLight("KeyLight")
        dlight.setColor((0.95, 0.95, 0.98, 1.0))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -55, 0)
        self.render.setLight(dlnp)

        # Fill Light (Soft right)
        fill = DirectionalLight("FillLight")
        fill.setColor((0.35, 0.40, 0.50, 1.0))
        fillnp = self.render.attachNewNode(fill)
        fillnp.setHpr(135, -35, 0)
        self.render.setLight(fillnp)

        # Ambient Studio Light
        alight = AmbientLight("AmbientStudio")
        alight.setColor((0.32, 0.34, 0.38, 1.0))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

    def _bind_parameter_hotkeys(self) -> None:
        """Binds instant keyboard shortcuts for camera, physical tuning, and sandbox spawning."""
        self.accept("[", lambda: self.tweak_mass(-0.20))
        self.accept("]", lambda: self.tweak_mass(0.20))
        self.accept("-", lambda: self.tweak_cd(-0.05))
        self.accept("=", lambda: self.tweak_cd(0.05))
        self.accept(";", lambda: self.tweak_thrust(-5000.0))
        self.accept("'", lambda: self.tweak_thrust(5000.0))
        self.accept(",", lambda: self.tweak_angle(-5.0))
        self.accept(".", lambda: self.tweak_angle(5.0))
        self.accept("r", self.launch_or_reset)
        self.accept("p", self.toggle_pause)
        self.accept("c", self._cycle_camera_mode)
        self.accept("1", lambda: self.set_camera_mode(0))  # Chase Behind
        self.accept("2", lambda: self.set_camera_mode(1))  # Side Profile
        self.accept("3", lambda: self.set_camera_mode(2))  # Free View
        self.accept("4", lambda: self.set_camera_mode(3))  # Orbit Target
        self.accept("5", lambda: self.set_camera_mode(4))  # Cockpit
        self.accept("g", lambda: self.spawn_sandbox_object(0)) # Spawn Crate
        self.accept("tab", self.control_panel.toggle_panel_visibility)
        self.accept("f", lambda: self.cam_controller.focus_target(self._get_target_pos()))

    def _get_target_pos(self) -> np.ndarray:
        if self.scenario_idx == 5:
            return np.array([0.0, 16.0, 0.0])
        elif self.scenario_idx == 7:
            return np.array([0.0, 10.0, 0.0])

        if 0 <= self.current_controlled_idx < self.state_buffer.max_entities:
            if self.state_buffer.data[self.current_controlled_idx, StateIdx.ACTIVE] > 0.5:
                pos = self.state_buffer.data[self.current_controlled_idx, StateIdx.PX:StateIdx.PZ + 1]
                if not (np.isnan(pos).any() or np.isinf(pos).any()):
                    return pos
        return np.array([0.0, 10.0, 0.0])

    def tweak_mass(self, delta_pct: float) -> None:
        self.curr_mass = max(0.1, self.curr_mass * (1.0 + delta_pct))
        if self.scenario_idx == 5:
            self.pendulum_solver.m1 = self.curr_mass
            self.pendulum_solver.m2 = self.curr_mass
        if 0 <= self.current_controlled_idx < self.state_buffer.max_entities:
            self.state_buffer.data[self.current_controlled_idx, StateIdx.MASS] = self.curr_mass
        self._update_panel_readouts()

    def tweak_cd(self, delta: float) -> None:
        self.curr_cd = max(0.005, min(2.5, self.curr_cd + delta))
        if 0 <= self.current_controlled_idx < self.state_buffer.max_entities:
            self.state_buffer.data[self.current_controlled_idx, StateIdx.CD] = self.curr_cd
        self._update_panel_readouts()

    def tweak_thrust(self, delta: float) -> None:
        self.curr_thrust = max(0.0, self.curr_thrust + delta)
        self._update_panel_readouts()

    def tweak_angle(self, delta_deg: float) -> None:
        self.curr_theta = np.clip(self.curr_theta + delta_deg, 0.0, 90.0)
        if self.scenario_idx == 5:
            self.pendulum_state = np.array([np.radians(self.curr_theta), 0.0, np.radians(self.curr_theta + 30.0), 0.0], dtype=np.float64)
        self._update_panel_readouts()

    def _update_panel_readouts(self) -> None:
        self.control_panel.update_parameter_readouts(self.curr_mass, self.curr_cd, self.curr_thrust, self.curr_theta)

    def switch_scenario(self, scenario_idx: int) -> None:
        self.scenario_idx = scenario_idx
        self.launch_or_reset()

    def launch_or_reset(self) -> None:
        print(f"[PyAero3D] Re-launching Scenario Preset #{self.scenario_idx}...")
        self.trajectory_ribbon.clear()
        for node in self.actor_nodes.values():
            node.removeNode()
        self.actor_nodes.clear()
        for prop in self.scenario_props:
            prop.removeNode()
        self.scenario_props.clear()
        for pn in self.pendulum_nodes.values():
            pn.removeNode()
        self.pendulum_nodes.clear()
        if self.spring_mesh_np: self.spring_mesh_np.removeNode(); self.spring_mesh_np = None
        if self.spring_mass_np: self.spring_mass_np.removeNode(); self.spring_mass_np = None
        if hasattr(self, "god_hand"): self.god_hand._on_grab_end()

        self.state_buffer.clear()
        self._spawn_preset_scenario(self.scenario_idx)
        self._update_panel_readouts()

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.physics_thread.stop()
            print("[PyAero3D] Physics Paused.")
        else:
            self.physics_thread.start()
            print("[PyAero3D] Physics Resumed.")

    def step_physics(self) -> None:
        if self.is_paused:
            self.physics_thread.engine.step(0.01)
            if self.scenario_idx == 5:
                self.pendulum_state = self.pendulum_solver.rk4_step(self.pendulum_state, 0.01)

    def set_camera_mode(self, mode_int: int) -> None:
        self.cam_controller.set_mode(CameraMode(mode_int))
        print(f"[PyAero3D] Camera Mode set to: {CameraMode(mode_int).name}")

    def _cycle_camera_mode(self) -> None:
        new_mode = self.cam_controller.cycle_mode()
        print(f"[PyAero3D] Camera Mode cycled to: {new_mode.name}")

    def toggle_axes(self) -> None:
        if self.axes_np.isHidden(): self.axes_np.show()
        else: self.axes_np.hide()

    def toggle_grid(self) -> None:
        if self.grid_np.isHidden(): self.grid_np.show()
        else: self.grid_np.hide()

    def toggle_trail(self) -> None:
        if self.trajectory_ribbon.trail_np and not self.trajectory_ribbon.trail_np.isHidden():
            self.trajectory_ribbon.trail_np.hide()
        elif self.trajectory_ribbon.trail_np:
            self.trajectory_ribbon.trail_np.show()

    def _spawn_preset_scenario(self, idx: int) -> int:
        """Spawns the exact 3D physical counterpart of the selected Graph Studio preset."""
        scenario_names = [
            "0: Ballistics Cannon Shell",
            "1: 6-DOF Fighter Jet",
            "2: NACA Aerodynamic Glider",
            "3: Multi-Stage Space Rocket",
            "4: Keplerian Orbital Satellite",
            "5: Chaotic Double Pendulum",
            "6: Lorentz Particle Cyclotron",
            "7: Viscoelastic Spring-Damper",
        ]
        scen_name = scenario_names[idx] if idx < len(scenario_names) else f"Preset #{idx}"
        self.hud.txt_scenario_header.setText(f"ACTIVE 3D PRESET: {scen_name.upper()}")

        if idx == 0:  # Ballistics Cannon Projectile
            self.curr_mass = 15.0; self.curr_cd = 0.30; self.curr_theta = 45.0
            return self.spawn_cannon_projectile()
        elif idx == 1:  # Fighter Jet 6-DOF
            self.curr_mass = 12000.0; self.curr_cd = 0.024; self.curr_thrust = 85000.0; self.curr_theta = 5.0
            return self.spawn_fighter_jet_airborne()
        elif idx == 2:  # NACA Airfoil Glider
            self.curr_mass = 650.0; self.curr_cd = 0.018; self.curr_theta = 3.0
            return self.spawn_airfoil_glider()
        elif idx == 3:  # Multi-Stage Rocket Launch
            self.curr_mass = 8500.0; self.curr_cd = 0.20; self.curr_thrust = 140000.0; self.curr_theta = 88.0
            return self.spawn_rocket_launch()
        elif idx == 4:  # Orbital Satellite
            self.curr_mass = 2400.0; self.curr_cd = 0.01; self.curr_thrust = 5000.0
            return self.spawn_orbital_satellite()
        elif idx == 5:  # Double Pendulum
            self.curr_mass = 10.0; self.curr_cd = 0.05; self.curr_theta = 60.0
            return self.spawn_double_pendulum()
        elif idx == 6:  # Lorentz Particle Cyclotron
            self.curr_mass = 1.0; self.curr_cd = 0.0
            return self.spawn_cyclotron_particle()
        elif idx == 7:  # Spring-Damper Oscillator
            self.curr_mass = 25.0; self.curr_cd = 0.45
            return self.spawn_spring_damper_system()
        else:
            return self.spawn_fighter_jet_airborne()

    def spawn_fighter_jet_airborne(self) -> int:
        theta_rad = np.radians(self.curr_theta)
        init_pos = np.array([0.0, 150.0, 0.0], dtype=np.float64)
        init_vel = np.array([0.0, self.curr_v0 * np.sin(theta_rad), self.curr_v0 * np.cos(theta_rad)], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.FIXED_WING_JET,
            mass=self.curr_mass,
            position=init_pos,
            velocity=init_vel,
            radius=8.5,
            cd=self.curr_cd,
            area=self.curr_area,
        )
        self.state_buffer.data[idx, StateIdx.THROTTLE] = 0.85

        actor_np = VehicleModelBuilder.create_fighter_jet()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        self.cam_controller.set_target_scale(8.5)
        self.cam_controller.focus_target(init_pos)
        print(f"[PyAero3D] Spawned 6-DOF Fighter Jet (Entity #{idx}).")
        return idx

    def spawn_cannon_projectile(self) -> int:
        theta_rad = np.radians(self.curr_theta)
        init_pos = np.array([0.0, 2.0, 0.0], dtype=np.float64)
        init_vel = np.array([0.0, self.curr_v0 * np.sin(theta_rad), self.curr_v0 * np.cos(theta_rad)], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.CANNON_PROJECTILE,
            mass=self.curr_mass,
            position=init_pos,
            velocity=init_vel,
            radius=1.2,
            cd=self.curr_cd,
            area=0.08,
        )

        actor_np = VehicleModelBuilder.create_cannon_projectile()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        self.cam_controller.set_target_scale(2.0)
        self.cam_controller.focus_target(init_pos)
        print(f"[PyAero3D] Fired Ballistic Artillery Shell (Entity #{idx}).")
        return idx

    def spawn_airfoil_glider(self) -> int:
        init_pos = np.array([0.0, 200.0, 0.0], dtype=np.float64)
        init_vel = np.array([0.0, -1.5, self.curr_v0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.AIRFOIL_GLIDER,
            mass=self.curr_mass,
            position=init_pos,
            velocity=init_vel,
            radius=7.5,
            cd=self.curr_cd,
            area=16.0,
        )

        actor_np = VehicleModelBuilder.create_airfoil_wing()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        self.cam_controller.set_target_scale(7.5)
        self.cam_controller.focus_target(init_pos)
        print(f"[PyAero3D] Spawned NACA Aerodynamic Glider (Entity #{idx}).")
        return idx

    def spawn_rocket_launch(self) -> int:
        init_pos = np.array([0.0, 2.0, 0.0], dtype=np.float64)
        init_vel = np.array([0.0, 5.0, 0.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.MULTI_STAGE_ROCKET,
            mass=self.curr_mass,
            position=init_pos,
            velocity=init_vel,
            radius=2.5,
            cd=self.curr_cd,
            area=4.5,
        )
        self.state_buffer.data[idx, StateIdx.THROTTLE] = 1.0

        actor_np = VehicleModelBuilder.create_rocket()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        self.cam_controller.set_target_scale(8.0)
        self.cam_controller.focus_target(init_pos)
        print(f"[PyAero3D] Initiated Multi-Stage Rocket Liftoff (Entity #{idx}).")
        return idx

    def spawn_orbital_satellite(self) -> int:
        init_pos = np.array([0.0, 1500.0, 0.0], dtype=np.float64)
        init_vel = np.array([0.0, 0.0, 180.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.ORBITAL_SATELLITE,
            mass=self.curr_mass,
            position=init_pos,
            velocity=init_vel,
            radius=3.5,
            cd=self.curr_cd,
            area=8.0,
        )

        actor_np = VehicleModelBuilder.create_satellite()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        self.cam_controller.set_target_scale(5.0)
        self.cam_controller.focus_target(init_pos)
        print(f"[PyAero3D] Spawned Orbital Satellite (Entity #{idx}).")
        return idx

    def spawn_double_pendulum(self) -> int:
        pivot_pos = np.array([0.0, 16.0, 0.0])

        # Floor Mounting Stand
        stand_np = SpatialReferenceBuilder.create_pendulum_stand()
        stand_np.reparentTo(self.render)
        stand_np.setPos(0.0, 0.0, 0.0)
        self.scenario_props.append(stand_np)

        # Pivot Bearing Housing
        pivot_housing = MeshPrimitiveBuilder.build_cylinder(
            radius=0.25, length=0.6, segments=24, color=(0.95, 0.75, 0.20, 1.0), name="PivotHousing"
        )
        pivot_housing.reparentTo(self.render)
        pivot_housing.setPos(0.0, -0.3, 16.0)
        self.scenario_props.append(pivot_housing)

        # Rod 1 & Bob 1
        rod1 = MeshPrimitiveBuilder.build_cylinder(
            radius=0.07, length=3.5, segments=24, color=(0.85, 0.88, 0.92, 1.0), name="Rod1"
        )
        rod1.reparentTo(self.render)
        self.pendulum_nodes["rod1"] = rod1

        bob1 = MeshPrimitiveBuilder.build_uv_sphere(
            radius=0.55, rings=24, sectors=32, color=(0.95, 0.65, 0.15, 1.0), name="Bob1"
        )
        bob1.reparentTo(self.render)
        self.pendulum_nodes["bob1"] = bob1

        # Rod 2 & Bob 2
        rod2 = MeshPrimitiveBuilder.build_cylinder(
            radius=0.07, length=3.5, segments=24, color=(0.75, 0.80, 0.85, 1.0), name="Rod2"
        )
        rod2.reparentTo(self.render)
        self.pendulum_nodes["rod2"] = rod2

        bob2 = MeshPrimitiveBuilder.build_uv_sphere(
            radius=0.65, rings=24, sectors=32, color=(0.15, 0.85, 1.0, 1.0), name="Bob2"
        )
        bob2.reparentTo(self.render)
        self.pendulum_nodes["bob2"] = bob2

        self.pendulum_solver = ChaoticDoublePendulumSolver(l1=3.5, l2=3.5, m1=self.curr_mass, m2=self.curr_mass)
        self.pendulum_state = np.array([np.radians(self.curr_theta), 0.0, np.radians(self.curr_theta + 30.0), 0.0], dtype=np.float64)

        self.cam_controller.set_target_scale(4.5)
        self.cam_controller.focus_target(pivot_pos)
        print(f"[PyAero3D] Initialized 3D Articulated Double Pendulum at ({pivot_pos[0]}, {pivot_pos[1]}, {pivot_pos[2]}).")
        return 0

    def spawn_cyclotron_particle(self) -> int:
        init_pos = np.array([0.0, 18.0, 0.0], dtype=np.float64)
        init_vel = np.array([50.0, 20.0, 0.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.LORENTZ_PARTICLE,
            mass=self.curr_mass,
            position=init_pos,
            velocity=init_vel,
            radius=2.5,
            cd=0.0,
            area=0.1,
        )

        actor_np = VehicleModelBuilder.create_cyclotron_chamber()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        self.cam_controller.set_target_scale(3.5)
        self.cam_controller.focus_target(init_pos)
        print(f"[PyAero3D] Spawned 3D Lorentz Particle Cyclotron (Entity #{idx}).")
        return idx

    def spawn_spring_damper_system(self) -> int:
        """Spawns 3D viscoelastic harmonic oscillator with dynamic coil spring and weight."""
        mount_pos = np.array([0.0, 16.0, 0.0])

        # Floor mounting frame
        stand_np = SpatialReferenceBuilder.create_pendulum_stand()
        stand_np.reparentTo(self.render)
        stand_np.setPos(0.0, 0.0, 0.0)
        self.scenario_props.append(stand_np)

        # Helical Spring
        self.spring_mesh_np = MeshPrimitiveBuilder.build_helical_spring(
            radius=0.9, length=6.0, coils=9, wire_radius=0.09, color=(0.95, 0.75, 0.15, 1.0)
        )
        self.spring_mesh_np.reparentTo(self.render)
        self.spring_mesh_np.setPos(0.0, 0.0, 13.0)

        # Mass block
        self.spring_mass_np = MeshPrimitiveBuilder.build_uv_sphere(
            radius=1.4, rings=18, sectors=24, color=(0.20, 0.65, 0.95, 1.0), name="SpringMass"
        )
        self.spring_mass_np.reparentTo(self.render)
        self.spring_mass_np.setPos(0.0, 0.0, 9.0)

        self.spring_y = -2.5
        self.spring_vy = 0.0
        self.cam_controller.set_target_scale(4.0)
        self.cam_controller.focus_target(mount_pos)
        print(f"[PyAero3D] Spawned 3D Spring-Damper Oscillator.")
        return 0

    def spawn_sandbox_object(self, s_type: int) -> int:
        """Drops physical sandbox objects into the world with real-time terrain collision."""
        target_pos = self._get_target_pos()
        spawn_phys = target_pos + np.array([float(np.random.uniform(-15.0, 15.0)), 28.0, float(np.random.uniform(-15.0, 15.0))])
        gh = float(self.terrain_gen.get_height(spawn_phys[0], spawn_phys[2]))
        spawn_phys[1] = max(spawn_phys[1], gh + 22.0)

        if s_type == 0:  # Physics Cargo Crate
            idx = self.state_buffer.allocate_entity(
                entity_type=EntityType.DEBRIS_FRAGMENT,
                mass=45.0,
                position=spawn_phys,
                velocity=np.array([float(np.random.uniform(-3, 3)), -2.0, float(np.random.uniform(-3, 3))]),
                radius=1.6,
                cd=0.85,
                area=3.0,
            )
            actor = VehicleModelBuilder.create_physics_crate(size=2.2)
        elif s_type == 1:  # Viscoelastic Bouncing Sphere
            idx = self.state_buffer.allocate_entity(
                entity_type=EntityType.BOUNCING_SPHERE,
                mass=20.0,
                position=spawn_phys,
                velocity=np.array([float(np.random.uniform(-4, 4)), 2.0, float(np.random.uniform(-4, 4))]),
                radius=1.2,
                cd=0.45,
                area=1.4,
            )
            actor = VehicleModelBuilder.create_bouncing_sphere()
        elif s_type == 2:  # Target Quadrotor Drone
            idx = self.state_buffer.allocate_entity(
                entity_type=EntityType.QUADROTOR_DRONE,
                mass=12.0,
                position=spawn_phys,
                velocity=np.array([0.0, 0.0, 10.0]),
                radius=2.0,
                cd=0.25,
                area=0.8,
            )
            actor = VehicleModelBuilder.create_quadrotor_drone()
        elif s_type == 3:  # Artillery Cannon Shell
            idx = self.state_buffer.allocate_entity(
                entity_type=EntityType.CANNON_PROJECTILE,
                mass=18.0,
                position=spawn_phys,
                velocity=np.array([0.0, 35.0, 95.0]),
                radius=0.9,
                cd=0.28,
                area=0.08,
            )
            actor = VehicleModelBuilder.create_cannon_projectile()
        else:
            return -1

        actor.reparentTo(self.render)
        self.actor_nodes[idx] = actor
        print(f"[PyAero3D] Spawned Sandbox Object #{s_type} (Entity #{idx}) into 3D world.")
        return idx

    def _render_frame_update(self, task):
        dt = globalClock.getDt()
        if dt > 0.1: dt = 0.016  # Clamp timestep spikes

        # Update flight yoke controls
        self.yoke.update(dt)

        # Update 3D God Hand interactive mouse spring pulling
        self.god_hand.update(dt)

        # 1. Update Double Pendulum 3D Physical Articulation
        if self.scenario_idx == 5 and len(self.pendulum_nodes) == 4:
            if not self.is_paused:
                # RK4 integration substeps for high accuracy
                for _ in range(4):
                    self.pendulum_state = self.pendulum_solver.rk4_step(self.pendulum_state, dt * 0.25)

            th1, _, th2, _ = self.pendulum_state
            l1 = self.pendulum_solver.l1
            l2 = self.pendulum_solver.l2

            # Pivot Point P0 in Panda3D coords (X, Y_depth, Z_up)
            p0 = self.pendulum_pivot_p3

            # Bob 1 position: x1 = l1 * sin(th1), z1 = -l1 * cos(th1)
            x1 = float(l1 * np.sin(th1))
            z1 = float(-l1 * np.cos(th1))
            p1 = Point3(p0.getX() + x1, p0.getY(), p0.getZ() + z1)
            self.pendulum_nodes["bob1"].setPos(p1)

            # Bob 2 position: x2 = x1 + l2 * sin(th2), z2 = z1 - l2 * cos(th2)
            x2 = float(x1 + l2 * np.sin(th2))
            z2 = float(z1 - l2 * np.cos(th2))
            p2 = Point3(p0.getX() + x2, p0.getY(), p0.getZ() + z2)
            self.pendulum_nodes["bob2"].setPos(p2)

            # Rod 1 starts at P0 and points directly at P1
            self.pendulum_nodes["rod1"].setPos(p0)
            self.pendulum_nodes["rod1"].lookAt(p1)

            # Rod 2 starts at P1 and points directly at P2
            self.pendulum_nodes["rod2"].setPos(p1)
            self.pendulum_nodes["rod2"].lookAt(p2)

            # Add tip of Bob 2 to trajectory ribbon
            self.trajectory_ribbon.add_point(np.array([p2.getX(), p2.getZ(), p2.getY()]))

        # 2. Update Spring-Damper 3D Oscillator
        elif self.scenario_idx == 7 and self.spring_mesh_np and self.spring_mass_np:
            if not self.is_paused:
                k_spring = 80.0
                c_damp = 1.5
                acc_y = (-k_spring * self.spring_y - c_damp * self.spring_vy) / max(1.0, self.curr_mass)
                self.spring_vy += acc_y * dt
                self.spring_y += self.spring_vy * dt

            mass_y = 9.0 + self.spring_y
            self.spring_mass_np.setPos(0.0, 0.0, mass_y)
            scale_y = max(0.2, (16.0 - mass_y) / 7.0)
            self.spring_mesh_np.setSz(scale_y)
            self.spring_mesh_np.setPos(0.0, 0.0, 16.0 - 3.5 * scale_y)
            self.trajectory_ribbon.add_point(np.array([0.0, mass_y, 0.0]))

        # 3. Update Standard Rigid Body Entities from StateBuffer
        snapshot = self.physics_thread.get_render_snapshot()
        active_mask = snapshot[:, StateIdx.ACTIVE] > 0.5
        active_indices = np.where(active_mask)[0]

        # Clean up deallocated / dead visual actor nodes
        dead_indices = [i for i in self.actor_nodes if i not in active_indices]
        for dead_idx in dead_indices:
            dead_node = self.actor_nodes.pop(dead_idx, None)
            if dead_node:
                dead_node.removeNode()

        for idx in active_indices:
            row = snapshot[idx]
            ent_type = int(row[StateIdx.ENTITY_TYPE])

            if idx not in self.actor_nodes:
                if ent_type == EntityType.DEBRIS_FRAGMENT:
                    if row[StateIdx.AREA] > 2.0:
                        actor_np = VehicleModelBuilder.create_physics_crate(size=2.2)
                    else:
                        actor_np = VehicleModelBuilder.create_debris_shard()
                elif ent_type == EntityType.FIXED_WING_JET:
                    actor_np = VehicleModelBuilder.create_fighter_jet()
                elif ent_type == EntityType.QUADROTOR_DRONE:
                    actor_np = VehicleModelBuilder.create_quadrotor_drone()
                elif ent_type == EntityType.CARGO_PARACHUTE:
                    actor_np = VehicleModelBuilder.create_cargo_parachute()
                elif ent_type == EntityType.MULTI_STAGE_ROCKET:
                    actor_np = VehicleModelBuilder.create_rocket()
                elif ent_type == EntityType.CANNON_PROJECTILE:
                    actor_np = VehicleModelBuilder.create_cannon_projectile()
                elif ent_type == EntityType.AIRFOIL_GLIDER:
                    actor_np = VehicleModelBuilder.create_airfoil_wing()
                elif ent_type == EntityType.ORBITAL_SATELLITE:
                    actor_np = VehicleModelBuilder.create_satellite()
                elif ent_type == EntityType.LORENTZ_PARTICLE:
                    actor_np = VehicleModelBuilder.create_cyclotron_chamber()
                elif ent_type == EntityType.BOUNCING_SPHERE:
                    actor_np = VehicleModelBuilder.create_bouncing_sphere()
                else:
                    actor_np = VehicleModelBuilder.create_debris_shard()

                actor_np.reparentTo(self.render)
                self.actor_nodes[idx] = actor_np

            node = self.actor_nodes.get(idx)
            if node:
                pos = row[StateIdx.PX:StateIdx.PZ + 1]
                quat = row[StateIdx.QW:StateIdx.QZ + 1]

                # Defensive NaN / Inf guards
                if np.isnan(pos).any() or np.isinf(pos).any():
                    pos = np.array([0.0, 10.0, -1000.0])
                q_len = float(np.linalg.norm(quat))
                if np.isnan(quat).any() or np.isinf(quat).any() or q_len < 1e-4:
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    quat = quat / q_len

                node.setPos(pos[0], pos[2], pos[1])
                node.setQuat(LQuaternionf(quat[0], quat[1], quat[3], quat[2]))

                if idx == self.current_controlled_idx and self.scenario_idx not in (5, 7):
                    self.trajectory_ribbon.add_point(pos)

        # 4. Update Camera Controller
        target_pos = self._get_target_pos()
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_vel = np.zeros(3)

        if 0 <= self.current_controlled_idx < self.state_buffer.max_entities and self.current_controlled_idx in active_indices:
            r = snapshot[self.current_controlled_idx]
            q = r[StateIdx.QW:StateIdx.QZ + 1]
            v = r[StateIdx.VX:StateIdx.VZ + 1]
            if not (np.isnan(q).any() or np.isinf(q).any() or np.linalg.norm(q) < 1e-4):
                target_quat = q
            if not (np.isnan(v).any() or np.isinf(v).any()):
                target_vel = v

        self.cam_controller.update(target_pos, target_quat, target_vel, dt)

        # 5. Update HUD Telemetry
        ground_h = self.terrain_gen.get_height(target_pos[0], target_pos[2])
        self.hud.update(
            state_snapshot=snapshot,
            controlled_idx=self.current_controlled_idx,
            ground_height_m=ground_h,
            physics_hz=self.physics_thread.effective_hz,
            dt=dt,
        )

        return task.cont


def launch_3d_simulator(**kwargs) -> None:
    """Launches the 3D Panda3D Application."""
    app = PyAero3DSimulatorApp(**kwargs)
    app.run()


if __name__ == "__main__":
    launch_3d_simulator()
