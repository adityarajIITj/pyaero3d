"""
PyAero3D - 3D Panda3D Mountain Simulation & Multi-Vehicle Sandbox Application.
Supports all 8 physical presets launched seamlessly from the XY Graph Studio or CLI.
"""

import sys
import numpy as np
from typing import Optional, Dict, Any

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    load_prc_file_data,
    Vec3,
    WindowProperties,
    DirectionalLight,
    AmbientLight,
    NodePath,
    LQuaternionf,
)

# Panda3D Window & Graphics Configuration
load_prc_file_data("", """
    window-title PyAero3D // Mountain World Simulation & Multi-Scenario Sandbox (3D)
    win-size 1600 900
    sync-video #t
    show-frame-rate-meter #t
    textures-power-2 none
    gl-version 3 2
""")

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer
from pyaero3d.render.terrain_gen import MountainTerrainGenerator
from pyaero3d.render.sky_dome import EnvironmentGeometryBuilder
from pyaero3d.render.vehicle_models import VehicleModelBuilder
from pyaero3d.render.camera_controller import FlightCameraController
from pyaero3d.ui.hud_overlay import FlightHUDOverlay
from pyaero3d.controls.flight_yoke import FlightYokeController
from pyaero3d.physics.engine_thread import PhysicsEngineThread
from pyaero3d.physics.fragmentation import FragmentationEngine
from pyaero3d.physics.mountain_collision import MountainCollisionEngine


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

        self.scenario_idx = scenario_idx
        self.param_v0 = v0
        self.param_theta = theta
        self.param_mass = mass
        self.param_cd = cd
        self.param_area = area
        self.param_wind = wind
        self.param_thrust = thrust

        # Configure Camera Lens FOV
        self.camLens.setFov(75)
        self.camLens.setNearFar(0.5, 50000.0)

        # 1. State Buffer Tensor
        self.state_buffer = StateBuffer(max_entities=256)

        # 2. Procedural Mountain Terrain & Sky Dome
        print("[PyAero3D] Generating Alpine Mountain Terrain...")
        self.terrain_gen = MountainTerrainGenerator(world_size_m=12000.0, max_height_m=2400.0, grid_resolution=512)
        self.terrain_node = self.terrain_gen.build_shader_terrain(self.render, self.camera)

        self.sky_dome = EnvironmentGeometryBuilder.create_sky_dome(radius=40000.0)
        self.sky_dome.reparentTo(self.render)
        self.runway = EnvironmentGeometryBuilder.create_runway_strip(elevation=1.0)
        self.runway.reparentTo(self.render)

        # 3. Sunlight & Ambient Lighting
        self._setup_lighting()

        # 4. Collision & Fragmentation Engines
        self.collision_engine = MountainCollisionEngine(self.terrain_gen)
        self.fragmentation_engine = FragmentationEngine

        # 5. 1000Hz Background Physics Engine Thread
        self.physics_thread = PhysicsEngineThread(
            state_buffer=self.state_buffer,
            collision_engine=self.collision_engine,
            target_hz=1000.0,
        )

        # 6. Glass Cockpit Telemetry HUD & Input Yoke
        self.hud = FlightHUDOverlay()
        self.yoke = FlightYokeController(self, self.state_buffer)

        # 7. Dynamic Camera Controller
        self.cam_controller = FlightCameraController(self.camera)

        # 8. Visual Actor Nodes Mapping
        self.actor_nodes: Dict[int, NodePath] = {}
        self.current_controlled_idx = 0

        # Connect yoke callbacks
        self.yoke.on_spawn_jet = self.spawn_fighter_jet_airborne
        self.yoke.on_spawn_drone = self.spawn_drone_mountain
        self.yoke.on_spawn_cargo = self.spawn_cargo_drop
        self.yoke.on_spawn_rocket = self.spawn_rocket_launch
        self.yoke.on_spawn_cannon = self.spawn_cannon_projectile
        self.yoke.on_spawn_glider = self.spawn_airfoil_glider
        self.yoke.on_spawn_satellite = self.spawn_orbital_satellite
        self.yoke.on_spawn_sphere = self.spawn_bouncing_spheres
        self.yoke.on_cycle_cam = self._cycle_camera_mode
        self.yoke.on_trigger_breakup = self._trigger_manual_breakup
        self.yoke.on_toggle_units = self.hud.toggle_unit_system
        self.yoke.on_toggle_help = self.hud.toggle_help_guide
        self.yoke.on_reset_world = self._reset_simulation_world

        # Start Physics Engine Thread
        self.physics_thread.start()

        # Spawn Active Scenario Preset
        self._spawn_preset_scenario(self.scenario_idx)

        # Register Main Render Frame Task
        self.taskMgr.add(self._render_frame_update, "PyAero3D_RenderUpdate")

    def _setup_lighting(self) -> None:
        """Sets up realistic sunlight and ambient lighting."""
        dlight = DirectionalLight("sunlight")
        dlight.setColor((1.0, 0.96, 0.90, 1.0))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(-45, -60, 0)
        self.render.setLight(dlnp)

        alight = AmbientLight("ambient")
        alight.setColor((0.35, 0.40, 0.50, 1.0))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

    def _spawn_preset_scenario(self, idx: int) -> int:
        """Spawns the exact 3D physical counterpart of the selected Graph Studio preset."""
        if idx == 0:  # Ballistics Cannon Projectile
            return self.spawn_cannon_projectile()
        elif idx == 1:  # Fighter Jet 6-DOF
            return self.spawn_fighter_jet_airborne()
        elif idx == 2:  # NACA Airfoil Glider
            return self.spawn_airfoil_glider()
        elif idx == 3:  # Multi-Stage Rocket Launch
            return self.spawn_rocket_launch()
        elif idx == 4:  # Orbital Satellite
            return self.spawn_orbital_satellite()
        elif idx == 5:  # Double Pendulum
            return self.spawn_double_pendulum()
        elif idx == 6:  # Lorentz Particle Cyclotron
            return self.spawn_cyclotron_particle()
        elif idx == 7:  # Bouncing Viscoelastic Spheres
            return self.spawn_bouncing_spheres()
        else:
            return self.spawn_fighter_jet_airborne()

    def spawn_fighter_jet_airborne(self) -> int:
        """Spawns Fighter Jet in mid-air with high forward airspeed and responsive throttle."""
        v0 = self.param_v0 if self.param_v0 is not None else 220.0
        theta_deg = self.param_theta if self.param_theta is not None else 5.0
        mass = self.param_mass if self.param_mass is not None else 12000.0
        cd = self.param_cd if self.param_cd is not None else 0.024
        area = self.param_area if self.param_area is not None else 28.0

        theta_rad = np.radians(theta_deg)
        init_pos = np.array([0.0, 1200.0, -500.0], dtype=np.float64)
        init_vel = np.array([0.0, v0 * np.sin(theta_rad), v0 * np.cos(theta_rad)], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.FIXED_WING_JET,
            mass=mass,
            position=init_pos,
            velocity=init_vel,
            radius=8.5,
            cd=cd,
            area=area,
        )
        self.state_buffer.data[idx, StateIdx.THROTTLE] = 0.85

        actor_np = VehicleModelBuilder.create_fighter_jet()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned 6-DOF Fighter Jet (Entity #{idx}) at 1,200m altitude ({v0*3.6:.0f} km/h, 85% Throttle).")
        return idx

    def spawn_cannon_projectile(self) -> int:
        """Spawns 3D artillery cannon shell with exact launch angle and velocity from graph."""
        v0 = self.param_v0 if self.param_v0 is not None else 320.0
        theta_deg = self.param_theta if self.param_theta is not None else 45.0
        mass = self.param_mass if self.param_mass is not None else 15.0
        cd = self.param_cd if self.param_cd is not None else 0.30
        area = self.param_area if self.param_area is not None else 0.08

        theta_rad = np.radians(theta_deg)
        init_pos = np.array([0.0, 5.0, -1200.0], dtype=np.float64)
        init_vel = np.array([0.0, v0 * np.sin(theta_rad), v0 * np.cos(theta_rad)], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.CANNON_PROJECTILE,
            mass=mass,
            position=init_pos,
            velocity=init_vel,
            radius=1.2,
            cd=cd,
            area=area,
        )

        actor_np = VehicleModelBuilder.create_cannon_projectile()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Fired Ballistic Artillery Shell (Entity #{idx}) at {theta_deg:.1f}° pitch, v0={v0:.0f} m/s.")
        return idx

    def spawn_airfoil_glider(self) -> int:
        """Spawns NACA Glider Wing vehicle gliding over mountain valley."""
        v0 = self.param_v0 if self.param_v0 is not None else 95.0
        mass = self.param_mass if self.param_mass is not None else 650.0
        cd = self.param_cd if self.param_cd is not None else 0.018
        area = self.param_area if self.param_area is not None else 16.0

        init_pos = np.array([0.0, 1600.0, -200.0], dtype=np.float64)
        init_vel = np.array([0.0, -1.5, v0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.AIRFOIL_GLIDER,
            mass=mass,
            position=init_pos,
            velocity=init_vel,
            radius=7.5,
            cd=cd,
            area=area,
        )

        actor_np = VehicleModelBuilder.create_airfoil_wing()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned NACA Aerodynamic Glider (Entity #{idx}) at 1,600m altitude.")
        return idx

    def spawn_rocket_launch(self) -> int:
        """Spawns 3D multi-stage rocket on launch pad with full vertical throttle."""
        mass = self.param_mass if self.param_mass is not None else 8500.0
        cd = self.param_cd if self.param_cd is not None else 0.20
        area = self.param_area if self.param_area is not None else 4.5

        init_pos = np.array([300.0, 5.0, -1200.0], dtype=np.float64)
        init_vel = np.array([0.0, 5.0, 0.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.MULTI_STAGE_ROCKET,
            mass=mass,
            position=init_pos,
            velocity=init_vel,
            radius=1.8,
            cd=cd,
            area=area,
        )
        self.state_buffer.data[idx, StateIdx.THROTTLE] = 1.0

        actor_np = VehicleModelBuilder.create_rocket()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Initiated Multi-Stage Rocket Liftoff (Entity #{idx}) from Launch Pad.")
        return idx

    def spawn_orbital_satellite(self) -> int:
        """Spawns 3D satellite in space orbit with solar panels."""
        init_pos = np.array([0.0, 3500.0, 0.0], dtype=np.float64)
        init_vel = np.array([0.0, 0.0, 180.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.ORBITAL_SATELLITE,
            mass=2400.0,
            position=init_pos,
            velocity=init_vel,
            radius=3.5,
            cd=0.01,
            area=8.0,
        )

        actor_np = VehicleModelBuilder.create_satellite()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned Orbital Satellite (Entity #{idx}) in Orbit at 3,500m.")
        return idx

    def spawn_double_pendulum(self) -> int:
        """Spawns 3D articulated double pendulum swinging in alpine clearing."""
        init_pos = np.array([0.0, 20.0, -900.0], dtype=np.float64)
        init_vel = np.array([5.0, 0.0, 0.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.DOUBLE_PENDULUM,
            mass=10.0,
            position=init_pos,
            velocity=init_vel,
            radius=4.0,
            cd=0.10,
            area=1.0,
        )

        actor_np = VehicleModelBuilder.create_double_pendulum_rods()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned 3D Articulated Double Pendulum (Entity #{idx}).")
        return idx

    def spawn_cyclotron_particle(self) -> int:
        """Spawns 3D particle cyclotron chamber with magnetic gyromotion."""
        init_pos = np.array([0.0, 30.0, -800.0], dtype=np.float64)
        init_vel = np.array([50.0, 20.0, 0.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.LORENTZ_PARTICLE,
            mass=1.0,
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
        print(f"[PyAero3D] Spawned 3D Lorentz Particle Cyclotron (Entity #{idx}).")
        return idx

    def spawn_bouncing_spheres(self) -> int:
        """Spawns bouncing viscoelastic physical spheres down mountain slope."""
        init_pos = np.array([0.0, 850.0, -200.0], dtype=np.float64)
        init_vel = np.array([15.0, 5.0, 40.0], dtype=np.float64)

        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.BOUNCING_SPHERE,
            mass=25.0,
            position=init_pos,
            velocity=init_vel,
            radius=1.5,
            cd=0.45,
            area=1.8,
        )

        actor_np = VehicleModelBuilder.create_bouncing_sphere()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned Viscoelastic Bouncing Sphere (Entity #{idx}) on Mountain Slope.")
        return idx

    def spawn_drone_mountain(self) -> int:
        """Spawns 6-DOF quadrotor drone hovering over mountain ridge."""
        init_pos = np.array([1500.0, 950.0, 1200.0], dtype=np.float64)
        init_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.QUADROTOR_DRONE,
            mass=4.5,
            position=init_pos,
            velocity=init_vel,
            radius=0.6,
            cd=1.1,
            area=0.25,
        )
        self.state_buffer.data[idx, StateIdx.THROTTLE] = 0.55

        actor_np = VehicleModelBuilder.create_quadrotor_drone()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned Quadrotor Drone (Entity #{idx}) hovering over Mountain Ridge.")
        return idx

    def spawn_cargo_drop(self) -> int:
        """Spawns cargo parachute drop at high altitude."""
        init_pos = np.array([0.0, 2200.0, 500.0], dtype=np.float64)
        init_vel = np.array([0.0, 0.0, 80.0], dtype=np.float64)
        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.CARGO_PARACHUTE,
            mass=1200.0,
            position=init_pos,
            velocity=init_vel,
            radius=2.5,
            cd=1.45,
            area=45.0,
        )

        actor_np = VehicleModelBuilder.create_cargo_parachute()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned High-Altitude Cargo Parachute Drop (Entity #{idx}).")
        return idx

    def _cycle_camera_mode(self) -> None:
        new_mode = self.cam_controller.cycle_mode()
        print(f"[PyAero3D] Camera Mode switched to: {new_mode.name}")

    def _trigger_manual_breakup(self) -> None:
        if self.current_controlled_idx in self.actor_nodes:
            print(f"[PyAero3D] Triggering Kinetic Breakup on Entity #{self.current_controlled_idx}...")
            FragmentationEngine.explode_entity(self.state_buffer, self.current_controlled_idx, num_shards=16)
            old_np = self.actor_nodes.pop(self.current_controlled_idx, None)
            if old_np:
                old_np.removeNode()

    def _reset_simulation_world(self) -> None:
        print("[PyAero3D] Resetting simulation world...")
        for node in self.actor_nodes.values():
            node.removeNode()
        self.actor_nodes.clear()
        self.state_buffer.clear()
        self._spawn_preset_scenario(self.scenario_idx)

    def _render_frame_update(self, task):
        dt = globalClock.getDt()

        # Update controls
        self.yoke.update(dt)

        # Get double-buffered physics snapshot
        snapshot = self.physics_thread.get_render_snapshot()
        active_mask = snapshot[:, StateIdx.ACTIVE] > 0.5
        active_indices = np.where(active_mask)[0]

        # Sync 3D Visual Actors
        for idx in active_indices:
            row = snapshot[idx]
            ent_type = int(row[StateIdx.ENTITY_TYPE])

            if idx not in self.actor_nodes:
                if ent_type == EntityType.DEBRIS_FRAGMENT:
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
                elif ent_type == EntityType.DOUBLE_PENDULUM:
                    actor_np = VehicleModelBuilder.create_double_pendulum_rods()
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
                node.setPos(pos[0], pos[2], pos[1])
                node.setQuat(LQuaternionf(quat[0], quat[1], quat[3], quat[2]))

        # Update Camera Controller
        target_row = snapshot[self.current_controlled_idx] if self.current_controlled_idx in active_indices else None
        if target_row is not None:
            t_pos = target_row[StateIdx.PX:StateIdx.PZ + 1]
            t_quat = target_row[StateIdx.QW:StateIdx.QZ + 1]
            t_vel = target_row[StateIdx.VX:StateIdx.VZ + 1]
            self.cam_controller.update(t_pos, t_quat, t_vel, dt)

        # Update HUD Telemetry
        ground_h = 0.0
        if target_row is not None:
            ground_h = self.terrain_gen.get_height(target_row[StateIdx.PX], target_row[StateIdx.PZ])
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
