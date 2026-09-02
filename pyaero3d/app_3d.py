"""
PyAero3D - 3D Panda3D Mountain Simulation & Multi-Vehicle Sandbox Application.
"""

import sys
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    load_prc_file_data,
    Vec3,
    WindowProperties,
    DirectionalLight,
    AmbientLight,
    NodePath,
)

# Panda3D Window & Graphics Configuration
load_prc_file_data("", """
    window-title PyAero3D // Mountain World Simulation & Aerospace Sandbox
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

    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.05, 0.05, 0.08, 1.0)

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
        self.actor_nodes = {}
        self.current_controlled_idx = 0

        # Connect yoke callbacks
        self.yoke.on_spawn_jet = self.spawn_jet_on_runway
        self.yoke.on_spawn_drone = self.spawn_drone_mountain
        self.yoke.on_spawn_cargo = self.spawn_cargo_drop
        self.yoke.on_spawn_rocket = self.spawn_rocket
        self.yoke.on_cycle_cam = self._cycle_camera_mode
        self.yoke.on_trigger_breakup = self._trigger_manual_breakup
        self.yoke.on_toggle_units = self.hud.toggle_unit_system
        self.yoke.on_toggle_help = self.hud.toggle_help_guide
        self.yoke.on_reset_world = self._reset_simulation_world

        # Start Physics Engine Thread
        self.physics_thread.start()

        # Spawn Initial Vehicle (Fighter Jet on Runway)
        self.spawn_jet_on_runway()

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

    def spawn_jet_on_runway(self) -> int:
        init_pos = np.array([0.0, 1.8, -1000.0], dtype=np.float64)
        init_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.FIXED_WING_JET,
            mass=12500.0,
            position=init_pos,
            velocity=init_vel,
            radius=8.5,
            cd=0.024,
            area=28.0,
        )

        actor_np = VehicleModelBuilder.create_fighter_jet()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned Fighter Jet (Entity #{idx}) on Runway.")
        return idx

    def spawn_drone_mountain(self) -> int:
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

        actor_np = VehicleModelBuilder.create_quadrotor()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned Quadrotor Drone (Entity #{idx}) over Mountain Ridge.")
        return idx

    def spawn_cargo_drop(self) -> int:
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

    def spawn_rocket(self) -> int:
        init_pos = np.array([300.0, 5.0, -1200.0], dtype=np.float64)
        init_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        idx = self.state_buffer.allocate_entity(
            entity_type=EntityType.MULTI_STAGE_ROCKET,
            mass=8500.0,
            position=init_pos,
            velocity=init_vel,
            radius=1.8,
            cd=0.20,
            area=4.5,
        )

        actor_np = VehicleModelBuilder.create_rocket()
        actor_np.reparentTo(self.render)
        self.actor_nodes[idx] = actor_np
        self.current_controlled_idx = idx
        self.yoke.set_target_entity(idx)
        print(f"[PyAero3D] Spawned Multi-Stage Launch Rocket (Entity #{idx}).")
        return idx

    def _trigger_manual_breakup(self) -> None:
        if self.current_controlled_idx in self.actor_nodes:
            print(f"[PyAero3D] Triggering Kinetic Breakup on Entity #{self.current_controlled_idx}...")
            FragmentationEngine.explode_entity(self.state_buffer, self.current_controlled_idx, num_shards=16)
            old_np = self.actor_nodes.pop(self.current_controlled_idx, None)
            if old_np:
                old_np.removeNode()

    def _reset_simulation_world(self) -> None:
        """Resets simulation world, clears all active entities, and respawns fighter jet."""
        print("[PyAero3D] Resetting Simulation World...")
        for idx, node in list(self.actor_nodes.items()):
            node.removeNode()
            self.state_buffer.free_entity(idx)
        self.actor_nodes.clear()
        self.spawn_jet_on_runway()

    def _cycle_camera_mode(self) -> None:
        mode = self.cam_controller.cycle_mode()
        print(f"[PyAero3D] Camera Mode: {mode.name}")

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
                    actor_np = VehicleModelBuilder.create_quadrotor()
                elif ent_type == EntityType.CARGO_PARACHUTE:
                    actor_np = VehicleModelBuilder.create_cargo_parachute()
                elif ent_type == EntityType.MULTI_STAGE_ROCKET:
                    actor_np = VehicleModelBuilder.create_rocket()
                else:
                    actor_np = VehicleModelBuilder.create_debris_shard()

                actor_np.reparentTo(self.render)
                self.actor_nodes[idx] = actor_np

            node = self.actor_nodes.get(idx)
            if node:
                pos = row[StateIdx.PX:StateIdx.PZ + 1]
                quat = row[StateIdx.QW:StateIdx.QZ + 1]
                node.setPos(pos[0], pos[2], pos[1])

                from panda3d.core import LQuaternionf
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
            px, pz = target_row[StateIdx.PX], target_row[StateIdx.PZ]
            ground_h = self.terrain_gen.get_height(px, pz)

        self.hud.update_telemetry(
            state_row=target_row,
            ground_h=ground_h,
            camera_mode_name=self.cam_controller.mode.name,
            physics_hz=self.physics_thread.effective_hz,
            total_active=len(active_indices),
        )

        return task.cont

    def userExit(self):
        print("[PyAero3D] Stopping Physics Thread...")
        self.physics_thread.stop()
        super().userExit()
