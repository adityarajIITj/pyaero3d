"""
Unit & Integration Tests for PyAero3D Phase 2:
Side Profile Camera, Living World Geometry, Interactive God Hand, and Sandbox Spawner.
"""

import numpy as np
import pytest
from panda3d.core import NodePath, load_prc_file_data, Point3, Vec3

# Headless window config for testing
load_prc_file_data("", """
    window-type offscreen
    audio-library-name null
""")

from direct.showbase.ShowBase import ShowBase
from pyaero3d.app_3d import PyAero3DSimulatorApp
from pyaero3d.render.camera_controller import FlightCameraController, CameraMode
from pyaero3d.render.sky_dome import EnvironmentGeometryBuilder
from pyaero3d.render.vehicle_models import VehicleModelBuilder
from pyaero3d.controls.god_hand import GodHandController
from pyaero3d.core.state import StateBuffer
from pyaero3d.core.types import StateIdx, EntityType


def test_camera_controller_modes_and_side_profile():
    dummy_node = NodePath("TestCam")
    cam_ctrl = FlightCameraController(dummy_node)

    # Test CameraMode enumerations
    assert cam_ctrl.mode == CameraMode.CHASE_BEHIND
    assert CameraMode.SIDE_PROFILE == 1
    assert CameraMode.FREE_VIEW == 2

    target_pos = np.array([0.0, 1000.0, 500.0]) # Physics X, Y_alt, Z_depth
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    target_vel = np.array([0.0, 0.0, 150.0])    # Flying along physics +Z

    # 1. Update in Chase Behind mode
    cam_ctrl.set_mode(CameraMode.CHASE_BEHIND)
    cam_ctrl.update(target_pos, target_quat, target_vel, dt=0.016)

    # Position in Panda3D should be behind target in depth (target Z is 500, so camera Z_depth should be < 500)
    cam_p = dummy_node.getPos()
    assert not np.isnan(cam_p.getX()) and not np.isnan(cam_p.getY()) and not np.isnan(cam_p.getZ())
    # In Panda3D: Y is depth, Z is altitude
    assert cam_p.getY() < 500.0 # Behind the aircraft along flight path!

    # 2. Update in Side Profile mode
    cam_ctrl.set_mode(CameraMode.SIDE_PROFILE)
    cam_ctrl.update(target_pos, target_quat, target_vel, dt=0.016)
    side_cam_p = dummy_node.getPos()
    # Side camera should have a lateral offset in X
    assert abs(side_cam_p.getX()) > 10.0


def test_environment_scenery_geometry_builders():
    tower = EnvironmentGeometryBuilder.create_atc_tower()
    assert not tower.isEmpty()
    assert tower.getNumChildren() >= 3

    hangar = EnvironmentGeometryBuilder.create_hangar()
    assert not hangar.isEmpty()
    assert hangar.getNumChildren() >= 2

    tree = EnvironmentGeometryBuilder.create_pine_tree(height=10.0)
    assert not tree.isEmpty()
    assert tree.getNumChildren() >= 4

    crate = VehicleModelBuilder.create_physics_crate(size=2.0)
    assert not crate.isEmpty()


def test_god_hand_and_sandbox_spawner_integration():
    import builtins
    if hasattr(builtins, "base"):
        del builtins.base

    app = PyAero3DSimulatorApp(scenario_idx=1)

    # Verify flat grey canvas and CAD grid
    assert not app.floor_canvas.isEmpty()
    assert not app.grid_np.isEmpty()
    assert app.terrain_gen.is_flat is True

    # Verify God Hand controller initialized
    assert isinstance(app.god_hand, GodHandController)
    assert app.god_hand.god_mode_enabled is True

    # Test spawning sandbox objects
    crate_idx = app.spawn_sandbox_object(0)
    assert crate_idx >= 0
    assert app.state_buffer.data[crate_idx, StateIdx.ACTIVE] > 0.5

    sphere_idx = app.spawn_sandbox_object(1)
    assert sphere_idx >= 0
    assert app.state_buffer.data[sphere_idx, StateIdx.ACTIVE] > 0.5

    # Step simulation frames
    for _ in range(10):
        app.taskMgr.step()

    # Verify camera mode switches
    app.set_camera_mode(1) # Side Profile
    assert app.cam_controller.mode == CameraMode.SIDE_PROFILE
    app.taskMgr.step()

    app.set_camera_mode(0) # Chase Behind
    assert app.cam_controller.mode == CameraMode.CHASE_BEHIND
    app.taskMgr.step()

    app.physics_thread.stop()
    app.destroy()
    if hasattr(builtins, "base"):
        del builtins.base
