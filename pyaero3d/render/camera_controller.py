"""
PyAero3D - Advanced 3D Free View & Dynamic Flight Camera Controller.
Supports Free Flight Camera (Arrow keys / WASD / Mouse Look), Orbit View, Chase Spring, and Cockpit.
"""

from enum import IntEnum
from typing import Optional
import numpy as np
from panda3d.core import NodePath, Vec3, Point3

from pyaero3d.core.quaternion_math import SpatialQuaternion


class CameraMode(IntEnum):
    FREE_VIEW = 0
    ORBIT_TARGET = 1
    CHASE_SPRING = 2
    COCKPIT_FIRST_PERSON = 3


class FlightCameraController:
    """
    Multi-Mode Dynamic Camera Controller with smooth 3D Free View navigation, Mouse Orbit, and Zoom.
    """

    def __init__(self, camera_np: NodePath, base_app=None):
        self.cam = camera_np
        self.base = base_app
        self.mode = CameraMode.FREE_VIEW

        # Free View Camera Position & Orientation (World Frame: X=East, Y=Alt Up, Z=North)
        self.free_cam_pos = np.array([0.0, 15.0, -35.0], dtype=np.float64)
        self.free_yaw = 0.0      # deg
        self.free_pitch = -12.0  # deg
        self.move_speed = 35.0   # m/s

        # Target framing distance (dynamically scaled per preset)
        self.chase_distance = 20.0
        self.chase_height = 6.0
        self.curr_cam_pos = np.array([0.0, 50.0, -50.0], dtype=np.float64)

        # 3D Orbit parameters
        self.orbit_yaw = 45.0      # Azimuth angle (deg)
        self.orbit_pitch = 20.0    # Elevation angle (deg)
        self.orbit_dist = 22.0     # Radius from target (m)

        # Mouse interaction state
        self.is_dragging = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Keyboard camera movement states
        self.cam_keys = {
            "cam_fwd": False,
            "cam_back": False,
            "cam_left": False,
            "cam_right": False,
            "cam_up": False,
            "cam_down": False,
        }

        if self.base is not None:
            self._bind_controls()

    def set_target_scale(self, target_radius: float) -> None:
        """Dynamically adjusts camera framing distance based on object size."""
        r = max(1.0, float(target_radius))
        self.chase_distance = max(6.0, r * 2.8)
        self.chase_height = max(2.0, r * 0.8)
        self.orbit_dist = max(6.0, r * 3.0)

    def _bind_controls(self) -> None:
        """Binds mouse drag, wheel zoom, and arrow keys for intuitive 3D camera navigation."""
        # Mouse
        self.base.accept("mouse3", self._on_mouse_down)
        self.base.accept("mouse3-up", self._on_mouse_up)
        self.base.accept("mouse1", self._on_mouse1_down)
        self.base.accept("mouse1-up", self._on_mouse1_up)
        self.base.accept("wheel_up", self._zoom_in)
        self.base.accept("wheel_down", self._zoom_out)

        # Arrow Keys for Free 3D Camera Movement
        self.base.accept("arrow_up", self._set_cam_key, ["cam_fwd", True])
        self.base.accept("arrow_up-up", self._set_cam_key, ["cam_fwd", False])
        self.base.accept("arrow_down", self._set_cam_key, ["cam_back", True])
        self.base.accept("arrow_down-up", self._set_cam_key, ["cam_back", False])
        self.base.accept("arrow_left", self._set_cam_key, ["cam_left", True])
        self.base.accept("arrow_left-up", self._set_cam_key, ["cam_left", False])
        self.base.accept("arrow_right", self._set_cam_key, ["cam_right", True])
        self.base.accept("arrow_right-up", self._set_cam_key, ["cam_right", False])
        self.base.accept("page_up", self._set_cam_key, ["cam_up", True])
        self.base.accept("page_up-up", self._set_cam_key, ["cam_up", False])
        self.base.accept("page_down", self._set_cam_key, ["cam_down", True])
        self.base.accept("page_down-up", self._set_cam_key, ["cam_down", False])

        # Fast camera speed modifier (Shift)
        self.base.accept("shift", lambda: setattr(self, "move_speed", 90.0))
        self.base.accept("shift-up", lambda: setattr(self, "move_speed", 35.0))

    def _set_cam_key(self, key_name: str, value: bool) -> None:
        self.cam_keys[key_name] = value

    def _on_mouse_down(self) -> None:
        self.is_dragging = True
        if self.base.mouseWatcherNode.hasMouse():
            self.last_mouse_x = self.base.mouseWatcherNode.getMouseX()
            self.last_mouse_y = self.base.mouseWatcherNode.getMouseY()

    def _on_mouse_up(self) -> None:
        self.is_dragging = False

    def _on_mouse1_down(self) -> None:
        if self.mode in (CameraMode.FREE_VIEW, CameraMode.ORBIT_TARGET):
            self.is_dragging = True
            if self.base.mouseWatcherNode.hasMouse():
                self.last_mouse_x = self.base.mouseWatcherNode.getMouseX()
                self.last_mouse_y = self.base.mouseWatcherNode.getMouseY()

    def _on_mouse1_up(self) -> None:
        if self.mode in (CameraMode.FREE_VIEW, CameraMode.ORBIT_TARGET):
            self.is_dragging = False

    def _zoom_in(self) -> None:
        self.orbit_dist = max(2.0, self.orbit_dist * 0.85)
        self.chase_distance = max(4.0, self.chase_distance * 0.85)
        if self.mode == CameraMode.FREE_VIEW:
            # Move free cam forward in direction of view
            rad_y = np.radians(self.free_yaw)
            rad_p = np.radians(self.free_pitch)
            fwd = np.array([np.sin(rad_y) * np.cos(rad_p), np.sin(rad_p), np.cos(rad_y) * np.cos(rad_p)])
            self.free_cam_pos += fwd * 6.0

    def _zoom_out(self) -> None:
        self.orbit_dist = min(800.0, self.orbit_dist * 1.18)
        self.chase_distance = min(800.0, self.chase_distance * 1.18)
        if self.mode == CameraMode.FREE_VIEW:
            rad_y = np.radians(self.free_yaw)
            rad_p = np.radians(self.free_pitch)
            fwd = np.array([np.sin(rad_y) * np.cos(rad_p), np.sin(rad_p), np.cos(rad_y) * np.cos(rad_p)])
            self.free_cam_pos -= fwd * 6.0

    def set_mode(self, mode: CameraMode) -> None:
        self.mode = mode

    def cycle_mode(self) -> CameraMode:
        """Cycles through available camera modes."""
        self.mode = CameraMode((int(self.mode) + 1) % 4)
        return self.mode

    def focus_target(self, target_pos: np.ndarray) -> None:
        """Snaps free view or orbit camera to frame target with clear 3D perspective."""
        self.orbit_yaw = 25.0
        self.orbit_pitch = 12.0
        self.free_cam_pos = target_pos + np.array([3.0, 2.0, -self.orbit_dist * 0.9])
        self.free_yaw = 5.0
        self.free_pitch = -8.0

    def update(
        self,
        target_pos: np.ndarray,      # [X, Y, Z] (Y-up in world)
        target_quat: np.ndarray,     # [w, x, y, z]
        target_vel: np.ndarray,      # [vx, vy, vz]
        dt: float,
    ) -> None:
        """
        Updates 3D camera position and orientation according to active mode.
        """
        # 1. Process Mouse Drag Look / Orbit
        if self.is_dragging and self.base and self.base.mouseWatcherNode.hasMouse():
            mx = self.base.mouseWatcherNode.getMouseX()
            my = self.base.mouseWatcherNode.getMouseY()
            dx = mx - self.last_mouse_x
            dy = my - self.last_mouse_y
            self.last_mouse_x = mx
            self.last_mouse_y = my

            if self.mode == CameraMode.FREE_VIEW:
                self.free_yaw += dx * 140.0
                self.free_pitch = np.clip(self.free_pitch + dy * 100.0, -89.0, 89.0)
            elif self.mode in (CameraMode.ORBIT_TARGET, CameraMode.CHASE_SPRING):
                self.orbit_yaw -= dx * 150.0
                self.orbit_pitch = np.clip(self.orbit_pitch + dy * 110.0, -85.0, 85.0)

        # 2. Mode-Specific Camera Position & Orientation
        if self.mode == CameraMode.FREE_VIEW:
            # Update Free Flight Position via Arrow Keys
            rad_y = np.radians(self.free_yaw)
            rad_p = np.radians(self.free_pitch)
            fwd = np.array([np.sin(rad_y) * np.cos(rad_p), np.sin(rad_p), np.cos(rad_y) * np.cos(rad_p)])
            rgt = np.array([np.cos(rad_y), 0.0, -np.sin(rad_y)])
            up_v = np.array([0.0, 1.0, 0.0])

            move_vec = np.zeros(3)
            if self.cam_keys["cam_fwd"]: move_vec += fwd
            if self.cam_keys["cam_back"]: move_vec -= fwd
            if self.cam_keys["cam_right"]: move_vec += rgt
            if self.cam_keys["cam_left"]: move_vec -= rgt
            if self.cam_keys["cam_up"]: move_vec += up_v
            if self.cam_keys["cam_down"]: move_vec -= up_v

            if np.linalg.norm(move_vec) > 1e-4:
                move_vec = move_vec / np.linalg.norm(move_vec)
                self.free_cam_pos += move_vec * (self.move_speed * dt)

            # Apply to camera: Panda3D is (X, Z, Y)
            self.cam.setPos(Point3(self.free_cam_pos[0], self.free_cam_pos[2], self.free_cam_pos[1]))
            look_p = self.free_cam_pos + fwd * 20.0
            self.cam.lookAt(Point3(look_p[0], look_p[2], look_p[1]))

        elif self.mode == CameraMode.ORBIT_TARGET:
            rad_yaw = np.radians(self.orbit_yaw)
            rad_pitch = np.radians(self.orbit_pitch)

            ox = self.orbit_dist * np.cos(rad_pitch) * np.sin(rad_yaw)
            oy = self.orbit_dist * np.sin(rad_pitch)
            oz = self.orbit_dist * np.cos(rad_pitch) * np.cos(rad_yaw)

            c_pos = target_pos + np.array([ox, oy, oz])
            self.cam.setPos(Point3(c_pos[0], c_pos[2], c_pos[1]))
            self.cam.lookAt(Point3(target_pos[0], target_pos[2], target_pos[1]))

        elif self.mode == CameraMode.CHASE_SPRING:
            R = SpatialQuaternion.to_dcm(target_quat)
            forward = R[:, 2]
            up = R[:, 1]

            speed = float(np.linalg.norm(target_vel))
            dynamic_dist = self.chase_distance + min(15.0, speed * 0.06)
            desired_pos = target_pos - forward * dynamic_dist + np.array([0.0, self.chase_height, 0.0])

            lerp_factor = 1.0 - np.exp(-7.0 * dt)
            self.curr_cam_pos += (desired_pos - self.curr_cam_pos) * lerp_factor

            self.cam.setPos(Point3(self.curr_cam_pos[0], self.curr_cam_pos[2], self.curr_cam_pos[1]))
            look_target = target_pos + forward * max(5.0, speed * 0.1)
            self.cam.lookAt(Point3(look_target[0], look_target[2], look_target[1]))

        elif self.mode == CameraMode.COCKPIT_FIRST_PERSON:
            R = SpatialQuaternion.to_dcm(target_quat)
            cockpit_offset = R @ np.array([0.0, 0.9, 2.0])
            cam_p = target_pos + cockpit_offset
            self.cam.setPos(Point3(cam_p[0], cam_p[2], cam_p[1]))

            forward = R[:, 2]
            look_p = cam_p + forward * 50.0
            self.cam.lookAt(Point3(look_p[0], look_p[2], look_p[1]))
