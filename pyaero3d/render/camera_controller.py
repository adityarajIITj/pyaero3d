"""
PyAero3D - Advanced 3D Dynamic Flight & Trajectory Camera Controller.
Supports True Chase Behind, Side Profile Trajectory Tracking, Free Flight, Mouse Orbit, and Cockpit views.
"""

from enum import IntEnum
from typing import Optional
import numpy as np
from panda3d.core import NodePath, Vec3, Point3

from pyaero3d.core.quaternion_math import SpatialQuaternion


class CameraMode(IntEnum):
    CHASE_BEHIND = 0      # True 3rd-person behind vehicle looking along flight path (NOT top-down)
    SIDE_PROFILE = 1      # 90-deg side view tracking vehicle & trajectory arc
    FREE_VIEW = 2         # Fly freely with arrow keys & mouse look
    ORBIT_TARGET = 3      # 360-deg mouse orbit around vehicle
    COCKPIT_FIRST_PERSON = 4  # 1st-person cockpit view


class FlightCameraController:
    """
    Multi-Mode Dynamic Camera Controller with smooth 3D Free View navigation,
    chase camera, side profile trajectory tracking, and mouse orbit.
    """

    def __init__(self, camera_np: NodePath, base_app=None):
        self.cam = camera_np
        self.base = base_app
        self.mode = CameraMode.CHASE_BEHIND

        # Free View Camera Position & Orientation (World Frame: X=East, Y=Alt Up, Z=North)
        self.free_cam_pos = np.array([0.0, 15.0, -35.0], dtype=np.float64)
        self.free_yaw = 0.0      # deg
        self.free_pitch = -10.0  # deg
        self.move_speed = 35.0   # m/s

        # Target framing distance (dynamically scaled per preset)
        self.chase_distance = 24.0
        self.chase_height = 5.0
        self.curr_cam_pos = np.array([0.0, 50.0, -50.0], dtype=np.float64)
        self.has_snapped = False

        # 3D Orbit parameters
        self.orbit_yaw = 35.0      # Azimuth angle (deg)
        self.orbit_pitch = 15.0    # Elevation angle (deg)
        self.orbit_dist = 28.0     # Radius from target (m)

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
        self.chase_distance = max(8.0, r * 3.2)
        self.chase_height = max(2.0, r * 0.7)
        self.orbit_dist = max(8.0, r * 3.5)

    def _bind_controls(self) -> None:
        """Binds mouse drag, wheel zoom, and keys for camera control."""
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

        # Number Key Camera Shortcuts
        self.base.accept("1", lambda: self.set_mode(CameraMode.CHASE_BEHIND))
        self.base.accept("2", lambda: self.set_mode(CameraMode.SIDE_PROFILE))
        self.base.accept("3", lambda: self.set_mode(CameraMode.FREE_VIEW))
        self.base.accept("4", lambda: self.set_mode(CameraMode.ORBIT_TARGET))
        self.base.accept("5", lambda: self.set_mode(CameraMode.COCKPIT_FIRST_PERSON))
        self.base.accept("c", self.cycle_mode)

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
        self.has_snapped = False
        print(f"[PyAero3D] Camera Mode set to: {self.mode.name}")

    def cycle_mode(self) -> CameraMode:
        """Cycles sequentially through available camera modes."""
        self.mode = CameraMode((int(self.mode) + 1) % 5)
        self.has_snapped = False
        print(f"[PyAero3D] Camera Mode cycled to: {self.mode.name}")
        return self.mode

    def focus_target(self, target_pos: np.ndarray) -> None:
        """Snaps camera directly onto target with immediate framing."""
        self.orbit_yaw = 35.0
        self.orbit_pitch = 15.0
        self.free_cam_pos = target_pos + np.array([5.0, 3.0, -self.orbit_dist * 0.85])
        self.free_yaw = 10.0
        self.free_pitch = -8.0
        self.curr_cam_pos = target_pos - np.array([0.0, 0.0, self.chase_distance]) + np.array([0.0, self.chase_height, 0.0])
        self.has_snapped = False

    def update(
        self,
        target_pos: np.ndarray,      # [X, Y, Z] (Y-up in world physics coords)
        target_quat: np.ndarray,     # [w, x, y, z]
        target_vel: np.ndarray,      # [vx, vy, vz]
        dt: float,
    ) -> None:
        """
        Updates 3D camera position and orientation according to active mode.
        """
        # 1. Mouse Drag Look / Orbit
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
            elif self.mode in (CameraMode.ORBIT_TARGET, CameraMode.CHASE_BEHIND, CameraMode.SIDE_PROFILE):
                self.orbit_yaw -= dx * 150.0
                self.orbit_pitch = np.clip(self.orbit_pitch + dy * 110.0, -85.0, 85.0)

        # Extract target forward & up directions in physics coordinate system
        R = SpatialQuaternion.to_dcm(target_quat)
        speed = float(np.linalg.norm(target_vel))

        # Vehicle forward direction: prioritize velocity direction if moving, else body +Z forward
        if speed > 2.0:
            fwd_dir = target_vel / speed
        else:
            fwd_dir = R[:, 2]
            f_norm = np.linalg.norm(fwd_dir)
            if f_norm > 1e-4:
                fwd_dir = fwd_dir / f_norm
            else:
                fwd_dir = np.array([0.0, 0.0, 1.0])

        up_world = np.array([0.0, 1.0, 0.0])

        # 2. Camera Placement Modes
        if self.mode == CameraMode.CHASE_BEHIND:
            # Sits behind the vehicle looking ahead along flight path, tilted slightly up (NOT top-down)
            dynamic_dist = self.chase_distance + min(25.0, speed * 0.05)
            desired_pos = target_pos - fwd_dir * dynamic_dist + up_world * self.chase_height

            if not self.has_snapped or np.linalg.norm(desired_pos - self.curr_cam_pos) > 200.0:
                self.curr_cam_pos = desired_pos.copy()
                self.has_snapped = True
            else:
                lerp_factor = 1.0 - np.exp(-8.0 * dt)
                self.curr_cam_pos += (desired_pos - self.curr_cam_pos) * lerp_factor

            # Convert to Panda3D coords: (X=curr[0], Y_depth=curr[2], Z_alt=curr[1])
            self.cam.setPos(Point3(self.curr_cam_pos[0], self.curr_cam_pos[2], self.curr_cam_pos[1]))
            look_target = target_pos + fwd_dir * max(8.0, speed * 0.15) + up_world * 1.5
            self.cam.lookAt(Point3(look_target[0], look_target[2], look_target[1]))

        elif self.mode == CameraMode.SIDE_PROFILE:
            # 90-degree side profile tracking camera
            # Side vector perpendicular to flight direction in horizontal plane
            side_vec = np.cross(fwd_dir, up_world)
            s_len = np.linalg.norm(side_vec)
            if s_len > 1e-4:
                side_vec = side_vec / s_len
            else:
                side_vec = np.array([1.0, 0.0, 0.0])

            side_dist = self.chase_distance * 1.8
            desired_pos = target_pos + side_vec * side_dist + up_world * (self.chase_height * 0.8)

            if not self.has_snapped or np.linalg.norm(desired_pos - self.curr_cam_pos) > 200.0:
                self.curr_cam_pos = desired_pos.copy()
                self.has_snapped = True
            else:
                lerp_factor = 1.0 - np.exp(-9.0 * dt)
                self.curr_cam_pos += (desired_pos - self.curr_cam_pos) * lerp_factor

            self.cam.setPos(Point3(self.curr_cam_pos[0], self.curr_cam_pos[2], self.curr_cam_pos[1]))
            look_target = target_pos + fwd_dir * 4.0
            self.cam.lookAt(Point3(look_target[0], look_target[2], look_target[1]))

        elif self.mode == CameraMode.ORBIT_TARGET:
            rad_yaw = np.radians(self.orbit_yaw)
            rad_pitch = np.radians(self.orbit_pitch)

            ox = self.orbit_dist * np.cos(rad_pitch) * np.sin(rad_yaw)
            oy = self.orbit_dist * np.sin(rad_pitch)
            oz = self.orbit_dist * np.cos(rad_pitch) * np.cos(rad_yaw)

            c_pos = target_pos + np.array([ox, oy, oz])
            self.cam.setPos(Point3(c_pos[0], c_pos[2], c_pos[1]))
            self.cam.lookAt(Point3(target_pos[0], target_pos[2], target_pos[1]))

        elif self.mode == CameraMode.COCKPIT_FIRST_PERSON:
            cockpit_offset = R @ np.array([0.0, 0.9, 2.2])
            cam_p = target_pos + cockpit_offset
            self.cam.setPos(Point3(cam_p[0], cam_p[2], cam_p[1]))

            look_p = cam_p + fwd_dir * 60.0
            self.cam.lookAt(Point3(look_p[0], look_p[2], look_p[1]))

        elif self.mode == CameraMode.FREE_VIEW:
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

            self.cam.setPos(Point3(self.free_cam_pos[0], self.free_cam_pos[2], self.free_cam_pos[1]))
            look_p = self.free_cam_pos + fwd * 25.0
            self.cam.lookAt(Point3(look_p[0], look_p[2], look_p[1]))
