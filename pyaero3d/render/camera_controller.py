"""
PyAero3D - Dynamic Camera Controller (Chase, Cockpit, Free Orbit).
"""

from enum import IntEnum
import numpy as np
from panda3d.core import NodePath, Vec3, Point3

from pyaero3d.core.quaternion_math import SpatialQuaternion


class CameraMode(IntEnum):
    CHASE_SPRING = 0
    COCKPIT_FIRST_PERSON = 1
    FREE_ORBIT = 2


class FlightCameraController:
    """
    Multi-Mode Dynamic Camera Controller.
    """

    def __init__(self, camera_np: NodePath):
        self.cam = camera_np
        self.mode = CameraMode.CHASE_SPRING

        # Chase cam spring parameters
        self.chase_distance = 18.0
        self.chase_height = 5.5
        self.curr_cam_pos = np.array([0.0, 50.0, -50.0], dtype=np.float64)

        # Orbit angles
        self.orbit_yaw = 0.0
        self.orbit_pitch = 15.0
        self.orbit_dist = 25.0

    def cycle_mode(self) -> CameraMode:
        """Cycles through available camera modes."""
        self.mode = CameraMode((int(self.mode) + 1) % 3)
        return self.mode

    def update(
        self,
        target_pos: np.ndarray,      # [X, Y, Z]
        target_quat: np.ndarray,     # [w, x, y, z]
        target_vel: np.ndarray,      # [vx, vy, vz]
        dt: float,
    ) -> None:
        """
        Updates camera transform according to current mode and target vehicle motion.
        """
        R = SpatialQuaternion.to_dcm(target_quat)

        if self.mode == CameraMode.CHASE_SPRING:
            # Body forward is +Z, body up is +Y
            forward = R[:, 2]
            up = R[:, 1]

            # Desired camera position: behind and above target
            speed = float(np.linalg.norm(target_vel))
            dynamic_dist = self.chase_distance + min(12.0, speed * 0.08)

            desired_pos = target_pos - forward * dynamic_dist + np.array([0.0, self.chase_height, 0.0])

            # Smooth exponential spring damping
            lerp_factor = 1.0 - np.exp(-6.0 * dt)
            self.curr_cam_pos += (desired_pos - self.curr_cam_pos) * lerp_factor

            self.cam.setPos(Point3(self.curr_cam_pos[0], self.curr_cam_pos[2], self.curr_cam_pos[1]))
            look_target = target_pos + forward * 10.0
            self.cam.lookAt(Point3(look_target[0], look_target[2], look_target[1]))

        elif self.mode == CameraMode.COCKPIT_FIRST_PERSON:
            # Mount camera slightly above and ahead of CG in cockpit
            cockpit_offset = R @ np.array([0.0, 0.8, 1.8])
            cam_p = target_pos + cockpit_offset
            self.cam.setPos(Point3(cam_p[0], cam_p[2], cam_p[1]))

            forward = R[:, 2]
            look_p = cam_p + forward * 50.0
            self.cam.lookAt(Point3(look_p[0], look_p[2], look_p[1]))

        elif self.mode == CameraMode.FREE_ORBIT:
            rad_yaw = np.radians(self.orbit_yaw)
            rad_pitch = np.radians(self.orbit_pitch)

            ox = self.orbit_dist * np.cos(rad_pitch) * np.sin(rad_yaw)
            oy = self.orbit_dist * np.sin(rad_pitch)
            oz = self.orbit_dist * np.cos(rad_pitch) * np.cos(rad_yaw)

            c_pos = target_pos + np.array([ox, oy, oz])
            self.cam.setPos(Point3(c_pos[0], c_pos[2], c_pos[1]))
            self.cam.lookAt(Point3(target_pos[0], target_pos[2], target_pos[1]))
