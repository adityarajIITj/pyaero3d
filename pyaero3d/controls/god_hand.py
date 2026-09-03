"""
PyAero3D - Interactive 3D "God Hand" Mouse Manipulation & Spring Force Controller.
Allows real-time picking, dragging, and slingshot flinging of rigid bodies, pendulum bobs, and aircraft in 3D.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from panda3d.core import (
    NodePath,
    Point3,
    Vec3,
    Vec4,
    LineSegs,
    CollisionRay,
    CollisionNode,
    CollisionTraverser,
    CollisionHandlerQueue,
)

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer


class GodHandController:
    """
    Direct 3D interactive physics manipulator.
    Casts screen rays, grabs physical bodies in the world, and applies spring tension forces with visual feedback.
    """

    def __init__(self, base_app, state_buffer: StateBuffer, render_node: NodePath):
        self.base = base_app
        self.state_buffer = state_buffer
        self.render = render_node

        # Interaction state
        self.is_active = False
        self.grabbed_entity_idx: Optional[int] = None
        self.grabbed_pendulum_bob: Optional[str] = None  # "bob1" or "bob2"
        self.drag_plane_dist = 25.0
        self.target_world_pos = np.zeros(3)

        # Spring force constants
        self.k_spring = 450.0   # N/m
        self.c_damp = 25.0      # N*s/m

        # Dynamic visual rubber-band line
        self.line_np: Optional[NodePath] = None

        # Bind God Hand controls (Ctrl + Left Click, or Left Click in God Hand mode)
        self.god_mode_enabled = True
        self._bind_events()

    def _bind_events(self) -> None:
        self.base.accept("control-mouse1", self._on_grab_start)
        self.base.accept("control-mouse1-up", self._on_grab_end)
        self.base.accept("g", self._toggle_god_hand)

    def _toggle_god_hand(self) -> None:
        self.god_mode_enabled = not self.god_mode_enabled
        status = "ENABLED" if self.god_mode_enabled else "DISABLED"
        print(f"[PyAero3D] God Hand Interaction Mode: {status} (Hold CTRL + Left Click to grab)")

    def _get_mouse_ray(self) -> Optional[Tuple[Point3, Vec3]]:
        """Computes origin and direction of ray cast from camera through mouse cursor in render space."""
        if not self.base.mouseWatcherNode.hasMouse():
            return None

        mpos = self.base.mouseWatcherNode.getMouse()
        p_near = Point3()
        p_far = Point3()
        self.base.camLens.extrude(mpos, p_near, p_far)

        p_from = self.render.getRelativePoint(self.base.camera, p_near)
        p_to = self.render.getRelativePoint(self.base.camera, p_far)
        ray_dir = (p_to - p_from).normalized()
        return p_from, ray_dir

    def _on_grab_start(self) -> None:
        if not self.god_mode_enabled:
            return

        ray = self._get_mouse_ray()
        if ray is None:
            return

        ray_origin, ray_dir = ray
        r_orig = np.array([ray_origin.getX(), ray_origin.getZ(), ray_origin.getY()])
        r_dir = np.array([ray_dir.getX(), ray_dir.getZ(), ray_dir.getY()])

        best_dist = 8.0  # Max picking threshold radius (m)
        best_idx = None
        best_bob = None

        # 1. Check Double Pendulum bobs
        if hasattr(self.base, "pendulum_nodes") and len(self.base.pendulum_nodes) == 4:
            for bob_name in ("bob1", "bob2"):
                node = self.base.pendulum_nodes.get(bob_name)
                if node and not node.isEmpty():
                    np_pos = node.getPos(self.render)
                    # Convert Panda3D pos to physics pos [X, Z, Y] -> [X, Y_alt, Z_depth]
                    obj_pos = np.array([np_pos.getX(), np_pos.getZ(), np_pos.getY()])
                    # Ray to point distance
                    v = obj_pos - r_orig
                    proj = np.dot(v, r_dir)
                    if proj > 0.5:
                        perp_dist = np.linalg.norm(v - proj * r_dir)
                        if perp_dist < best_dist:
                            best_dist = perp_dist
                            best_bob = bob_name

        # 2. Check Standard Rigid Body Entities in StateBuffer
        if best_bob is None:
            active_mask = self.state_buffer.get_active_mask()
            active_indices = np.where(active_mask)[0]
            for idx in active_indices:
                row = self.state_buffer.data[idx]
                pos = row[StateIdx.PX:StateIdx.PZ + 1]
                radius = max(1.5, float(row[StateIdx.RADIUS]))
                v = pos - r_orig
                proj = np.dot(v, r_dir)
                if proj > 1.0:
                    perp_dist = np.linalg.norm(v - proj * r_dir)
                    if perp_dist < max(best_dist, radius * 2.5):
                        best_dist = perp_dist
                        best_idx = idx

        if best_bob is not None:
            self.is_active = True
            self.grabbed_pendulum_bob = best_bob
            self.grabbed_entity_idx = None
            node = self.base.pendulum_nodes[best_bob]
            self.drag_plane_dist = float(self.base.camera.getDistance(node))
            print(f"[PyAero3D] God Hand grabbed Double Pendulum {best_bob.upper()}!")
        elif best_idx is not None:
            self.is_active = True
            self.grabbed_entity_idx = best_idx
            self.grabbed_pendulum_bob = None
            pos = self.state_buffer.data[best_idx, StateIdx.PX:StateIdx.PZ + 1]
            p_panda = Point3(pos[0], pos[2], pos[1])
            self.drag_plane_dist = float((self.render.getRelativePoint(self.render, p_panda) - ray_origin).length())
            print(f"[PyAero3D] God Hand grabbed Physical Entity #{best_idx}!")

    def _on_grab_end(self) -> None:
        if self.is_active:
            print("[PyAero3D] God Hand released object into free physics.")
        self.is_active = False
        self.grabbed_entity_idx = None
        self.grabbed_pendulum_bob = None
        self._clear_line()

    def _clear_line(self) -> None:
        if self.line_np:
            self.line_np.removeNode()
            self.line_np = None

    def update(self, dt: float) -> None:
        """Applies real-time spring tension force and updates visual rubber-band line."""
        if not self.is_active:
            self._clear_line()
            return

        ray = self._get_mouse_ray()
        if ray is None:
            self._on_grab_end()
            return

        ray_origin, ray_dir = ray
        target_point = ray_origin + ray_dir * self.drag_plane_dist

        # 1. Update Double Pendulum Bob Interaction
        if self.grabbed_pendulum_bob and hasattr(self.base, "pendulum_nodes"):
            node = self.base.pendulum_nodes.get(self.grabbed_pendulum_bob)
            if not node or node.isEmpty():
                self._on_grab_end()
                return

            obj_pos = node.getPos(self.render)
            self._draw_spring_line(obj_pos, target_point)

            # Pull angle towards mouse target
            p0 = Point3(0.0, -900.0, 20.0)
            dx = target_point.getX() - p0.getX()
            dz = target_point.getZ() - p0.getZ()
            target_angle = np.arctan2(dx, -dz)

            if self.grabbed_pendulum_bob == "bob1":
                diff = target_angle - self.base.pendulum_state[0]
                self.base.pendulum_state[1] += diff * 35.0 * dt
                self.base.pendulum_state[0] += diff * 12.0 * dt
            else:
                diff = target_angle - self.base.pendulum_state[2]
                self.base.pendulum_state[3] += diff * 45.0 * dt
                self.base.pendulum_state[2] += diff * 15.0 * dt

        # 2. Update Standard Rigid Body Interaction
        elif self.grabbed_entity_idx is not None:
            if self.state_buffer.data[self.grabbed_entity_idx, StateIdx.ACTIVE] < 0.5:
                self._on_grab_end()
                return

            row = self.state_buffer.data[self.grabbed_entity_idx]
            pos = row[StateIdx.PX:StateIdx.PZ + 1]
            vel = row[StateIdx.VX:StateIdx.VZ + 1]
            mass = max(1.0, float(row[StateIdx.MASS]))

            obj_pos_panda = Point3(pos[0], pos[2], pos[1])
            self._draw_spring_line(obj_pos_panda, target_point)

            # Target position in physics coordinates [X, Y_alt, Z_depth]
            t_phys = np.array([target_point.getX(), target_point.getZ(), target_point.getY()])

            # Spring force F = k * delta - c * vel
            delta = t_phys - pos
            dist = float(np.linalg.norm(delta))
            if dist > 0.05:
                force = (delta * (self.k_spring * mass * 0.15)) - (vel * self.c_damp * (mass * 0.05))
                # Apply directly to physics row
                row[StateIdx.FX:StateIdx.FZ + 1] += force

    def _draw_spring_line(self, p1: Point3, p2: Point3) -> None:
        """Draws glowing 3D rubber-band physical line between object and mouse point."""
        self._clear_line()
        segs = LineSegs("GodHandTensionLine")
        dist = (p2 - p1).length()
        # Tension color: Green -> Yellow -> Bright Red under extreme strain
        stress = min(1.0, dist / 40.0)
        col = Vec4(stress, 1.0 - stress * 0.7, 0.2, 0.95)
        segs.setColor(col)
        segs.setThickness(3.5)

        # Draw segmented tension curve
        steps = 10
        for i in range(steps + 1):
            t = i / float(steps)
            pt = p1 + (p2 - p1) * t
            # Add slight catenary sag
            sag = np.sin(t * np.pi) * min(2.5, dist * 0.08)
            pt.setZ(pt.getZ() - sag)
            if i == 0:
                segs.moveTo(pt)
            else:
                segs.drawTo(pt)

        geom = segs.create()
        self.line_np = self.render.attachNewNode(geom)
