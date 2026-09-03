"""
PyAero3D - 3D Spatial Reference Gizmos, Cartesian Coordinate Axes, Ground Grid, and Trajectory Ribbons.
Provides clear 3D spatial perception, depth cues, and real-time motion trail ribbons.
"""

from typing import List, Tuple, Optional
import numpy as np
from panda3d.core import (
    NodePath, Geom, GeomNode, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, GeomLines, GeomTriangles, Vec4, Vec3, Point3
)


class SpatialReferenceBuilder:
    """
    Constructs 3D Cartesian Coordinate Axes, Ground Reference Grids, and Launch Stands.
    """

    @staticmethod
    def create_coordinate_axes(length: float = 10.0, thickness: float = 0.15) -> NodePath:
        """
        Creates 3D XYZ Cartesian Axis Gizmo:
        X-Axis = Red (+X, East)
        Y-Axis = Green (+Y, Altitude Up)
        Z-Axis = Blue (+Z, North / Forward)
        """
        vdata = GeomVertexData("AxesMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        lines = GeomLines(Geom.UHDynamic)

        red   = (1.0, 0.20, 0.20, 1.0) # +X
        green = (0.20, 0.95, 0.30, 1.0) # +Y (Up)
        blue  = (0.20, 0.50, 1.00, 1.0) # +Z (Forward)

        # Origin
        v_writer.addData3(0, 0, 0)
        c_writer.addData4(*red)
        v_writer.addData3(length, 0, 0)
        c_writer.addData4(*red)

        v_writer.addData3(0, 0, 0)
        c_writer.addData4(*green)
        v_writer.addData3(0, length, 0)
        c_writer.addData4(*green)

        v_writer.addData3(0, 0, 0)
        c_writer.addData4(*blue)
        v_writer.addData3(0, 0, length)
        c_writer.addData4(*blue)

        lines.addVertices(0, 1)
        lines.addVertices(2, 3)
        lines.addVertices(4, 5)

        geom = Geom(vdata)
        geom.addPrimitive(lines)
        node = GeomNode("CoordinateAxesNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_flat_canvas(size: float = 25000.0, color: Tuple[float, float, float, float] = (0.16, 0.17, 0.20, 1.0)) -> NodePath:
        """Creates a smooth flat grey floor canvas for clean CAD/laboratory world."""
        from panda3d.core import CardMaker
        cm = CardMaker("FlatCanvasFloor")
        cm.setFrame(-size * 0.5, size * 0.5, -size * 0.5, size * 0.5)
        np_card = NodePath(cm.generate())
        np_card.setP(-90)
        np_card.setPos(0.0, 0.0, 0.0)
        np_card.setColor(Vec4(*color))
        return np_card

    @staticmethod
    def create_cad_grid(size: float = 6000.0, major_step: float = 100.0, minor_step: float = 20.0, elevation: float = 0.02) -> NodePath:
        """Creates a crisp dual-tone CAD metric grid (minor + major accent lines)."""
        vdata = GeomVertexData("CadGrid", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        lines = GeomLines(Geom.UHDynamic)

        minor_col = (0.24, 0.26, 0.30, 0.6)
        major_col = (0.35, 0.40, 0.48, 0.9)
        axis_x_col = (0.85, 0.25, 0.25, 1.0)
        axis_z_col = (0.25, 0.45, 0.95, 1.0)

        half = size * 0.5
        ticks = np.arange(-half, half + 1e-4, minor_step)
        v_idx = 0

        for t in ticks:
            is_major = abs(t % major_step) < 1e-3 or abs(t % major_step - major_step) < 1e-3
            is_center = abs(t) < 1e-3

            col_x = axis_x_col if is_center else (major_col if is_major else minor_col)
            col_z = axis_z_col if is_center else (major_col if is_major else minor_col)

            # Line parallel to Y (depth)
            v_writer.addData3(t, -half, elevation)
            c_writer.addData4(*col_x)
            v_writer.addData3(t, half, elevation)
            c_writer.addData4(*col_x)
            lines.addVertices(v_idx, v_idx + 1)
            v_idx += 2

            # Line parallel to X
            v_writer.addData3(-half, t, elevation)
            c_writer.addData4(*col_z)
            v_writer.addData3(half, t, elevation)
            c_writer.addData4(*col_z)
            lines.addVertices(v_idx, v_idx + 1)
            v_idx += 2

        geom = Geom(vdata)
        geom.addPrimitive(lines)
        node = GeomNode("CadGridNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_ground_grid(size: float = 200.0, step: float = 10.0, elevation: float = 0.5) -> NodePath:
        """
        Creates a high-visibility 3D ground reference grid for distance and scale estimation.
        """
        vdata = GeomVertexData("GridMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        lines = GeomLines(Geom.UHDynamic)

        grid_col = (0.25, 0.35, 0.45, 0.6)
        center_col = (0.85, 0.70, 0.20, 0.9)

        half = size * 0.5
        ticks = np.arange(-half, half + 1e-4, step)
        v_idx = 0

        for t in ticks:
            col = center_col if abs(t) < 1e-3 else grid_col
            # Line parallel to Z (along X = t)
            v_writer.addData3(t, elevation, -half)
            c_writer.addData4(*col)
            v_writer.addData3(t, elevation, half)
            c_writer.addData4(*col)
            lines.addVertices(v_idx, v_idx + 1)
            v_idx += 2

            # Line parallel to X (along Z = t)
            v_writer.addData3(-half, elevation, t)
            c_writer.addData4(*col)
            v_writer.addData3(half, elevation, t)
            c_writer.addData4(*col)
            lines.addVertices(v_idx, v_idx + 1)
            v_idx += 2

        geom = Geom(vdata)
        geom.addPrimitive(lines)
        node = GeomNode("GroundGridNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_pendulum_stand() -> NodePath:
        """
        Creates a solid floor mounting stand & pivot bracket for the 3D double pendulum.
        """
        vdata = GeomVertexData("StandMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        dark_steel = (0.20, 0.22, 0.26, 1.0)
        brass_col  = (0.85, 0.65, 0.20, 1.0)

        # Base plate (2m x 2m on ground)
        pts = [
            (-1.5, 0.0, -1.5), (1.5, 0.0, -1.5), (1.5, 0.0, 1.5), (-1.5, 0.0, 1.5),
            # Vertical column (to y = 20m)
            (-0.2, 0.0, -0.2), (0.2, 0.0, -0.2), (0.2, 0.0, 0.2), (-0.2, 0.0, 0.2),
            (-0.2, 20.0, -0.2), (0.2, 20.0, -0.2), (0.2, 20.0, 0.2), (-0.2, 20.0, 0.2),
            # Top pivot cantilever arm (protruding along +Z by 3m)
            (-0.2, 20.0, 3.0), (0.2, 20.0, 3.0),
        ]
        for idx, p in enumerate(pts):
            v_writer.addData3(*p)
            if idx >= 12: c_writer.addData4(*brass_col)
            else: c_writer.addData4(*dark_steel)

        tri_list = [
            # Base
            (0, 1, 2), (0, 2, 3),
            # Pillar sides
            (4, 5, 9), (4, 9, 8), (5, 6, 10), (5, 10, 9),
            (6, 7, 11), (6, 11, 10), (7, 4, 8), (7, 8, 11),
            # Cantilever arm
            (8, 9, 13), (8, 13, 12),
        ]
        for (i0, i1, i2) in tri_list:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("PendulumStandNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_spring_damper_mechanism(stretch: float = 1.0) -> NodePath:
        """
        Builds 3D helical coil spring mesh that dynamically scales with displacement.
        """
        vdata = GeomVertexData("SpringMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        lines = GeomLines(Geom.UHDynamic)

        gold_spring = (0.95, 0.75, 0.15, 1.0)
        silver_rod  = (0.75, 0.80, 0.85, 1.0)

        # 1. Top mount
        v_writer.addData3(0.0, 15.0, 0.0)
        c_writer.addData4(*silver_rod)
        v_writer.addData3(0.0, 13.5, 0.0)
        c_writer.addData4(*silver_rod)
        lines.addVertices(0, 1)

        # 2. Helical Coil Spring
        num_coils = 8
        num_pts = 64
        y_start = 13.5
        y_end = max(1.5, 13.5 - 10.0 * stretch)
        r_coil = 0.8

        v_base = 2
        for i in range(num_pts):
            t = i / float(num_pts - 1)
            y = y_start + (y_end - y_start) * t
            angle = t * num_coils * 2.0 * np.pi
            x = r_coil * np.cos(angle)
            z = r_coil * np.sin(angle)
            v_writer.addData3(x, y, z)
            c_writer.addData4(*gold_spring)

        for i in range(num_pts - 1):
            lines.addVertices(v_base + i, v_base + i + 1)

        geom = Geom(vdata)
        geom.addPrimitive(lines)
        node = GeomNode("SpringMechanismNode")
        node.addGeom(geom)
        return NodePath(node)


class Dynamic3DTrajectoryRibbon:
    """
    Renders an illuminated real-time 3D trajectory trail ribbon following moving entities.
    """

    def __init__(self, render_node: NodePath, max_points: int = 400, color: Tuple[float, float, float, float] = (0.2, 0.8, 1.0, 0.9)):
        self.render = render_node
        self.max_points = max_points
        self.color = color
        self.points: List[Point3] = []
        self.trail_np: Optional[NodePath] = None

    def add_point(self, pos: np.ndarray) -> None:
        """Adds world position (X, Y, Z) and updates 3D line geometry."""
        # Global Y-up to Panda3D (X, Z, Y)
        p = Point3(pos[0], pos[2], pos[1])
        if len(self.points) > 0:
            dist = (p - self.points[-1]).length()
            if dist < 0.3:
                return
        self.points.append(p)
        if len(self.points) > self.max_points:
            self.points.pop(0)

        self._rebuild_geometry()

    def clear(self) -> None:
        self.points.clear()
        if self.trail_np:
            self.trail_np.removeNode()
            self.trail_np = None

    def _rebuild_geometry(self) -> None:
        if len(self.points) < 2:
            return

        vdata = GeomVertexData("TrailMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        lines = GeomLines(Geom.UHDynamic)

        n = len(self.points)
        for i, p in enumerate(self.points):
            alpha = (i / float(n)) * self.color[3]
            v_writer.addData3(p)
            c_writer.addData4(self.color[0], self.color[1], self.color[2], alpha)

        for i in range(n - 1):
            lines.addVertices(i, i + 1)

        geom = Geom(vdata)
        geom.addPrimitive(lines)
        node = GeomNode("DynamicTrajectoryNode")
        node.addGeom(geom)

        if self.trail_np:
            self.trail_np.removeNode()
        self.trail_np = self.render.attachNewNode(node)
