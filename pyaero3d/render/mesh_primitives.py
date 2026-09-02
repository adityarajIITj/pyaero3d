"""
PyAero3D - High-Detail Parametric 3D Mesh Primitives.
Generates smooth UV Spheres, Thin Cylinders, Helical Springs, and Aerodynamic Cones.
"""

from typing import Tuple, List, Optional
import numpy as np
from panda3d.core import (
    NodePath, Geom, GeomNode, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, GeomTriangles, Vec4, Vec3, Point3
)


class MeshPrimitiveBuilder:
    """
    Constructs high-detail smooth procedural 3D geometric meshes.
    """

    @staticmethod
    def build_uv_sphere(
        radius: float = 1.0,
        rings: int = 16,
        sectors: int = 24,
        color: Tuple[float, float, float, float] = (0.9, 0.5, 0.1, 1.0),
        name: str = "UVSphere",
    ) -> NodePath:
        """Constructs a smooth, high-poly UV sphere mesh."""
        vdata = GeomVertexData(name, GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(GeM := Geom.UHDynamic)

        # Generate vertices
        for r in range(rings + 1):
            theta = (r / rings) * np.pi  # 0 to pi (lat)
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)

            for s in range(sectors + 1):
                phi = (s / sectors) * (2.0 * np.pi)  # 0 to 2pi (lon)
                sin_p = np.sin(phi)
                cos_p = np.cos(phi)

                x = radius * sin_t * cos_p
                y = radius * cos_t
                z = radius * sin_t * sin_p

                v_writer.addData3(x, y, z)
                c_writer.addData4(*color)

        # Generate triangle indices
        for r in range(rings):
            for s in range(sectors):
                i0 = r * (sectors + 1) + s
                i1 = i0 + 1
                i2 = (r + 1) * (sectors + 1) + s
                i3 = i2 + 1

                tris.addVertices(i0, i2, i1)
                tris.addVertices(i1, i2, i3)
                tris.addVertices(i0, i1, i2)
                tris.addVertices(i1, i3, i2)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(f"{name}Node")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def build_cylinder(
        radius: float = 0.15,
        length: float = 4.0,
        axis: str = "y",  # "x", "y", or "z"
        segments: int = 20,
        color: Tuple[float, float, float, float] = (0.75, 0.80, 0.85, 1.0),
        name: str = "Cylinder",
    ) -> NodePath:
        """Constructs a smooth, thin cylindrical rod mesh with end caps."""
        vdata = GeomVertexData(name, GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        half_len = length * 0.5
        v_base = 0

        # Bottom cap center
        if axis == "y":
            v_writer.addData3(0, -half_len, 0)
        elif axis == "z":
            v_writer.addData3(0, 0, -half_len)
        else:
            v_writer.addData3(-half_len, 0, 0)
        c_writer.addData4(*color)
        bottom_center = 0
        v_base += 1

        # Top cap center
        if axis == "y":
            v_writer.addData3(0, half_len, 0)
        elif axis == "z":
            v_writer.addData3(0, 0, half_len)
        else:
            v_writer.addData3(half_len, 0, 0)
        c_writer.addData4(*color)
        top_center = 1
        v_base += 1

        # Generate ring vertices
        bottom_ring = []
        top_ring = []

        for i in range(segments):
            angle = (i / float(segments)) * (2.0 * np.pi)
            ca = radius * np.cos(angle)
            sa = radius * np.sin(angle)

            if axis == "y":
                # Bottom
                v_writer.addData3(ca, -half_len, sa)
                c_writer.addData4(*color)
                bottom_ring.append(v_base)
                v_base += 1

                # Top
                v_writer.addData3(ca, half_len, sa)
                c_writer.addData4(*color)
                top_ring.append(v_base)
                v_base += 1
            elif axis == "z":
                v_writer.addData3(ca, sa, -half_len)
                c_writer.addData4(*color)
                bottom_ring.append(v_base)
                v_base += 1

                v_writer.addData3(ca, sa, half_len)
                c_writer.addData4(*color)
                top_ring.append(v_base)
                v_base += 1
            else:
                v_writer.addData3(-half_len, ca, sa)
                c_writer.addData4(*color)
                bottom_ring.append(v_base)
                v_base += 1

                v_writer.addData3(half_len, ca, sa)
                c_writer.addData4(*color)
                top_ring.append(v_base)
                v_base += 1

        # Triangles for sides and caps
        for i in range(segments):
            next_i = (i + 1) % segments
            b0 = bottom_ring[i]
            b1 = bottom_ring[next_i]
            t0 = top_ring[i]
            t1 = top_ring[next_i]

            # Side quad
            tris.addVertices(b0, t0, b1)
            tris.addVertices(b1, t0, t1)
            tris.addVertices(b0, b1, t0)
            tris.addVertices(b1, t1, t0)

            # Bottom cap
            tris.addVertices(bottom_center, b1, b0)
            # Top cap
            tris.addVertices(top_center, t0, t1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(f"{name}Node")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def build_helical_spring(
        radius: float = 0.8,
        length: float = 6.0,
        coils: int = 8,
        wire_radius: float = 0.08,
        color: Tuple[float, float, float, float] = (0.95, 0.75, 0.20, 1.0),
    ) -> NodePath:
        """Constructs a smooth 3D helical coil spring."""
        vdata = GeomVertexData("SpringMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        num_steps = coils * 18
        half_l = length * 0.5

        prev_ring = None
        v_idx = 0

        for i in range(num_steps + 1):
            t = i / float(num_steps)
            y = -half_l + length * t
            angle = t * coils * 2.0 * np.pi
            cx = radius * np.cos(angle)
            cz = radius * np.sin(angle)

            # Circular cross-section with 6 vertices
            curr_ring = []
            for j in range(6):
                w_ang = (j / 6.0) * 2.0 * np.pi
                wx = cx + wire_radius * np.cos(w_ang) * np.cos(angle)
                wy = y + wire_radius * np.sin(w_ang)
                wz = cz + wire_radius * np.cos(w_ang) * np.sin(angle)
                v_writer.addData3(wx, wy, wz)
                c_writer.addData4(*color)
                curr_ring.append(v_idx)
                v_idx += 1

            if prev_ring is not None:
                for j in range(6):
                    next_j = (j + 1) % 6
                    r0 = prev_ring[j]
                    r1 = prev_ring[next_j]
                    n0 = curr_ring[j]
                    n1 = curr_ring[next_j]

                    tris.addVertices(r0, n0, r1)
                    tris.addVertices(r1, n0, n1)
                    tris.addVertices(r0, r1, n0)
                    tris.addVertices(r1, n1, n0)

            prev_ring = curr_ring

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("HelicalSpringNode")
        node.addGeom(geom)
        return NodePath(node)
