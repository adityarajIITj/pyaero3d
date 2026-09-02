"""
PyAero3D - High-Detail Parametric 3D Mesh Primitives with Vertex Normals & Shading.
Generates smooth UV Spheres, Thin Cylinders, Helical Springs, and Aerodynamic Cones with realistic lighting.
"""

from typing import Tuple, List, Optional
import numpy as np
from panda3d.core import (
    NodePath, Geom, GeomNode, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, GeomTriangles, Vec4, Vec3, Point3, Material
)


class MeshPrimitiveBuilder:
    """
    Constructs high-detail smooth procedural 3D geometric meshes with surface normals and metallic materials.
    """

    @staticmethod
    def build_uv_sphere(
        radius: float = 1.0,
        rings: int = 24,
        sectors: int = 32,
        color: Tuple[float, float, float, float] = (0.95, 0.65, 0.15, 1.0),
        name: str = "UVSphere",
    ) -> NodePath:
        """Constructs a smooth, high-poly UV sphere mesh with accurate radial normals."""
        vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        for r in range(rings + 1):
            theta = (r / rings) * np.pi  # 0 to pi (lat)
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)

            for s in range(sectors + 1):
                phi = (s / sectors) * (2.0 * np.pi)  # 0 to 2pi (lon)
                sin_p = np.sin(phi)
                cos_p = np.cos(phi)

                nx = sin_t * cos_p
                ny = cos_t
                nz = sin_t * sin_p

                x = radius * nx
                y = radius * ny
                z = radius * nz

                v_writer.addData3(x, z, y)  # Panda3D: Z is Up
                n_writer.addData3(nx, nz, ny)
                c_writer.addData4(*color)

        for r in range(rings):
            for s in range(sectors):
                i0 = r * (sectors + 1) + s
                i1 = i0 + 1
                i2 = (r + 1) * (sectors + 1) + s
                i3 = i2 + 1

                tris.addVertices(i0, i2, i1)
                tris.addVertices(i1, i2, i3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(f"{name}Node")
        node.addGeom(geom)
        np_node = NodePath(node)

        # Apply metallic material
        mat = Material()
        mat.setSpecular((0.8, 0.8, 0.8, 1.0))
        mat.setShininess(45.0)
        np_node.setMaterial(mat, 1)
        return np_node

    @staticmethod
    def build_cylinder(
        radius: float = 0.08,
        length: float = 3.5,
        axis: str = "y",
        segments: int = 24,
        color: Tuple[float, float, float, float] = (0.80, 0.85, 0.90, 1.0),
        name: str = "Cylinder",
    ) -> NodePath:
        """
        Constructs a smooth cylindrical rod mesh with end caps and surface normals.
        """
        vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        v_base = 0

        # Cap Centers
        if axis == "y":
            v_writer.addData3(0.0, 0.0, 0.0)
            n_writer.addData3(0.0, -1.0, 0.0)
            c_writer.addData4(*color)

            v_writer.addData3(0.0, length, 0.0)
            n_writer.addData3(0.0, 1.0, 0.0)
            c_writer.addData4(*color)
        elif axis == "z":
            v_writer.addData3(0.0, 0.0, -length * 0.5)
            n_writer.addData3(0.0, 0.0, -1.0)
            c_writer.addData4(*color)

            v_writer.addData3(0.0, 0.0, length * 0.5)
            n_writer.addData3(0.0, 0.0, 1.0)
            c_writer.addData4(*color)
        else:
            v_writer.addData3(-length * 0.5, 0.0, 0.0)
            n_writer.addData3(-1.0, 0.0, 0.0)
            c_writer.addData4(*color)

            v_writer.addData3(length * 0.5, 0.0, 0.0)
            n_writer.addData3(1.0, 0.0, 0.0)
            c_writer.addData4(*color)

        bottom_center = 0
        top_center = 1
        v_base = 2

        bottom_ring = []
        top_ring = []

        for i in range(segments + 1):
            angle = (i / float(segments)) * (2.0 * np.pi)
            ca = np.cos(angle)
            sa = np.sin(angle)

            if axis == "y":
                v_writer.addData3(radius * ca, 0.0, radius * sa)
                n_writer.addData3(ca, 0.0, sa)
                c_writer.addData4(*color)
                bottom_ring.append(v_base); v_base += 1

                v_writer.addData3(radius * ca, length, radius * sa)
                n_writer.addData3(ca, 0.0, sa)
                c_writer.addData4(*color)
                top_ring.append(v_base); v_base += 1
            elif axis == "z":
                v_writer.addData3(radius * ca, radius * sa, -length * 0.5)
                n_writer.addData3(ca, sa, 0.0)
                c_writer.addData4(*color)
                bottom_ring.append(v_base); v_base += 1

                v_writer.addData3(radius * ca, radius * sa, length * 0.5)
                n_writer.addData3(ca, sa, 0.0)
                c_writer.addData4(*color)
                top_ring.append(v_base); v_base += 1
            else:
                v_writer.addData3(-length * 0.5, radius * ca, radius * sa)
                n_writer.addData3(0.0, ca, sa)
                c_writer.addData4(*color)
                bottom_ring.append(v_base); v_base += 1

                v_writer.addData3(length * 0.5, radius * ca, radius * sa)
                n_writer.addData3(0.0, ca, sa)
                c_writer.addData4(*color)
                top_ring.append(v_base); v_base += 1

        for i in range(segments):
            b0 = bottom_ring[i]
            b1 = bottom_ring[i + 1]
            t0 = top_ring[i]
            t1 = top_ring[i + 1]

            # Side Quad
            tris.addVertices(b0, b1, t0)
            tris.addVertices(t0, b1, t1)

            # Bottom Cap
            tris.addVertices(bottom_center, b0, b1)
            # Top Cap
            tris.addVertices(top_center, t1, t0)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(f"{name}Node")
        node.addGeom(geom)
        np_node = NodePath(node)

        # Chrome/metallic material
        mat = Material()
        mat.setSpecular((0.9, 0.9, 1.0, 1.0))
        mat.setShininess(60.0)
        np_node.setMaterial(mat, 1)
        return np_node

    @staticmethod
    def build_helical_spring(
        radius: float = 0.8,
        length: float = 6.0,
        coils: int = 9,
        wire_radius: float = 0.08,
        color: Tuple[float, float, float, float] = (0.95, 0.75, 0.20, 1.0),
    ) -> NodePath:
        """Constructs a smooth 3D helical coil spring with surface normals."""
        vdata = GeomVertexData("SpringMesh", GeomVertexFormat.getV3n3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        num_steps = coils * 20
        v_idx = 0
        prev_ring = None

        for i in range(num_steps + 1):
            t = i / float(num_steps)
            z = -length * 0.5 + length * t  # Z is Up
            angle = t * coils * 2.0 * np.pi
            cx = radius * np.cos(angle)
            cy = radius * np.sin(angle)

            curr_ring = []
            for j in range(8):
                w_ang = (j / 8.0) * 2.0 * np.pi
                nx = np.cos(w_ang) * np.cos(angle)
                ny = np.cos(w_ang) * np.sin(angle)
                nz = np.sin(w_ang)

                wx = cx + wire_radius * nx
                wy = cy + wire_radius * ny
                wz = z + wire_radius * nz

                v_writer.addData3(wx, wy, wz)
                n_writer.addData3(nx, ny, nz)
                c_writer.addData4(*color)
                curr_ring.append(v_idx)
                v_idx += 1

            if prev_ring is not None:
                for j in range(8):
                    next_j = (j + 1) % 8
                    r0 = prev_ring[j]
                    r1 = prev_ring[next_j]
                    n0 = curr_ring[j]
                    n1 = curr_ring[next_j]

                    tris.addVertices(r0, n0, r1)
                    tris.addVertices(r1, n0, n1)

            prev_ring = curr_ring

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("HelicalSpringNode")
        node.addGeom(geom)
        np_node = NodePath(node)

        mat = Material()
        mat.setSpecular((0.8, 0.8, 0.4, 1.0))
        mat.setShininess(50.0)
        np_node.setMaterial(mat, 1)
        return np_node
