"""
PyAero3D - Atmospheric Sky Dome, Airfield Infrastructure, and Living World Geometry Builders.
Includes Runway with Markings, ATC Tower, Aircraft Hangars, Windsock, and Scattered Valley Pine Trees.
"""

from typing import List, Tuple
import numpy as np
from panda3d.core import (
    NodePath,
    CardMaker,
    Geom,
    GeomNode,
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    GeomTriangles,
    Vec4,
    Vec3,
    Point3,
    Material,
)

from pyaero3d.render.mesh_primitives import MeshPrimitiveBuilder


class EnvironmentGeometryBuilder:
    """
    Constructs atmospheric sky dome, realistic airfield infrastructure, and valley scenery.
    """

    @staticmethod
    def create_runway_strip(
        length: float = 2600.0,
        width: float = 65.0,
        elevation: float = 1.0,
    ) -> NodePath:
        """
        Builds asphalt runway surface with white centerline dashes and threshold stripes.
        """
        parent = NodePath("AirfieldRunway")

        # 1. Main Asphalt Surface
        cm = CardMaker("RunwayAsphalt")
        cm.setFrame(-width * 0.5, width * 0.5, -length * 0.5, length * 0.5)
        asphalt_np = NodePath(cm.generate())
        asphalt_np.setP(-90)
        asphalt_np.setPos(0.0, 0.0, elevation)
        asphalt_np.setColor(Vec4(0.15, 0.16, 0.18, 1.0))
        asphalt_np.reparentTo(parent)

        # 2. Runway Edge Borders
        for side in (-1.0, 1.0):
            cm_edge = CardMaker(f"EdgeLine_{side}")
            cm_edge.setFrame(-0.8, 0.8, -length * 0.5, length * 0.5)
            edge_np = NodePath(cm_edge.generate())
            edge_np.setP(-90)
            edge_np.setPos(side * (width * 0.5 - 1.5), 0.0, elevation + 0.05)
            edge_np.setColor(Vec4(0.92, 0.92, 0.95, 1.0))
            edge_np.reparentTo(parent)

        # 3. Centerline Dashes (every 50m)
        dash_len = 25.0
        dash_width = 1.6
        dash_step = 60.0
        num_dashes = int(length / dash_step)
        for i in range(-num_dashes // 2, num_dashes // 2):
            cm_dash = CardMaker(f"CenterDash_{i}")
            cm_dash.setFrame(-dash_width * 0.5, dash_width * 0.5, -dash_len * 0.5, dash_len * 0.5)
            d_np = NodePath(cm_dash.generate())
            d_np.setP(-90)
            d_np.setPos(0.0, i * dash_step, elevation + 0.06)
            d_np.setColor(Vec4(0.95, 0.95, 0.98, 1.0))
            d_np.reparentTo(parent)

        # 4. Threshold Piano Keys at Ends
        for end_sign in (-1.0, 1.0):
            end_y = end_sign * (length * 0.5 - 45.0)
            for k in range(-6, 7):
                if k == 0: continue
                cm_key = CardMaker(f"ThreshKey_{end_sign}_{k}")
                cm_key.setFrame(-1.2, 1.2, -18.0, 18.0)
                k_np = NodePath(cm_key.generate())
                k_np.setP(-90)
                k_np.setPos(k * 4.2, end_y, elevation + 0.07)
                k_np.setColor(Vec4(0.95, 0.95, 0.95, 1.0))
                k_np.reparentTo(parent)

        return parent

    @staticmethod
    def create_atc_tower() -> NodePath:
        """Builds an Air Traffic Control Tower with observation cab and radar dish."""
        parent = NodePath("ATCTower")

        # Shaft (Concrete cylinder, h=45m, r=4.5m)
        shaft = MeshPrimitiveBuilder.build_cylinder(
            radius=4.5, length=42.0, axis="z", segments=20, color=(0.82, 0.84, 0.88, 1.0), name="TowerShaft"
        )
        shaft.reparentTo(parent)
        shaft.setPos(0.0, 0.0, 21.0)

        # Observation Cab (Tinted glass octagonal deck, r=7.5m, h=6m)
        cab = MeshPrimitiveBuilder.build_cylinder(
            radius=7.5, length=7.0, axis="z", segments=16, color=(0.20, 0.45, 0.70, 0.85), name="CabGlass"
        )
        cab.reparentTo(parent)
        cab.setPos(0.0, 0.0, 44.0)

        # Roof Dome & Antenna Mast
        roof = MeshPrimitiveBuilder.build_uv_sphere(
            radius=7.6, rings=10, sectors=16, color=(0.75, 0.78, 0.82, 1.0), name="CabRoof"
        )
        roof.reparentTo(parent)
        roof.setPos(0.0, 0.0, 48.5)
        roof.setSz(0.3)

        mast = MeshPrimitiveBuilder.build_cylinder(
            radius=0.25, length=8.0, axis="z", segments=8, color=(0.95, 0.25, 0.20, 1.0), name="RadarMast"
        )
        mast.reparentTo(parent)
        mast.setPos(0.0, 0.0, 54.0)

        return parent

    @staticmethod
    def create_hangar(width: float = 60.0, length: float = 75.0, height: float = 24.0) -> NodePath:
        """Builds an arched aviation hangar building."""
        parent = NodePath("Hangar")
        cm = CardMaker("HangarWalls")

        # Floor / Base Pad
        cm.setFrame(-width * 0.5 - 5, width * 0.5 + 5, -length * 0.5 - 5, length * 0.5 + 5)
        pad = NodePath(cm.generate())
        pad.setP(-90)
        pad.setColor(Vec4(0.35, 0.38, 0.42, 1.0))
        pad.setPos(0, 0, 0.2)
        pad.reparentTo(parent)

        # Main Arched Roof Cylinder
        roof = MeshPrimitiveBuilder.build_cylinder(
            radius=width * 0.5, length=length, axis="y", segments=20, color=(0.65, 0.68, 0.72, 1.0), name="HangarRoof"
        )
        roof.reparentTo(parent)
        roof.setPos(0.0, -length * 0.5, height * 0.5)
        roof.setSz(height / (width * 0.5))

        # Entrance Bay Frame
        door_frame = MeshPrimitiveBuilder.build_cylinder(
            radius=width * 0.45, length=4.0, axis="y", segments=16, color=(0.18, 0.20, 0.24, 1.0), name="DoorOpening"
        )
        door_frame.reparentTo(parent)
        door_frame.setPos(0.0, length * 0.48, height * 0.45)
        door_frame.setSz(height * 0.85 / (width * 0.45))

        return parent

    @staticmethod
    def create_pine_tree(height: float = 9.0) -> NodePath:
        """Builds a low-poly pine tree with trunk and conical foliage."""
        tree = NodePath("PineTree")

        # Trunk (Brown cylinder)
        trunk = MeshPrimitiveBuilder.build_cylinder(
            radius=0.35, length=height * 0.35, axis="z", segments=8, color=(0.40, 0.26, 0.16, 1.0), name="Trunk"
        )
        trunk.reparentTo(tree)
        trunk.setPos(0.0, 0.0, height * 0.175)

        # 3 Tiered Foliage Cones
        foliage_colors = [
            (0.15, 0.38, 0.18, 1.0), # Dark forest green
            (0.18, 0.44, 0.22, 1.0), # Mid green
            (0.22, 0.50, 0.26, 1.0), # Top bright needle green
        ]
        for tier in range(3):
            t_h = height * (0.45 - tier * 0.08)
            t_r = height * (0.32 - tier * 0.07)
            t_z = height * (0.28 + tier * 0.22)

            cone = MeshPrimitiveBuilder.build_uv_sphere(
                radius=t_r, rings=8, sectors=10, color=foliage_colors[tier], name=f"CanopyTier_{tier}"
            )
            cone.reparentTo(tree)
            cone.setPos(0.0, 0.0, t_z)
            cone.setSz(1.6)

        return tree

    @staticmethod
    def create_sky_dome(radius: float = 35000.0, rings: int = 20, segments: int = 36) -> NodePath:
        """
        Builds hemisphere sky dome with vertical daylight color gradient:
        Atmospheric cyan horizon (#89CFF0) fading into deep alpine blue (#0E2A47).
        """
        vdata = GeomVertexData("SkyDome", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(GeM := Geom.UHDynamic)

        for i in range(rings + 1):
            theta = (i / rings) * (np.pi * 0.5)
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            t = i / rings

            # Realistic daylight sky gradient
            c_r = 0.58 * (1.0 - t) + 0.08 * t
            c_g = 0.74 * (1.0 - t) + 0.18 * t
            c_b = 0.94 * (1.0 - t) + 0.45 * t

            for j in range(segments + 1):
                phi = (j / segments) * (np.pi * 2.0)
                x = radius * cos_t * np.cos(phi)
                y = radius * cos_t * np.sin(phi)
                z = radius * sin_t

                vertex.addData3(x, y, z)
                color.addData4(c_r, c_g, c_b, 1.0)

        for i in range(rings):
            for j in range(segments):
                i0 = i * (segments + 1) + j
                i1 = (i + 1) * (segments + 1) + j
                i2 = (i + 1) * (segments + 1) + (j + 1)
                i3 = i * (segments + 1) + (j + 1)

                tris.addVertices(i0, i2, i1)
                tris.addVertices(i0, i3, i2)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("SkyDomeNode")
        node.addGeom(geom)

        sky_np = NodePath(node)
        sky_np.setTwoSided(True)
        return sky_np
