"""
PyAero3D - Atmospheric Sky Dome and Runway Strip Geometry Builders.
"""

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
)


class EnvironmentGeometryBuilder:
    """
    Constructs atmospheric sky dome and precision runway geometry for Panda3D scene graph.
    """

    @staticmethod
    def create_runway_strip(
        length: float = 2400.0,
        width: float = 60.0,
        elevation: float = 1.0,
    ) -> NodePath:
        """
        Builds asphalt runway surface mesh centered at (0, 0, elevation).
        """
        cm = CardMaker("RunwayAsphalt")
        cm.setFrame(-length * 0.5, length * 0.5, -width * 0.5, width * 0.5)
        runway_np = NodePath(cm.generate())
        runway_np.setP(-90)  # Orient flat on ground (Z-up in Panda3D)
        runway_np.setPos(0.0, 0.0, elevation)
        runway_np.setColor(Vec4(0.12, 0.12, 0.14, 1.0))  # Dark asphalt
        return runway_np

    @staticmethod
    def create_sky_dome(radius: float = 25000.0, rings: int = 16, segments: int = 32) -> NodePath:
        """
        Builds hemisphere sky dome with vertical color gradient:
        Horizon cyan/blue (#89CFF0) fading into zenith deep blue (#0B132B).
        """
        vdata = GeomVertexData("SkyDome", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")

        tris = GeomTriangles(Geom.UHDynamic)

        for i in range(rings + 1):
            theta = (i / rings) * (np.pi * 0.5) # 0 to 90 deg elevation
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            t_factor = i / rings

            # Gradient from horizon to zenith
            c_r = 0.55 * (1.0 - t_factor) + 0.05 * t_factor
            c_g = 0.72 * (1.0 - t_factor) + 0.10 * t_factor
            c_b = 0.92 * (1.0 - t_factor) + 0.35 * t_factor

            for j in range(segments + 1):
                phi = (j / segments) * (np.pi * 2.0)
                sin_p = np.sin(phi)
                cos_p = np.cos(phi)

                x = radius * cos_t * cos_p
                y = radius * cos_t * sin_p
                z = radius * sin_t

                vertex.addData3(x, y, z)
                color.addData4(c_r, c_g, c_b, 1.0)

        for i in range(rings):
            for j in range(segments):
                i0 = i * (segments + 1) + j
                i1 = (i + 1) * (segments + 1) + j
                i2 = (i + 1) * (segments + 1) + (j + 1)
                i3 = i * (segments + 1) + (j + 1)

                # Inward facing winding order
                tris.addVertices(i0, i2, i1)
                tris.addVertices(i0, i3, i2)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("SkyDomeNode")
        node.addGeom(geom)

        sky_np = NodePath(node)
        sky_np.setTwoSided(True)
        return sky_np
