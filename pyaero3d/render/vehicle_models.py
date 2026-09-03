"""
PyAero3D - Procedural 3D CAD Vehicle Meshes & Visual Actor Generators.
Builds high-detail geometries for Fighter Jet, Quadrotor Drone, Cargo Parachute, Rocket, and Debris Shards.
"""

import numpy as np
from panda3d.core import (
    NodePath,
    Geom,
    GeomNode,
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    GeomTriangles,
    Vec4,
    Vec3,
)

from pyaero3d.core.types import EntityType
from pyaero3d.render.mesh_primitives import MeshPrimitiveBuilder


class VehicleModelBuilder:
    """
    Constructs high-detail procedural 3D models for the Panda3D scene graph.
    """

    @staticmethod
    def create_fighter_jet() -> NodePath:
        """
        Builds high-performance twin-engine fighter jet with delta wings and canted vertical tails.
        """
        vdata = GeomVertexData("JetMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        # Body colors
        fuselage_col = (0.75, 0.78, 0.82, 1.0) # Naval aircraft grey (#CBD2D9)
        wing_col     = (0.55, 0.58, 0.62, 1.0) # Darker wing surfaces
        canopy_col   = (0.10, 0.35, 0.65, 0.8) # Tinted gold/blue cockpit glass
        exhaust_col  = (1.00, 0.55, 0.10, 1.0) # Glowing orange afterburner

        vertices = [
            # 0: Nose tip (Forward = +Z in body frame)
            (0.0, 0.0, 7.5),
            # 1..4: Cockpit canopy section
            (0.7, 0.3, 3.5), (-0.7, 0.3, 3.5), (0.0, 1.2, 4.0), (0.0, -0.4, 3.5),
            # 5..8: Mid fuselage & Wing roots
            (1.2, 0.0, 0.0), (-1.2, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, -0.6, 0.0),
            # 9, 10: Left & Right Delta Wingtips
            (5.5, 0.0, -2.5), (-5.5, 0.0, -2.5),
            # 11, 12: Wing trailing edge root
            (1.4, 0.0, -4.5), (-1.4, 0.0, -4.5),
            # 13, 14: Twin canted vertical tailfins
            (1.5, 2.4, -4.8), (-1.5, 2.4, -4.8),
            # 15, 16: Tail exhaust nozzles
            (0.6, -0.2, -5.5), (-0.6, -0.2, -5.5),
        ]

        for idx, (x, y, z) in enumerate(vertices):
            v_writer.addData3(x, y, z)
            if idx == 3: # Canopy top
                c_writer.addData4(*canopy_col)
            elif idx in (15, 16): # Exhausts
                c_writer.addData4(*exhaust_col)
            elif idx in (9, 10): # Wingtips
                c_writer.addData4(*wing_col)
            else:
                c_writer.addData4(*fuselage_col)

        triangle_indices = [
            # Nose cone
            (0, 3, 1), (0, 2, 3), (0, 1, 4), (0, 4, 2),
            # Canopy to mid fuselage
            (3, 7, 5), (3, 5, 1), (3, 6, 7), (3, 2, 6),
            (4, 1, 5), (4, 5, 8), (4, 8, 6), (4, 6, 2),
            # Right Wing (+X)
            (5, 9, 11), (1, 9, 5),
            # Left Wing (-X)
            (6, 12, 10), (2, 6, 10),
            # Twin Tailfins
            (11, 13, 15), (12, 16, 14),
            # Aft Fuselage closure
            (7, 15, 16), (7, 16, 8), (7, 8, 15),
        ]

        for (i0, i1, i2) in triangle_indices:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1) # Two-sided

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("FighterJetNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_quadrotor_drone() -> NodePath:
        """
        Builds 6-DOF Quadrotor Drone with carbon-fiber X-arms and 4 rotor discs.
        """
        vdata = GeomVertexData("DroneMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        body_col  = (0.15, 0.15, 0.16, 1.0) # Matte carbon fiber
        pod_col   = (0.85, 0.20, 0.20, 1.0) # Red flight controller top
        prop_col  = (0.10, 0.85, 0.95, 0.7) # Cyan translucent spinning props

        arm = 0.45 # Arm span (m)
        r_p = 0.22 # Propeller radius

        vertices = [
            # 0..3: Central Avionics Body Hub
            (0.15, 0.05, 0.15), (-0.15, 0.05, 0.15),
            (-0.15, 0.05, -0.15), (0.15, 0.05, -0.15),
            # 4: Top LED/GPS dome
            (0.0, 0.18, 0.0),
            # 5..8: 4 Motor Hubs
            (arm, 0.02, arm), (-arm, 0.02, arm),
            (-arm, 0.02, -arm), (arm, 0.02, -arm),
        ]

        for idx, (x, y, z) in enumerate(vertices):
            v_writer.addData3(x, y, z)
            c_writer.addData4(*(pod_col if idx == 4 else body_col))

        tri_list = [
            # Hub top pyramid
            (4, 0, 1), (4, 1, 2), (4, 2, 3), (4, 3, 0),
            # Hub bottom
            (0, 2, 1), (0, 3, 2),
            # X-Arms
            (0, 5, 1), (1, 6, 2), (2, 7, 3), (3, 8, 0),
        ]

        # Add 4 circular propeller discs
        base_idx = len(vertices)
        for m_i, (mx, my, mz) in enumerate([(arm, 0.05, arm), (-arm, 0.05, arm), (-arm, 0.05, -arm), (arm, 0.05, -arm)]):
            center_v = base_idx
            v_writer.addData3(mx, my, mz)
            c_writer.addData4(*prop_col)
            base_idx += 1

            for s in range(8):
                ang = (s / 8.0) * (2.0 * np.pi)
                px = mx + r_p * np.cos(ang)
                pz = mz + r_p * np.sin(ang)
                v_writer.addData3(px, my, pz)
                c_writer.addData4(*prop_col)

                s_curr = center_v + 1 + s
                s_next = center_v + 1 + ((s + 1) % 8)
                tri_list.append((center_v, s_curr, s_next))

            base_idx += 8

        for (i0, i1, i2) in tri_list:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("QuadrotorNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_cargo_parachute() -> NodePath:
        """
        Builds military cargo crate with inflated hemispherical parachute canopy.
        """
        vdata = GeomVertexData("CargoMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        crate_col = (0.55, 0.45, 0.25, 1.0) # Khaki military crate
        canopy_col = (0.95, 0.55, 0.15, 1.0) # Hi-vis rescue orange canopy
        line_col = (0.85, 0.85, 0.85, 0.5)

        # 1. Supply Crate Box (-1m to +1m)
        s = 0.8
        corners = [
            (s, s, s), (-s, s, s), (-s, -s, s), (s, -s, s),
            (s, s, -s), (-s, s, -s), (-s, -s, -s), (s, -s, -s),
        ]
        for (x, y, z) in corners:
            v_writer.addData3(x, y, z)
            c_writer.addData4(*crate_col)

        box_tris = [
            (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
            (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
        ]

        # 2. Parachute Canopy (Hemisphere suspended 6m above crate)
        base_v = 8
        canopy_y = 7.0
        canopy_r = 4.0
        # Apex vertex
        v_writer.addData3(0.0, canopy_y + 2.0, 0.0)
        c_writer.addData4(*canopy_col)
        apex_idx = base_v
        base_v += 1

        ring_indices = []
        for s_i in range(12):
            ang = (s_i / 12.0) * (2.0 * np.pi)
            cx = canopy_r * np.cos(ang)
            cz = canopy_r * np.sin(ang)
            v_writer.addData3(cx, canopy_y, cz)
            c_writer.addData4(*canopy_col)
            ring_indices.append(base_v)
            base_v += 1

        for s_i in range(12):
            c_curr = ring_indices[s_i]
            c_next = ring_indices[(s_i + 1) % 12]
            box_tris.append((apex_idx, c_curr, c_next))
            # Suspension line from canopy rim to crate top
            corner_idx = s_i % 4
            box_tris.append((c_curr, c_next, corner_idx))

        for (i0, i1, i2) in box_tris:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("CargoParachuteNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_rocket() -> NodePath:
        """
        Builds slender multi-stage launch rocket with 4 grid fins.
        """
        vdata = GeomVertexData("RocketMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        white_col = (0.95, 0.95, 0.96, 1.0)
        black_col = (0.15, 0.15, 0.18, 1.0)
        flame_col = (1.0, 0.45, 0.05, 1.0)

        length = 14.0 # 14m rocket
        radius = 0.9

        # Nose cone tip
        v_writer.addData3(0.0, length * 0.5 + 3.0, 0.0)
        c_writer.addData4(*white_col)

        # Upper stage ring
        v_writer.addData3(radius, length * 0.5, 0.0)
        v_writer.addData3(0.0, length * 0.5, radius)
        v_writer.addData3(-radius, length * 0.5, 0.0)
        v_writer.addData3(0.0, length * 0.5, -radius)
        for _ in range(4): c_writer.addData4(*white_col)

        # Base stage ring
        v_writer.addData3(radius, -length * 0.5, 0.0)
        v_writer.addData3(0.0, -length * 0.5, radius)
        v_writer.addData3(-radius, -length * 0.5, 0.0)
        v_writer.addData3(0.0, -length * 0.5, -radius)
        for _ in range(4): c_writer.addData4(*black_col)

        # Exhaust plume tip
        v_writer.addData3(0.0, -length * 0.5 - 4.5, 0.0)
        c_writer.addData4(*flame_col)

        # 4 Fins
        fin_v = [
            (radius + 1.2, -length * 0.5, 0.0),
            (0.0, -length * 0.5, radius + 1.2),
            (-radius - 1.2, -length * 0.5, 0.0),
            (0.0, -length * 0.5, -radius - 1.2),
        ]
        for (fx, fy, fz) in fin_v:
            v_writer.addData3(fx, fy, fz)
            c_writer.addData4(*black_col)

        tri_list = [
            # Nose cone
            (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
            # Fuselage cylinder sides
            (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
            (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
            # Engine exhaust flame cone
            (9, 6, 5), (9, 7, 6), (9, 8, 7), (9, 5, 8),
            # Fins
            (1, 5, 10), (2, 6, 11), (3, 7, 12), (4, 8, 13),
        ]

        for (i0, i1, i2) in tri_list:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("RocketNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_debris_shard() -> NodePath:
        """
        Builds jagged kinetic fragment debris piece.
        """
        vdata = GeomVertexData("ShardMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        col = (0.85, 0.40, 0.15, 1.0) # Burning orange/metal shard
        pts = [
            (0.0, 0.4, 0.0), (-0.3, -0.2, 0.2), (0.3, -0.2, 0.2),
            (0.0, -0.2, -0.3), (0.0, -0.5, 0.0)
        ]
        for p in pts:
            v_writer.addData3(*p)
            c_writer.addData4(*col)

        t_idx = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (4, 2, 1), (4, 3, 2), (4, 1, 3)]
        for (i0, i1, i2) in t_idx:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("ShardNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_cannon_projectile() -> NodePath:
        """
        Builds aerodynamic ballistic artillery shell with copper driving band.
        """
        vdata = GeomVertexData("CannonMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        steel_col = (0.35, 0.40, 0.45, 1.0)
        copper_col = (0.85, 0.50, 0.20, 1.0)
        tip_col = (0.90, 0.90, 0.95, 1.0)

        # Ogive nose tip (+Z)
        v_writer.addData3(0.0, 0.0, 2.8)
        c_writer.addData4(*tip_col)

        r = 0.55
        # Mid body ring
        v_writer.addData3(r, 0.0, 1.0)
        v_writer.addData3(0.0, r, 1.0)
        v_writer.addData3(-r, 0.0, 1.0)
        v_writer.addData3(0.0, -r, 1.0)
        for _ in range(4): c_writer.addData4(*steel_col)

        # Base ring
        v_writer.addData3(r, 0.0, -1.5)
        v_writer.addData3(0.0, r, -1.5)
        v_writer.addData3(-r, 0.0, -1.5)
        v_writer.addData3(0.0, -r, -1.5)
        for _ in range(4): c_writer.addData4(*copper_col)

        # Base cap
        v_writer.addData3(0.0, 0.0, -1.8)
        c_writer.addData4(*steel_col)

        tri_list = [
            (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
            (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
            (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
            (9, 6, 5), (9, 7, 6), (9, 8, 7), (9, 5, 8),
        ]
        for (i0, i1, i2) in tri_list:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("CannonProjectileNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_airfoil_wing() -> NodePath:
        """
        Builds aerodynamic glider test wing vehicle with swept wings and T-tail.
        """
        vdata = GeomVertexData("WingMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        white_col = (0.92, 0.94, 0.96, 1.0)
        blue_col = (0.15, 0.50, 0.90, 1.0)

        # Slender glider fuselage & high-aspect wings
        pts = [
            (0.0, 0.0, 4.5), # 0: Nose
            (0.4, 0.2, 1.0), (-0.4, 0.2, 1.0), (0.0, -0.3, 1.0), # 1,2,3
            (7.5, 0.3, -0.5), (-7.5, 0.3, -0.5), # 4,5: Wingtips
            (0.3, 0.0, -3.5), (-0.3, 0.0, -3.5), # 6,7: Tail root
            (0.0, 1.6, -3.8), # 8: T-tail top
            (1.8, 1.6, -4.0), (-1.8, 1.6, -4.0), # 9,10: T-tail horizontal tips
        ]
        for idx, (x, y, z) in enumerate(pts):
            v_writer.addData3(x, y, z)
            if idx in (4, 5, 9, 10):
                c_writer.addData4(*blue_col)
            else:
                c_writer.addData4(*white_col)

        tri_list = [
            (0, 1, 2), (0, 2, 3), (0, 3, 1),
            (1, 4, 6), (2, 7, 5), (1, 6, 2), (2, 6, 7),
            (6, 8, 7), (8, 9, 10), (8, 10, 7), (8, 6, 9),
        ]
        for (i0, i1, i2) in tri_list:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("AirfoilWingNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_satellite() -> NodePath:
        """
        Builds orbital satellite with gold foil bus, twin solar arrays, and high-gain dish.
        """
        vdata = GeomVertexData("SatMesh", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        gold_col = (0.95, 0.75, 0.20, 1.0) # Gold MLI insulation
        solar_col = (0.10, 0.25, 0.70, 1.0) # Dark blue photovoltaic cells
        dish_col = (0.85, 0.88, 0.90, 1.0)

        # Central bus cube
        s = 0.8
        for x in [-s, s]:
            for y in [-s, s]:
                for z in [-s, s]:
                    v_writer.addData3(x, y, z)
                    c_writer.addData4(*gold_col)

        # Solar panels (Left & Right)
        # Left array (+X)
        v_writer.addData3(s + 3.0, 0.6, 0.0)
        v_writer.addData3(s + 3.0, -0.6, 0.0)
        v_writer.addData3(s, -0.6, 0.0)
        v_writer.addData3(s, 0.6, 0.0)
        # Right array (-X)
        v_writer.addData3(-s - 3.0, 0.6, 0.0)
        v_writer.addData3(-s - 3.0, -0.6, 0.0)
        v_writer.addData3(-s, -0.6, 0.0)
        v_writer.addData3(-s, 0.6, 0.0)
        for _ in range(8): c_writer.addData4(*solar_col)

        # High gain dish (+Z)
        v_writer.addData3(0.0, 0.0, s + 1.2)
        v_writer.addData3(0.7, 0.7, s + 0.4)
        v_writer.addData3(-0.7, 0.7, s + 0.4)
        v_writer.addData3(-0.7, -0.7, s + 0.4)
        v_writer.addData3(0.7, -0.7, s + 0.4)
        for _ in range(5): c_writer.addData4(*dish_col)

        tri_list = [
            # Cube
            (0, 1, 2), (1, 3, 2), (4, 6, 5), (5, 6, 7),
            (0, 4, 1), (1, 4, 5), (2, 3, 6), (3, 7, 6),
            (0, 2, 4), (2, 6, 4), (1, 5, 3), (3, 5, 7),
            # Solar panels
            (8, 9, 10), (8, 10, 11), (12, 14, 13), (12, 15, 14),
            # Dish
            (16, 17, 18), (16, 18, 19), (16, 19, 20), (16, 20, 17),
        ]
        for (i0, i1, i2) in tri_list:
            tris.addVertices(i0, i1, i2)
            tris.addVertices(i0, i2, i1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("SatelliteNode")
        node.addGeom(geom)
        return NodePath(node)

    @staticmethod
    def create_cannon_projectile() -> NodePath:
        """Builds high-detail smooth artillery projectile with cylindrical body and ogive nose."""
        parent = NodePath("ArtilleryShell")
        body = MeshPrimitiveBuilder.build_cylinder(
            radius=0.45, length=2.2, axis="z", segments=20, color=(0.35, 0.40, 0.45, 1.0), name="ShellBody"
        )
        body.reparentTo(parent)

        nose = MeshPrimitiveBuilder.build_uv_sphere(
            radius=0.45, rings=14, sectors=20, color=(0.90, 0.92, 0.95, 1.0), name="ShellNose"
        )
        nose.reparentTo(parent)
        nose.setPos(0.0, 0.0, 1.1)

        band = MeshPrimitiveBuilder.build_cylinder(
            radius=0.48, length=0.35, axis="z", segments=20, color=(0.85, 0.50, 0.20, 1.0), name="DrivingBand"
        )
        band.reparentTo(parent)
        band.setPos(0.0, 0.0, -0.6)
        return parent

    @staticmethod
    def create_double_pendulum_rods() -> NodePath:
        """
        Builds high-detail articulated double pendulum with smooth thin cylinders and UV spheres.
        """
        parent = NodePath("DoublePendulumAssembly")

        # Upper Rod (L = 3.5m, r = 0.07m, chrome)
        rod1 = MeshPrimitiveBuilder.build_cylinder(
            radius=0.07, length=3.5, axis="z", segments=20, color=(0.85, 0.88, 0.92, 1.0), name="Rod1"
        )
        rod1.reparentTo(parent)
        rod1.setPos(0.0, 0.0, -1.75)

        # Upper Bob 1 (Smooth UV Sphere, Brass / Amber Gold, R = 0.55m)
        bob1 = MeshPrimitiveBuilder.build_uv_sphere(
            radius=0.55, rings=16, sectors=24, color=(0.95, 0.65, 0.15, 1.0), name="Bob1"
        )
        bob1.reparentTo(parent)
        bob1.setPos(0.0, 0.0, -3.5)

        # Lower Rod (L = 3.5m, r = 0.07m, chrome)
        rod2 = MeshPrimitiveBuilder.build_cylinder(
            radius=0.07, length=3.5, axis="z", segments=20, color=(0.75, 0.80, 0.85, 1.0), name="Rod2"
        )
        rod2.reparentTo(parent)
        rod2.setPos(0.0, 0.0, -5.25)

        # Lower Bob 2 (Smooth UV Sphere, Neon Electric Blue / Cyan, R = 0.65m)
        bob2 = MeshPrimitiveBuilder.build_uv_sphere(
            radius=0.65, rings=16, sectors=24, color=(0.15, 0.85, 1.0, 1.0), name="Bob2"
        )
        bob2.reparentTo(parent)
        bob2.setPos(0.0, 0.0, -7.0)

        return parent

    @staticmethod
    def create_cyclotron_chamber() -> NodePath:
        """Builds electromagnetic cyclotron field chamber with glowing central particle."""
        parent = NodePath("CyclotronAssembly")
        particle = MeshPrimitiveBuilder.build_uv_sphere(
            radius=0.6, rings=16, sectors=20, color=(0.10, 0.95, 0.65, 1.0), name="CycloParticle"
        )
        particle.reparentTo(parent)

        ring = MeshPrimitiveBuilder.build_helical_spring(
            radius=2.5, length=1.2, coils=4, wire_radius=0.08, color=(0.20, 0.50, 0.85, 0.9)
        )
        ring.reparentTo(parent)
        return parent

    @staticmethod
    def create_bouncing_sphere() -> NodePath:
        """Builds smooth high-poly UV sphere with glowing core."""
        return MeshPrimitiveBuilder.build_uv_sphere(
            radius=1.2, rings=20, sectors=28, color=(0.95, 0.75, 0.10, 1.0), name="BouncingSphere"
        )

    @staticmethod
    def create_physics_crate(size: float = 2.0) -> NodePath:
        """Builds an industrial shipping crate with surface normals."""
        parent = NodePath("PhysicsCrate")
        half = size * 0.5
        vdata = GeomVertexData("CrateMesh", GeomVertexFormat.getV3n3c4(), Geom.UHDynamic)
        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        c_writer = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        col = (0.72, 0.52, 0.28, 1.0)
        faces = [
            ([(half, half, -half), (-half, half, -half), (-half, half, half), (half, half, half)], (0, 1, 0)),
            ([(-half, -half, -half), (half, -half, -half), (half, -half, half), (-half, -half, half)], (0, -1, 0)),
            ([(-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half)], (0, 0, 1)),
            ([(-half, half, -half), (half, half, -half), (half, -half, -half), (-half, -half, -half)], (0, 0, -1)),
            ([(half, -half, -half), (half, half, -half), (half, half, half), (half, -half, half)], (1, 0, 0)),
            ([(-half, half, -half), (-half, -half, -half), (-half, -half, half), (-half, half, half)], (-1, 0, 0)),
        ]
        v_idx = 0
        for quad_verts, norm in faces:
            for (x, y, z) in quad_verts:
                v_writer.addData3(x, y, z)
                n_writer.addData3(*norm)
                c_writer.addData4(*col)
            tris.addVertices(v_idx, v_idx + 1, v_idx + 2)
            tris.addVertices(v_idx, v_idx + 2, v_idx + 3)
            v_idx += 4

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("CrateGeomNode")
        node.addGeom(geom)
        np_node = NodePath(node)
        np_node.reparentTo(parent)
        return parent

    create_quadrotor = create_quadrotor_drone
