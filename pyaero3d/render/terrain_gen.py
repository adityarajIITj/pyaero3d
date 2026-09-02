"""
PyAero3D - Procedural Mountain Terrain Generation & Continuous Heightfield Sampler.
Generates multi-octave Fractal Brownian Motion (FBM) heightfields for GPU ShaderTerrainMesh
and provides continuous bilinear elevation queries h(x, z) for physics collisions.
"""

from typing import Tuple
import numpy as np
from panda3d.core import PNMImage, PerlinNoise2


class MountainTerrainGenerator:
    """
    Multi-Scale Alpine Mountain Terrain Generator.
    Produces both Panda3D PNMImage heightfield textures and CPU collision height arrays.
    """

    def __init__(
        self,
        grid_resolution: int = 512,      # Power-of-2 (e.g. 256, 512, 1024)
        world_size_m: float = 12000.0,   # 12km x 12km continuous world
        max_height_m: float = 2400.0,    # 2.4km alpine summit peaks
        seed: int = 42,
    ):
        self.grid_res = grid_resolution
        self.world_size = world_size_m
        self.max_height = max_height_m
        self.seed = seed
        self.dx = world_size_m / (grid_resolution - 1)

        # Precompute height matrix (grid_res x grid_res)
        self.height_matrix = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        self._generate_fbm_terrain()

    def _generate_fbm_terrain(self) -> None:
        """
        Synthesizes multi-octave Fractal Brownian Motion (FBM) with ridge sharpening.
        """
        res = self.grid_res
        xs = np.linspace(-0.5, 0.5, res)
        zs = np.linspace(-0.5, 0.5, res)
        X, Z = np.meshgrid(xs, zs)

        # 1. Base continental mountain ridge noise (Low freq, high amplitude)
        n1 = np.sin(3.0 * X + 1.2) * np.cos(3.0 * Z + 0.8) * 0.4
        # 2. Alpine sharp ridge noise (Ridged multifractal)
        n2 = 1.0 - np.abs(np.sin(7.0 * X + 2.0 * Z) * np.cos(6.0 * Z - 1.5 * X))
        n2 = (n2 ** 2.0) * 0.35
        # 3. High-frequency jagged crags & rock formations
        n3 = np.sin(18.0 * X) * np.sin(18.0 * Z) * 0.15
        # 4. Micro-texture rough slope details
        n4 = (np.sin(42.0 * X + 10.0 * Z) + np.cos(42.0 * Z - 10.0 * X)) * 0.05

        # 5. Natural valley depression in center for airport/runway corridor
        center_dist = np.sqrt(X**2 + Z**2)
        valley_mask = np.clip((center_dist - 0.08) / 0.35, 0.0, 1.0)

        elevation_raw = (n1 + n2 + n3 + n4)
        elevation_norm = (elevation_raw - np.min(elevation_raw)) / (np.max(elevation_raw) - np.min(elevation_raw))

        # Shape mountains around flat central airfield basin
        elevation_final = elevation_norm * (0.2 + 0.8 * valley_mask)
        # Ensure flat runway zone around (0, 0)
        runway_corridor = np.exp(-((X / 0.04)**4 + (Z / 0.15)**4))
        elevation_final = elevation_final * (1.0 - runway_corridor * 0.95)

        self.height_matrix = (elevation_final * self.max_height).astype(np.float32)

    def create_pnm_image(self) -> PNMImage:
        """
        Converts normalized height matrix to 16-bit Panda3D PNMImage.
        """
        res = self.grid_res
        img = PNMImage(res, res, 1)  # 1-channel grayscale
        img.setMaxval(65535)         # 16-bit depth

        norm_matrix = np.clip(self.height_matrix / self.max_height, 0.0, 1.0)
        for y in range(res):
            for x in range(res):
                val = float(norm_matrix[y, x])
                img.setGray(x, y, val)
        return img

    def build_shader_terrain(self, render_node, camera_node):
        """
        Constructs and attaches Panda3D ShaderTerrainMesh with GPU continuous LOD.
        """
        from panda3d.core import ShaderTerrainMesh, Texture, SamplerState, NodePath, Shader, Filename
        import os

        pnm = self.create_pnm_image()
        height_tex = Texture("MountainHeightmap")
        height_tex.load(pnm)
        height_tex.setMagfilter(SamplerState.FT_linear)
        height_tex.setMinfilter(SamplerState.FT_linear)
        height_tex.setWrapU(SamplerState.WM_clamp)
        height_tex.setWrapV(SamplerState.WM_clamp)

        terrain_mesh = ShaderTerrainMesh()
        terrain_mesh.heightfield = height_tex
        terrain_mesh.target_triangle_width = 10.0
        terrain_mesh.generate()

        terrain_np = render_node.attachNewNode(terrain_mesh)
        terrain_np.setScale(self.world_size, self.world_size, self.max_height)
        terrain_np.setPos(-self.world_size * 0.5, -self.world_size * 0.5, 0.0)

        # Load GLSL Shaders if present
        v_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shaders", "mountain_terrain.vert.glsl"))
        f_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shaders", "mountain_terrain.frag.glsl"))
        if os.path.exists(v_path) and os.path.exists(f_path):
            try:
                shader = Shader.load(Shader.SL_GLSL, Filename.fromOsSpecific(v_path), Filename.fromOsSpecific(f_path))
                terrain_np.setShader(shader)
                terrain_np.set_shader_input("camera", camera_node)
            except Exception as e:
                print(f"[PyAero3D] Shader warning: {e}")

        return terrain_np

    def get_height(self, world_x: float, world_z: float) -> float:
        """
        Bilinear interpolation continuous height query h(x, z) in meters.
        """
        if np.isnan(world_x) or np.isnan(world_z) or np.isinf(world_x) or np.isinf(world_z):
            return 0.0

        half_world = self.world_size * 0.5
        inv_dx = 1.0 / self.dx
        gx = (world_x + half_world) * inv_dx
        gz = (world_z + half_world) * inv_dx

        if np.isnan(gx) or np.isnan(gz):
            return 0.0

        ix0 = int(np.clip(gx, 0, self.grid_res - 2))
        iz0 = int(np.clip(gz, 0, self.grid_res - 2))
        ix1 = ix0 + 1
        iz1 = iz0 + 1

        fx = np.clip(gx - ix0, 0.0, 1.0)
        fz = np.clip(gz - iz0, 0.0, 1.0)

        mat = self.height_matrix
        h00 = float(mat[iz0, ix0])
        h10 = float(mat[iz0, ix1])
        h01 = float(mat[iz1, ix0])
        h11 = float(mat[iz1, ix1])

        h0 = h00 + (h10 - h00) * fx
        h1 = h01 + (h11 - h01) * fx
        return float(h0 + (h1 - h0) * fz)

    def get_height_vectorized(self, world_xs: np.ndarray, world_zs: np.ndarray) -> np.ndarray:
        """
        Vectorized bilinear height query for N entities simultaneously.
        """
        half_world = self.world_size * 0.5
        inv_dx = 1.0 / self.dx
        gxs = (world_xs + half_world) * inv_dx
        gzs = (world_zs + half_world) * inv_dx

        ix0 = np.clip(np.floor(gxs).astype(np.int32), 0, self.grid_res - 2)
        iz0 = np.clip(np.floor(gzs).astype(np.int32), 0, self.grid_res - 2)
        ix1 = ix0 + 1
        iz1 = iz0 + 1

        fx = gxs - ix0
        fz = gzs - iz0

        mat = self.height_matrix
        h00 = mat[iz0, ix0]
        h10 = mat[iz0, ix1]
        h01 = mat[iz1, ix0]
        h11 = mat[iz1, ix1]

        h0 = h00 + (h10 - h00) * fx
        h1 = h01 + (h11 - h01) * fx
        return h0 + (h1 - h0) * fz

    def get_surface_normal(self, world_x: float, world_z: float, delta: float = 2.0) -> np.ndarray:
        """
        Computes analytical surface normal vector n(x, z) for physics slope reaction forces.
        """
        h_left = self.get_height(world_x - delta, world_z)
        h_right = self.get_height(world_x + delta, world_z)
        h_back = self.get_height(world_x, world_z - delta)
        h_forward = self.get_height(world_x, world_z + delta)

        # Gradient vectors: Y-up viewport (dh/dx, 1, dh/dz)
        dh_dx = (h_right - h_left) / (2.0 * delta)
        dh_dz = (h_forward - h_back) / (2.0 * delta)

        normal = np.array([-dh_dx, 1.0, -dh_dz], dtype=np.float64)
        norm_len = np.linalg.norm(normal)
        return normal / (norm_len if norm_len > 1e-6 else 1.0)
