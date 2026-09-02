# PyAero3D Technical Architecture & Engineering Specification

## 1. System Architecture Pipeline

```
+---------------------------------------------------------------------------------------------------+
|                        GPU PROCEDURAL TERRAIN & RENDERING PIPELINE (60 - 144 Hz)                  |
|  - GPU Quadtree CLOD: Panda3D `ShaderTerrainMesh` (512x512 multi-octave FBM elevation grid)      |
|  - Custom GLSL Shaders: Real-time analytical normal computation, slope tri-texture blending       |
|    (Valleys: Grass #2E5339, Steep Cliffs: Slate #4A4E51, High Peaks: Snow #E8F0FE)                |
|  - Atmospheric Scattering Fog: Rayleigh/Mie exponential depth extinction & horizon gradient       |
|  - Continuous CPU Heightfield h(x, z) & surface normals n(x, z) for sub-millisecond queries        |
+---------------------------------------------------------------------------------------------------+
                                                  │
                Atomic State Synchronization via Double-Buffered Tensor Snapshot
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                    1000Hz UNIFIED GENERAL-PURPOSE & AEROSPACE PHYSICS CORE                        |
|  - Contiguous 32-Stride State Tensor `StateBuffer` (Zero GC allocation in hot loop)               |
|  - Spatial Quaternions q = [w, x, y, z] & Direction Cosine Matrix (No Gimbal Lock)                |
|  - Rigid Body 6-DOF Dynamics: Newton-Euler, 3x3 Moment of Inertia Tensor I_body, Euler equations  |
|  - GJK / EPA Narrowphase Convex Collision Detection (Spheres, Boxes, Capsules, Convex Hulls)      |
|  - Projected Gauss-Seidel (PGS) Sequential Impulse Constraint Solver                              |
|  - Multi-Body Articulated Joints: Ball-and-Socket, Revolute Hinge, Spring-Dampers                 |
|  - Universal Force Generators: Gravity fields, Fluid Buoyancy, Aero Lift/Drag, Explosions         |
|  - Class 11 Fragmentation Engine: Exact Zero-Residual Linear Momentum Conservation               |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                        NATIVE HIGH-PERFORMANCE C++ CORE & C-ABI EXPORTS                           |
|  - `pyaero3d_physics.h` & `pyaero3d_physics.cpp`: C++17 SIMD Vectorized Leapfrog Integrator       |
|  - Direct C-ABI bindings via `ctypes` / `cffi` for 2000Hz+ multi-core execution                  |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Rigid Body Dynamics & Newton-Euler Formulation

Each 6-DOF rigid body is defined by:
- Mass $m$, inverse mass $m^{-1}$.
- Body-frame Moment of Inertia Tensor:
  $$\mathbf{I}_{body} = \begin{bmatrix} I_{xx} & -I_{xy} & -I_{xz} \\ -I_{xy} & I_{yy} & -I_{yz} \\ -I_{xz} & -I_{yz} & I_{zz} \end{bmatrix}$$
- World-space inverse inertia tensor updated at every orientation step:
  $$\mathbf{I}_{world}^{-1} = \mathbf{R}(\mathbf{q}) \mathbf{I}_{body}^{-1} \mathbf{R}(\mathbf{q})^T$$
- Rotational Euler equations:
  $$\dot{\mathbf{\omega}} = \mathbf{I}_{world}^{-1} \left( \mathbf{\tau}_{net} - \mathbf{\omega} \times (\mathbf{I}_{world} \mathbf{\omega}) \right)$$

---

## 3. GJK (Gilbert-Johnson-Keerthi) & EPA Narrowphase

For any two convex shapes $A$ and $B$, the Minkowski difference is defined as:
$$A \ominus B = \{ \mathbf{a} - \mathbf{b} \mid \mathbf{a} \in A, \mathbf{b} \in B \}$$
The support mapping function returns the extreme point in search direction $\mathbf{d}$:
$$S_{A \ominus B}(\mathbf{d}) = S_A(\mathbf{d}) - S_B(-\mathbf{d})$$
1. **GJK Phase**: Evolves a 3D simplex (Point, Line, Triangle, Tetrahedron) to determine if the origin $\mathbf{0} \in A \ominus B$.
2. **EPA Phase**: When overlap occurs, EPA expands the 3D polytope outwards along closest triangle faces until it finds the exact penetration depth $d$ and contact normal $\hat{\mathbf{n}}$.

---

## 4. Projected Gauss-Seidel (PGS) Constraint Solver

Given contact point $\mathbf{p}$, normal $\hat{\mathbf{n}}$, and tangent basis $\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2$:
- **Relative Contact Velocity**: $\mathbf{v}_{rel} = \mathbf{v}_B(\mathbf{p}) - \mathbf{v}_A(\mathbf{p})$.
- **Normal Constraint**: $v_n = \mathbf{v}_{rel} \cdot \hat{\mathbf{n}} \ge 0$.
- **Effective Mass**:
  $$K_n = m_A^{-1} + m_B^{-1} + (\mathbf{r}_A \times \hat{\mathbf{n}})^T \mathbf{I}_A^{-1} (\mathbf{r}_A \times \hat{\mathbf{n}}) + (\mathbf{r}_B \times \hat{\mathbf{n}})^T \mathbf{I}_B^{-1} (\mathbf{r}_B \times \hat{\mathbf{n}})$$
- **Sequential Impulse with Baumgarte Stabilization**:
  $$\Delta \lambda_n = K_n^{-1} \left( -(v_n - \text{bias}) \right), \quad \lambda_n = \max(0, \lambda_n^{old} + \Delta \lambda_n)$$
- **Coulomb Friction Cone Pyramid**:
  $$-\mu \lambda_n \le \lambda_{t1, t2} \le \mu \lambda_n$$
