# PyAero3D — User Guide & Operator Manual

Welcome to **PyAero3D**, a general-purpose 3D multi-physics simulation engine and high-fidelity aerospace sandbox.

---

## 🎮 Desktop Flight Controls & Keybindings

| Keybinding | Action | Physical Description |
|---|---|---|
| `W` / `S` | **Elevator Pitch Down / Up** | Rotates horizontal tail elevators ($\delta_e$) to adjust Angle of Attack ($\alpha$). |
| `A` / `D` | **Aileron Roll Left / Right** | Deflects wing ailerons ($\delta_a$) to induce rolling moment $L_{roll}$ for banked turns. |
| `Q` / `E` | **Rudder Yaw Left / Right** | Deflects vertical fin rudder ($\delta_r$) to control sideslip angle ($\beta$). |
| `Shift` / `Ctrl` | **Throttle Increase / Decrease** | Modulates jet engine thrust [0% to 100%] with afterburner staging. |
| `Space` | **Wheel Brakes** | Engages dynamic Coulomb braking friction ($\mu = 0.85$) on landing wheels. |
| `Tab` | **Cycle Camera Mode** | Toggles between **Chase Cam**, **Cockpit Cam**, and **Free Orbit Cam**. |
| `U` | **Toggle Metric / Imperial** | Switches speedometer & altimeter between **Imperial** ($kt, ft, fpm$) and **Metric** ($km/h, m, m/s$). |
| `H` | **Toggle User Guide** | Displays/hides on-screen interactive control manual and HUD help overlay. |
| `[1]` | **Spawn Fighter Jet** | Spawns twin-engine delta fighter jet on runway threshold ready for takeoff. |
| `[2]` | **Spawn Mountain Drone** | Spawns agile 6-DOF quadrotor drone hovering over an alpine mountain ridge. |
| `[3]` | **Spawn Cargo Parachute** | Spawns heavy military supply crate dropped from 3,000m MSL altitude. |
| `[4]` | **Spawn Launch Rocket** | Spawns multi-stage rocket with TVC engine gimbaling on launchpad. |
| `F` | **Kinetic Breakup** | Triggers Class 11 structural fragmentation into balanced kinetic debris shards. |
| `F1` | **Reset World** | Clears all entities in the world and resets simulation to initial state. |

---

## 🏔️ The 3D Mountain Sandbox Environment

- **World Dimensions**: Continuous $12\text{km} \times 12\text{km}$ world with $2,400\text{m}$ alpine mountain summits.
- **Runway Corridor**: A flat asphalt runway ($2,400\text{m} \times 60\text{m}$) situated in the central valley basin.
- **GPU Continuous Level of Detail (CLOD)**: Panda3D `ShaderTerrainMesh` quadtree terrain running custom GLSL shaders with real-time gradient normal calculation, multi-layer slope blending (Grass, Rock Slate, Alpine Snow), and atmospheric Rayleigh/Mie scattering fog.

---

## 🛩️ Aerodynamic Flight Principles

### 1. Lift & Stall Mechanics
$$L = \frac{1}{2} \rho(h) v^2 S C_L(\alpha, \delta_e)$$
- Subsonic linear lift regime: $C_L(\alpha) = C_{L0} + C_{L\alpha} \alpha + 0.45 \delta_e$.
- Post-stall angle: $\alpha_{stall} = 16^\circ$ ($0.279\text{ rad}$). Beyond stall, lift decays sinusoidally.

### 2. Total Drag & Induced Drag
$$D = \frac{1}{2} \rho(h) v^2 S C_D = \frac{1}{2} \rho(h) v^2 S \left( C_{D0} + \frac{C_L^2}{\pi e AR} \right)$$
- Aspect Ratio: $AR = \frac{b^2}{S} = \frac{10.5^2}{28.0} = 3.94$.
- Oswald efficiency: $e = 0.82$.

### 3. Compressible Shockwave Drag Rise
- Subsonic ($M < 0.8$): Prandtl-Glauert compressibility correction: $C_{L, comp} = \frac{C_L}{\sqrt{1 - M^2}}$.
- Transonic ($0.8 \le M < 1.2$): Peak wave drag divergence from supersonic shockwave formation.
- Supersonic ($M \ge 1.2$): Ackeret wave drag decay: $C_D \propto \frac{1}{\sqrt{M^2 - 1}}$.

---

## 💥 Mountain Collision & Class 11 Kinetic Breakup

When an aircraft collides with a mountain cliff at impact speed $v_{impact} > 38\text{ m/s}$, the engine automatically triggers **Class 11 Mass-Weighted Structural Fragmentation**:
$$\sum_{k=1}^M m_k \Delta \mathbf{v}_k = \mathbf{0} \quad (\text{Residual Momentum } < 10^{-14}\text{ kg}\cdot\text{m/s})$$
Debris shards inherit parent velocity plus balanced isotropic dispersion impulses, tumbling down mountain slopes with bounce restitution and Coulomb friction.
