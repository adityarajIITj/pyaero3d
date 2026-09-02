"""
PyAero3D Performance & Invariant Conservation Benchmark Suite.
Tests 1000Hz simulation throughput, jitter, and Class 11 momentum balance.
"""

import time
import numpy as np

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer
from pyaero3d.render.terrain_gen import MountainTerrainGenerator
from pyaero3d.physics.mountain_collision import MountainCollisionEngine
from pyaero3d.physics.flight_dynamics import FlightDynamicsSolver
from pyaero3d.physics.fragmentation import FragmentationEngine


def benchmark_1000hz_physics_loop(num_steps: int = 10000) -> None:
    print("=" * 80)
    print(f"BENCHMARK 1: 1000Hz Physics & Collision Loop ({num_steps:,d} steps)")
    print("=" * 80)

    terrain = MountainTerrainGenerator(grid_resolution=256, world_size_m=6000.0, max_height_m=1800.0)
    collision = MountainCollisionEngine(terrain)
    buffer = StateBuffer(max_entities=500)

    # Spawn realistic operational fleet (4 jets, 4 drones, 2 rockets)
    for i in range(4):
        buffer.allocate_entity(
            entity_type=EntityType.FIXED_WING_JET,
            mass=14000.0,
            position=np.array([i * 50.0, 1500.0 + i * 20.0, -500.0 + i * 30.0]),
            velocity=np.array([0.0, 0.0, 220.0]),
            cd=0.024, area=28.0,
        )
    for i in range(4):
        buffer.allocate_entity(
            entity_type=EntityType.QUADROTOR_DRONE,
            mass=1.8,
            position=np.array([-500.0 + i * 40.0, 800.0, i * 60.0]),
            velocity=np.array([5.0, 0.0, 10.0]),
            cd=0.45, area=0.08,
        )
    for i in range(2):
        buffer.allocate_entity(
            entity_type=EntityType.MULTI_STAGE_ROCKET,
            mass=28000.0,
            position=np.array([i * 100.0, 50.0, i * 100.0]),
            velocity=np.array([0.0, 150.0, 0.0]),
            fuel_mass=20000.0,
        )

    dt = 0.001 # 1000Hz time step
    latencies = []

    t0 = time.perf_counter()
    for step in range(num_steps):
        s_t0 = time.perf_counter()

        mask = buffer.get_active_mask()
        active_idx = np.where(mask)[0]

        # 1. Aerodynamics & Dynamics
        for idx in active_idx:
            row = buffer.data[idx]
            f_tot, tau = FlightDynamicsSolver.evaluate_entity_dynamics(row, dt)
            row[StateIdx.FX:StateIdx.FZ + 1] += f_tot
            row[StateIdx.TX:StateIdx.TZ + 1] += tau

        # 2. Vectorized Integration
        masses = buffer.data[active_idx, StateIdx.MASS, np.newaxis]
        inv_masses = 1.0 / np.maximum(masses, 1e-6)
        forces = buffer.data[active_idx, StateIdx.FX:StateIdx.FZ + 1]

        buffer.data[active_idx, StateIdx.VX:StateIdx.VZ + 1] += (forces * inv_masses) * dt
        buffer.data[active_idx, StateIdx.PX:StateIdx.PZ + 1] += buffer.data[active_idx, StateIdx.VX:StateIdx.VZ + 1] * dt
        buffer.data[active_idx, StateIdx.FX:StateIdx.FZ + 1] = 0.0

        # 3. Vectorized Terrain Altitude & Collision Check
        xs = buffer.data[active_idx, StateIdx.PX]
        zs = buffer.data[active_idx, StateIdx.PZ]
        ground_hs = terrain.get_height_vectorized(xs, zs)
        radii = buffer.data[active_idx, StateIdx.RADIUS]

        penetrations = (ground_hs + radii) - buffer.data[active_idx, StateIdx.PY]
        collided = penetrations > 0.0
        if np.any(collided):
            col_idx = active_idx[collided]
            buffer.data[col_idx, StateIdx.PY] = ground_hs[collided] + radii[collided]
            buffer.data[col_idx, StateIdx.VY] = np.maximum(0.0, buffer.data[col_idx, StateIdx.VY] * -0.15)
            buffer.data[col_idx, StateIdx.ON_GROUND] = 1.0

        s_t1 = time.perf_counter()
        latencies.append((s_t1 - s_t0) * 1000.0) # ms

    t1 = time.perf_counter()
    total_sec = t1 - t0
    avg_ms = np.mean(latencies)
    peak_ms = np.max(latencies)
    std_ms = np.std(latencies)
    sim_rate = num_steps / total_sec

    print(f"  Active Entities      : {buffer.active_count}")
    print(f"  Total Duration       : {total_sec:.4f} s")
    print(f"  Average Step Latency : {avg_ms:.4f} ms")
    print(f"  Peak Step Latency    : {peak_ms:.4f} ms")
    print(f"  Jitter Std-Dev       : {std_ms:.4f} ms")
    print(f"  Simulation Rate      : {sim_rate:,.1f} Hz (Target: 1,000 Hz, Speedup: {sim_rate/1000.0:.2f}x)")
    print(f"  Status               : {'PASSED (> 1000 Hz)' if sim_rate >= 1000.0 else 'FAILED'}")


def benchmark_class11_momentum_conservation(num_trials: int = 500) -> None:
    print("\n" + "=" * 80)
    print(f"BENCHMARK 2: Class 11 Fragmentation Momentum Conservation ({num_trials} trials)")
    print("=" * 80)

    buffer = StateBuffer(max_entities=200)
    max_rel_residual = 0.0

    for trial in range(num_trials):
        p_mass = np.random.uniform(200.0, 50000.0)
        p_vel = np.random.uniform(-300.0, 300.0, size=3)
        p_pos = np.random.uniform(-5000.0, 5000.0, size=3)

        p_idx = buffer.allocate_entity(
            entity_type=EntityType.FIXED_WING_JET,
            mass=p_mass,
            position=p_pos,
            velocity=p_vel,
        )

        p_init_momentum = p_mass * p_vel
        num_shards = np.random.randint(6, 24)

        shards = FragmentationEngine.explode_entity(
            buffer, p_idx, num_shards=num_shards, dispersion_energy_j=500000.0
        )

        p_shard_sum = np.zeros(3, dtype=np.float64)
        for s in shards:
            m_s = buffer.data[s, StateIdx.MASS]
            v_s = buffer.data[s, StateIdx.VX:StateIdx.VZ + 1]
            p_shard_sum += m_s * v_s

        res = float(np.linalg.norm(p_shard_sum - p_init_momentum))
        rel_res = res / max(1.0, float(np.linalg.norm(p_init_momentum)))
        if rel_res > max_rel_residual:
            max_rel_residual = rel_res

        for s in shards:
            buffer.free_entity(s)

    print(f"  Trials Executed      : {num_trials}")
    print(f"  Max Relative Error   : {max_rel_residual:.2e}")
    print(f"  Strictly Conserved   : {max_rel_residual < 1e-12}")
    print(f"  Status               : {'PASSED' if max_rel_residual < 1e-12 else 'FAILED'}")


if __name__ == "__main__":
    benchmark_1000hz_physics_loop(num_steps=5000)
    benchmark_class11_momentum_conservation(num_trials=500)
