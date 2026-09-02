"""
PyAero3D - 1000Hz Threaded Simulation Physics Engine Runner.
Runs at 1000Hz fixed-dt on dedicated worker thread with atomic double-buffered state snapshots.
"""

import time
import threading
from typing import Dict, Any, List, Optional
import numpy as np

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer
from pyaero3d.core.quaternion_math import SpatialQuaternion
from pyaero3d.physics.flight_dynamics import FlightDynamicsSolver
from pyaero3d.physics.mountain_collision import MountainCollisionEngine
from pyaero3d.physics.fragmentation import FragmentationEngine


class PhysicsEngineThread:
    """
    1000Hz Background Physics Worker Thread.
    """

    def __init__(self, state_buffer: StateBuffer, collision_engine: MountainCollisionEngine, target_hz: float = 1000.0):
        self.buffer = state_buffer
        self.collision = collision_engine
        self.target_hz = target_hz
        self.fixed_dt = 1.0 / target_hz

        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Double-buffer snapshot container for zero-lock 60-144Hz visual rendering
        self.snapshot_data = np.zeros_like(self.buffer.data)
        self.step_count = 0
        self.actual_hz = 0.0
        self.crashed_entities: List[int] = []

    def start(self) -> None:
        """Starts 1000Hz background thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PyAero3D-Physics")
            self._thread.start()

    def stop(self) -> None:
        """Stops background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_paused(self) -> bool:
        return self._paused

    def get_snapshot(self) -> np.ndarray:
        """Returns atomic copy of current state tensor for visual rendering."""
        with self._lock:
            return self.snapshot_data.copy()

    def get_render_snapshot(self) -> np.ndarray:
        """Alias for get_snapshot."""
        return self.get_snapshot()

    @property
    def effective_hz(self) -> float:
        return self.actual_hz

    def step_single(self, dt: float) -> None:
        """Executes a single physics integration step (called in loop or manually)."""
        mask = self.buffer.get_active_mask()
        active_indices = np.where(mask)[0]

        half_dt = 0.5 * dt

        # 1. First Leapfrog Velocity Half-Step & Position Full-Step
        for idx in active_indices:
            row = self.buffer.data[idx]
            m = row[StateIdx.MASS]
            if m < 1e-6:
                continue
            inv_m = 1.0 / m

            # Velocity half-step from previous accumulated forces
            ax = row[StateIdx.FX] * inv_m
            ay = row[StateIdx.FY] * inv_m
            az = row[StateIdx.FZ] * inv_m

            row[StateIdx.VX] += ax * half_dt
            row[StateIdx.VY] += ay * half_dt
            row[StateIdx.VZ] += az * half_dt

            # Position full-step
            row[StateIdx.PX] += row[StateIdx.VX] * dt
            row[StateIdx.PY] += row[StateIdx.VY] * dt
            row[StateIdx.PZ] += row[StateIdx.VZ] * dt

            # Quaternion attitude step
            omega = row[StateIdx.WX:StateIdx.WZ + 1]
            quat = row[StateIdx.QW:StateIdx.QZ + 1]
            row[StateIdx.QW:StateIdx.QZ + 1] = SpatialQuaternion.integrate_quaternion(quat, omega, dt)

            # Reset forces for accumulation in this step
            row[StateIdx.FX:StateIdx.FZ + 1] = 0.0
            row[StateIdx.TX:StateIdx.TZ + 1] = 0.0

        # 2. Evaluate Aerodynamics, Engine Thrust & Gravity
        for idx in active_indices:
            row = self.buffer.data[idx]
            f_tot, tau_tot = FlightDynamicsSolver.evaluate_entity_dynamics(row, dt)
            row[StateIdx.FX:StateIdx.FZ + 1] += f_tot
            row[StateIdx.TX:StateIdx.TZ + 1] += tau_tot

        # 3. Resolve Mountain Terrain Collision & Kinetic Crashes
        crashed_this_step = []
        for idx in active_indices:
            row = self.buffer.data[idx]
            is_ground, is_crash = self.collision.resolve_entity_collision(row, dt)
            if is_crash:
                crashed_this_step.append(idx)

        # 4. Process Mountain Impact Breakups
        for crash_idx in crashed_this_step:
            FragmentationEngine.explode_entity(self.buffer, crash_idx, num_shards=10, dispersion_energy_j=150000.0)

        # 5. Second Leapfrog Velocity Half-Step
        mask = self.buffer.get_active_mask()
        active_indices = np.where(mask)[0]
        for idx in active_indices:
            row = self.buffer.data[idx]
            m = row[StateIdx.MASS]
            if m < 1e-6:
                continue
            inv_m = 1.0 / m

            ax = row[StateIdx.FX] * inv_m
            ay = row[StateIdx.FY] * inv_m
            az = row[StateIdx.FZ] * inv_m

            row[StateIdx.VX] += ax * half_dt
            row[StateIdx.VY] += ay * half_dt
            row[StateIdx.VZ] += az * half_dt

            # Angular acceleration: alpha = I^-1 * tau (simplified diagonal inertia)
            inertia = max(0.1, m * (row[StateIdx.RADIUS] ** 2) * 0.4)
            row[StateIdx.WX:StateIdx.WZ + 1] += (row[StateIdx.TX:StateIdx.TZ + 1] / inertia) * dt

        self.step_count += 1

    def _run_loop(self) -> None:
        """1000Hz fixed-step background loop."""
        last_time = time.perf_counter()
        last_hz_calc = last_time
        steps_since_calc = 0

        while self._running:
            if self._paused:
                time.sleep(0.01)
                last_time = time.perf_counter()
                continue

            now = time.perf_counter()
            elapsed = now - last_time

            if elapsed >= self.fixed_dt:
                with self._lock:
                    self.step_single(self.fixed_dt)
                    self.snapshot_data[:] = self.buffer.data[:]

                last_time = now
                steps_since_calc += 1

                if (now - last_hz_calc) >= 1.0:
                    self.actual_hz = steps_since_calc / (now - last_hz_calc)
                    steps_since_calc = 0
                    last_hz_calc = now
            else:
                # Sleep briefly to yield CPU
                sleep_sec = max(0.0, (self.fixed_dt - elapsed) * 0.5)
                if sleep_sec > 0.0001:
                    time.sleep(sleep_sec)
