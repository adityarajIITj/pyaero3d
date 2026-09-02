"""
PyAero3D - Contiguous Tensor State Buffer Management.
Pre-allocated flat float64 array with O(1) allocation/deallocation pool.
"""

from typing import List, Optional, Tuple
import numpy as np

from pyaero3d.core.types import StateIdx, EntityType, STRIDE_LEN


class StateBuffer:
    """
    Contiguous 2D NumPy array of shape (max_entities, 32), dtype=float64.
    Ensures C-contiguous memory layout for high-speed SIMD vectorization.
    """

    def __init__(self, max_entities: int = 10000):
        self.max_entities = max_entities
        self.data = np.zeros((max_entities, STRIDE_LEN), dtype=np.float64, order="C")
        self._free_indices: List[int] = list(reversed(range(max_entities)))
        self._active_indices: List[int] = []

    @property
    def active_count(self) -> int:
        """Returns number of currently allocated active entities."""
        return len(self._active_indices)

    def allocate_entity(
        self,
        entity_type: EntityType,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float = 1.0,
        cd: float = 0.05,
        area: float = 1.0,
        quaternion: Optional[np.ndarray] = None,
        fuel_mass: float = 0.0,
    ) -> int:
        """
        Allocates a slot in the state buffer in O(1) time and populates initial conditions.
        """
        if not self._free_indices:
            raise RuntimeError(f"StateBuffer overflow: exceeded capacity of {self.max_entities} entities")

        idx = self._free_indices.pop()
        self._active_indices.append(idx)

        # Clear memory slot
        self.data[idx, :] = 0.0

        # Set entity fields
        self.data[idx, StateIdx.MASS] = mass
        self.data[idx, StateIdx.PX:StateIdx.PZ + 1] = position[:3]
        self.data[idx, StateIdx.VX:StateIdx.VZ + 1] = velocity[:3]
        self.data[idx, StateIdx.RADIUS] = radius
        self.data[idx, StateIdx.ENTITY_TYPE] = float(entity_type)
        self.data[idx, StateIdx.CD] = cd
        self.data[idx, StateIdx.AREA] = area

        if quaternion is not None:
            self.data[idx, StateIdx.QW:StateIdx.QZ + 1] = quaternion[:4]
        else:
            self.data[idx, StateIdx.QW] = 1.0  # Identity quaternion

        self.data[idx, StateIdx.ACTIVE] = 1.0
        self.data[idx, StateIdx.FUEL_MASS] = fuel_mass
        self.data[idx, StateIdx.SURFACE_FRICTION] = 0.65
        return idx

    def free_entity(self, idx: int) -> None:
        """Deallocates an entity and returns its slot to the free pool."""
        if 0 <= idx < self.max_entities and self.data[idx, StateIdx.ACTIVE] > 0.5:
            self.data[idx, :] = 0.0
            if idx in self._active_indices:
                self._active_indices.remove(idx)
            self._free_indices.append(idx)

    def clear(self) -> None:
        """Deallocates all active entities and resets the free pool."""
        self.data.fill(0.0)
        self._free_indices = list(reversed(range(self.max_entities)))
        self._active_indices = []

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def get_active_mask(self) -> np.ndarray:
        """Returns boolean mask of active entity rows."""
        return self.data[:, StateIdx.ACTIVE] > 0.5

    def get_positions(self) -> np.ndarray:
        """Returns (N, 3) active position slice."""
        mask = self.get_active_mask()
        return self.data[mask, StateIdx.PX:StateIdx.PZ + 1]

    def get_velocities(self) -> np.ndarray:
        """Returns (N, 3) active velocity slice."""
        mask = self.get_active_mask()
        return self.data[mask, StateIdx.VX:StateIdx.VZ + 1]

    def compute_total_mass(self) -> float:
        """Vectorized sum of total system mass."""
        mask = self.get_active_mask()
        return float(np.sum(self.data[mask, StateIdx.MASS]))

    def compute_linear_momentum(self) -> np.ndarray:
        """Vectorized sum of total system linear momentum: sum(m_i * v_i)."""
        mask = self.get_active_mask()
        masses = self.data[mask, StateIdx.MASS, np.newaxis]
        vels = self.data[mask, StateIdx.VX:StateIdx.VZ + 1]
        return np.sum(masses * vels, axis=0)

    def compute_kinetic_energy(self) -> float:
        """Vectorized total kinetic energy: 0.5 * sum(m_i * v_i^2)."""
        mask = self.get_active_mask()
        masses = self.data[mask, StateIdx.MASS]
        vels = self.data[mask, StateIdx.VX:StateIdx.VZ + 1]
        v_sq = np.sum(vels**2, axis=1)
        return float(0.5 * np.sum(masses * v_sq))
