"""
PyAero3D - Keyboard Flight Yoke & Vehicle Control Input Handler.
"""

from typing import Dict, Any, Callable
from direct.showbase.ShowBase import ShowBase
from panda3d.core import ModifierButtons

from pyaero3d.core.types import StateIdx, EntityType
from pyaero3d.core.state import StateBuffer


class FlightYokeController:
    """
    Translates keyboard inputs into continuous aerodynamic control commands.
    """

    def __init__(self, showbase: ShowBase, state_buffer: StateBuffer):
        self.base = showbase
        self.buffer = state_buffer
        self.controlled_entity_idx = -1

        # Key state map
        self.keys: Dict[str, bool] = {
            "w": False, "s": False, "a": False, "d": False,
            "q": False, "e": False, "shift": False, "control": False,
            "space": False,
        }

        # Callback slots
        self.on_spawn_jet: Optional[Callable[[], None]] = None
        self.on_spawn_drone: Optional[Callable[[], None]] = None
        self.on_spawn_cargo: Optional[Callable[[], None]] = None
        self.on_spawn_rocket: Optional[Callable[[], None]] = None
        self.on_cycle_cam: Optional[Callable[[], None]] = None
        self.on_trigger_breakup: Optional[Callable[[], None]] = None
        self.on_toggle_units: Optional[Callable[[], None]] = None
        self.on_toggle_help: Optional[Callable[[], None]] = None
        self.on_reset_world: Optional[Callable[[], None]] = None

        self._bind_keys()

    def set_target_entity(self, idx: int) -> None:
        """Assigns the currently controlled entity index."""
        self.controlled_entity_idx = idx

    def _set_key(self, key: str, value: bool) -> None:
        self.keys[key] = value

    def _bind_keys(self) -> None:
        """Binds Panda3D keyboard event listeners."""
        # Continuous flight control inputs
        for k in ["w", "s", "a", "d", "q", "e", "space"]:
            self.base.accept(k, self._set_key, [k, True])
            self.base.accept(f"{k}-up", self._set_key, [k, False])

        self.base.accept("shift", self._set_key, ["shift", True])
        self.base.accept("shift-up", self._set_key, ["shift", False])
        self.base.accept("control", self._set_key, ["control", True])
        self.base.accept("control-up", self._set_key, ["control", False])

        # One-shot trigger keys
        self.base.accept("tab", lambda: self.on_cycle_cam() if self.on_cycle_cam else None)
        self.base.accept("1", lambda: self.on_spawn_jet() if self.on_spawn_jet else None)
        self.base.accept("2", lambda: self.on_spawn_drone() if self.on_spawn_drone else None)
        self.base.accept("3", lambda: self.on_spawn_cargo() if self.on_spawn_cargo else None)
        self.base.accept("4", lambda: self.on_spawn_rocket() if self.on_spawn_rocket else None)
        self.base.accept("f", lambda: self.on_trigger_breakup() if self.on_trigger_breakup else None)
        self.base.accept("u", lambda: self.on_toggle_units() if self.on_toggle_units else None)
        self.base.accept("h", lambda: self.on_toggle_help() if self.on_toggle_help else None)
        self.base.accept("f1", lambda: self.on_reset_world() if self.on_reset_world else None)

    def update(self, dt: float) -> None:
        """Applies current yoke inputs to target controlled entity."""
        idx = self.controlled_entity_idx
        if idx < 0 or idx >= self.buffer.max_entities:
            return
        if self.buffer.data[idx, StateIdx.ACTIVE] < 0.5:
            return

        row = self.buffer.data[idx]

        # 1. Pitch Elevator Control (W = Nose Down, S = Nose Up)
        target_elev = 0.0
        if self.keys["s"]: target_elev += 1.0
        if self.keys["w"]: target_elev -= 1.0
        # Smooth lerp
        row[StateIdx.CTRL_ELEVATOR] += (target_elev - row[StateIdx.CTRL_ELEVATOR]) * min(1.0, 10.0 * dt)

        # 2. Roll Aileron Control (A = Roll Left, D = Roll Right)
        target_ail = 0.0
        if self.keys["d"]: target_ail += 1.0
        if self.keys["a"]: target_ail -= 1.0
        row[StateIdx.CTRL_AILERON] += (target_ail - row[StateIdx.CTRL_AILERON]) * min(1.0, 10.0 * dt)

        # 3. Yaw Rudder Control (Q = Yaw Left, E = Yaw Right)
        target_rud = 0.0
        if self.keys["e"]: target_rud += 1.0
        if self.keys["q"]: target_rud -= 1.0
        row[StateIdx.CTRL_RUDDER] += (target_rud - row[StateIdx.CTRL_RUDDER]) * min(1.0, 8.0 * dt)

        # 4. Throttle Control (Shift = Throttle Up, Ctrl = Throttle Down)
        thr_rate = 0.40 # 40% per second
        if self.keys["shift"]:
            row[StateIdx.THROTTLE] = min(1.0, row[StateIdx.THROTTLE] + thr_rate * dt)
        if self.keys["control"]:
            row[StateIdx.THROTTLE] = max(0.0, row[StateIdx.THROTTLE] - thr_rate * dt)

        # 5. Wheel Brakes
        if self.keys["space"]:
            row[StateIdx.SURFACE_FRICTION] = 0.85
        else:
            row[StateIdx.SURFACE_FRICTION] = 0.03
