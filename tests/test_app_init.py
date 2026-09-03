"""
Verification test for PyAero3D full desktop engine instantiation.
"""

import os
import sys
import numpy as np
import pytest

from direct.showbase.ShowBase import ShowBase
from panda3d.core import load_prc_file_data

# Configure headless test window
load_prc_file_data("", """
    window-type offscreen
    audio-library-name null
""")

from pyaero3d.app_3d import PyAero3DSimulatorApp
from pyaero3d.core.types import StateIdx, EntityType


def test_pyaero3d_app_startup_and_subsystems():
    app = PyAero3DSimulatorApp()

    # Verify terrain generator
    assert app.terrain_gen.grid_res == 512
    assert app.terrain_gen.is_flat is True
    assert app.terrain_gen.max_height == 0.0

    # Verify physics thread is running
    assert app.physics_thread._running is True
    assert app.state_buffer.active_count >= 1

    # Verify controlled entity is fighter jet
    idx = app.yoke.controlled_entity_idx
    assert idx >= 0
    assert int(app.state_buffer.data[idx, StateIdx.ENTITY_TYPE]) == EntityType.FIXED_WING_JET

    # Test single task frame update
    app.taskMgr.step()

    # Stop physics thread and clean up
    app.physics_thread.stop()
    app.destroy()
    import builtins
    if hasattr(builtins, "base"):
        del builtins.base
