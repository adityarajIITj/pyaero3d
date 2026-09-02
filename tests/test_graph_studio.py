"""
Verification test for PyAero3D Physical Graph Studio logic and scenario switching.
"""

import os
import sys
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def test_graph_studio_physics_logic(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    from pyaero3d.gui.graph_studio import PyAero3DGraphStudio
    # Mock chart redrawing to run headlessly without paint loop
    monkeypatch.setattr(PyAero3DGraphStudio, "_redraw_charts", lambda self: None)

    studio = PyAero3DGraphStudio()
    studio.sim_timer.stop()

    assert studio.pos_x == 0.0
    assert studio.pos_y == 1.0
    assert studio.use_metric is True

    # Step simulation
    studio._integrate_step(0.02)
    assert studio.sim_time == 0.02

    # Test all 8 scenario mode switches
    for scen_idx in range(8):
        studio._on_scenario_changed(scen_idx)
        assert studio.sim_time == 0.0

    studio.close()
