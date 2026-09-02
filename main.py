"""
PyAero3D - Main Application Entry Point.
Launches the Physical XY Graph Studio by default, or the 3D Mountain Sandbox with '--3d'.
"""

import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="PyAero3D - Universal Physics Engine & Physical Graph Studio")
    parser.add_argument("--3d", dest="use_3d", action="store_true", help="Launch 3D Panda3D Mountain Sandbox Viewport")
    args, _ = parser.parse_known_args()

    if args.use_3d:
        from pyaero3d.app_3d import PyAero3DSimulatorApp
        print("=" * 75)
        print("  PyAero3D: Rugged Mountain World Simulation & Multi-Vehicle Sandbox (3D)")
        print("=" * 75)
        app = PyAero3DSimulatorApp()
        app.run()
    else:
        from pyaero3d.gui.graph_studio import launch_graph_studio
        print("=" * 75)
        print("  PyAero3D: Physical Graph Studio & Interactive XY Aerospace Lab")
        print("=" * 75)
        sys.exit(launch_graph_studio())


if __name__ == "__main__":
    main()
