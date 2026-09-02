"""
PyAero3D - Main Application Entry Point.
Launches the Physical XY Graph Studio by default, or the 3D Mountain Sandbox with '--3d'.
"""

import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="PyAero3D - Universal Physics Engine & Physical Graph Studio")
    parser.add_argument("--3d", dest="use_3d", action="store_true", help="Launch 3D Panda3D Mountain Sandbox Viewport")
    parser.add_argument("--scenario", dest="scenario", type=int, default=1, help="Physical Scenario Preset Index (0-7)")
    parser.add_argument("--v0", dest="v0", type=float, default=None, help="Initial Velocity (m/s)")
    parser.add_argument("--theta", dest="theta", type=float, default=None, help="Launch Pitch Angle (deg)")
    parser.add_argument("--mass", dest="mass", type=float, default=None, help="Vehicle Mass (kg)")
    parser.add_argument("--cd", dest="cd", type=float, default=None, help="Drag Coefficient")
    parser.add_argument("--area", dest="area", type=float, default=None, help="Surface Area (m^2)")
    parser.add_argument("--wind", dest="wind", type=float, default=None, help="Crosswind (m/s)")
    parser.add_argument("--thrust", dest="thrust", type=float, default=None, help="Continuous Thrust (N)")
    args, _ = parser.parse_known_args()

    if args.use_3d:
        from pyaero3d.app_3d import PyAero3DSimulatorApp
        print("=" * 75)
        print(f"  PyAero3D: 3D Simulation & Multi-Scenario Sandbox (Preset #{args.scenario})")
        print("=" * 75)
        app = PyAero3DSimulatorApp(
            scenario_idx=args.scenario,
            v0=args.v0,
            theta=args.theta,
            mass=args.mass,
            cd=args.cd,
            area=args.area,
            wind=args.wind,
            thrust=args.thrust,
        )
        app.run()
    else:
        from pyaero3d.gui.graph_studio import launch_graph_studio
        print("=" * 75)
        print("  PyAero3D: Physical Graph Studio & Interactive XY Aerospace Lab")
        print("=" * 75)
        sys.exit(launch_graph_studio())


if __name__ == "__main__":
    main()
