"""
PyAero3D - Comprehensive Physical Graph Studio & Interactive Aerospace Engineering Workspace.
Features 8 Multi-Domain Physical Scenarios, Airfoil Cp Pressure Analyzers, Orbital Hohmann Transfers,
Chaotic Lagrangian Pendulums, Lorentz Electromagnetic Gyromotion, Multi-Run Curve Overlays,
Speedometer Telemetry, and Complete Mathematical Formulations.
"""

import sys
import os
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QSlider, QComboBox,
    QDoubleSpinBox, QCheckBox, QTabWidget, QSplitter, QFileDialog,
    QMessageBox, QDialog, QTextEdit, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from pyaero3d.physics.atmosphere import StandardAtmosphere
from pyaero3d.physics.earth_ballistics import EarthGravityModel, EarthAirDragModel
from pyaero3d.physics.advanced_solvers import (
    NACA4AirfoilSolver, OrbitalMechanicsSolver,
    ChaoticDoublePendulumSolver, LorentzParticleSolver
)
from pyaero3d.core.types import STANDARD_GRAVITY, EARTH_RADIUS, EARTH_MASS, G_GRAVITATIONAL


class PhysicsHelpDialog(QDialog):
    """Interactive User Guide & Physics Formulation Reference."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PyAero3D - Physics Formulations & Engineering Guide")
        self.resize(800, 620)
        self.setStyleSheet("""
            QDialog { background-color: #0D1117; color: #E6EDF3; font-family: 'Segoe UI', sans-serif; }
            QTextEdit { background-color: #161B22; color: #E6EDF3; border: 1px solid #30363D; border-radius: 6px; font-size: 13px; line-height: 1.5; }
            QPushButton { background-color: #238636; color: white; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #2EA043; }
        """)
        layout = QVBoxLayout(self)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setHtml("""
        <h2 style='color: #58A6FF;'>PyAero3D Physical Graph Studio — Mathematical Formulations</h2>
        <p>A professional computational laboratory spanning aerospace flight dynamics, classical analytical mechanics, orbital mechanics, and fluid physics.</p>

        <h3 style='color: #3FB950;'>1. Earth Compressible Ballistics & Drag</h3>
        <p><b>Equation of Motion:</b> m d<b>v</b>/dt = m <b>g</b>(y) - 0.5 &rho;(y) C<sub>D</sub>(M) A ||<b>v</b>|| <b>v</b></p>
        <ul>
            <li><b>Altitude-Decay Gravity:</b> g(y) = g<sub>0</sub> (R<sub>&oplus;</sub> / (R<sub>&oplus;</sub> + y))<sup>2</sup></li>
            <li><b>Transonic Shock Wave Drag:</b> C<sub>D</sub> spikes around Mach 1.0 due to shockwave formation.</li>
        </ul>

        <h3 style='color: #58A6FF;'>2. NACA Airfoil Theory & Pressure Distribution C<sub>p</sub>(x/c)</h3>
        <p><b>Bernoulli Incompressible Pressure:</b> C<sub>p</sub>(x) = 1 - (V(x) / V<sub>&infin;</sub>)<sup>2</sup></p>
        <ul>
            <li>Computes suction peak on upper surface and lift coefficient C<sub>L</sub> = &oint; (C<sub>p,lower</sub> - C<sub>p,upper</sub>) d(x/c).</li>
        </ul>

        <h3 style='color: #E3B341;'>3. Keplerian Orbital Mechanics & Hohmann Transfers</h3>
        <ul>
            <li><b>Vis-Viva Equation:</b> v<sup>2</sup> = &mu; (2/r - 1/a)</li>
            <li><b>Delta-V Burns:</b> &Delta;v<sub>1</sub> = v<sub>transfer,peri</sub> - v<sub>circ,1</sub>, &Delta;v<sub>2</sub> = v<sub>circ,2</sub> - v<sub>transfer,apo</sub></li>
        </ul>

        <h3 style='color: #F85149;'>4. Chaotic Double Pendulum (Lagrange Mechanics)</h3>
        <p>Nonlinear coupled Euler-Lagrange equations exhibiting deterministic chaos and sensitive dependence on initial conditions (Lyapunov exponent &lambda; > 0).</p>

        <h3 style='color: #BC8CFF;'>5. Electromagnetic Lorentz Force (Boris Integrator)</h3>
        <p><b>Lorentz Equation:</b> <b>F</b> = q (<b>E</b> + <b>v</b> &times; <b>B</b>)</p>
        <p>Energy-conserving Boris algorithm preserves magnetic gyroradius r<sub>L</sub> = m v<sub>&perp;</sub> / (|q| B).</p>
        """)
        layout.addWidget(txt)

        btn_close = QPushButton("Close Reference")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)


class PyAero3DGraphStudio(QMainWindow):
    """
    Comprehensive Physical Simulation & Multi-Domain Graph Studio.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyAero3D // Universal Physical Graph Studio & Aerospace Lab")
        self.resize(1420, 880)
        self.setMinimumSize(1150, 720)

        # Physical Simulation State
        self.sim_time = 0.0
        self.dt = 0.02
        self.time_scale = 1.0
        self.is_running = False
        self.use_metric = True

        # Trajectory Buffers
        self.trajectory_x: List[float] = []
        self.trajectory_y: List[float] = []
        self.trajectory_t: List[float] = []
        self.trajectory_v: List[float] = []
        self.saved_runs: List[Dict[str, Any]] = []

        # Current State
        self.pos_x = 0.0
        self.pos_y = 1.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.current_mass = 15.0
        self.has_landed = False

        # Double Pendulum State: [th1, w1, th2, w2]
        self.pendulum_state = np.array([np.pi / 2.0, 0.0, np.pi / 2.0, 0.0])
        self.pendulum_solver = ChaoticDoublePendulumSolver()
        self.pendulum_trail_x: List[float] = []
        self.pendulum_trail_y: List[float] = []

        # Lorentz Particle State
        self.lorentz_pos = np.array([0.0, 0.0, 0.0])
        self.lorentz_vel = np.array([50.0, 20.0, 0.0])
        self.lorentz_trail_x: List[float] = []
        self.lorentz_trail_y: List[float] = []

        # Analytical Solvers
        self.atmosphere = StandardAtmosphere()
        self.gravity_model = EarthGravityModel()
        self.drag_model = EarthAirDragModel()

        # UI Initialization
        self._setup_dark_theme()
        self._init_ui()

        # 50Hz Real-Time Simulation Timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._simulation_tick)
        self.sim_timer.start(20)

        # Initial Reset
        self.reset_simulation()

    def _setup_dark_theme(self):
        """Applies scientific dark aesthetic."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0D1117; color: #C9D1D9; font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; }
            QGroupBox { border: 1px solid #30363D; border-radius: 8px; margin-top: 10px; font-weight: bold; color: #58A6FF; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background-color: #21262D; border: 1px solid #30363D; border-radius: 6px; padding: 6px 12px; font-weight: bold; color: #E6EDF3; }
            QPushButton:hover { background-color: #30363D; border-color: #8B949E; }
            QPushButton:pressed { background-color: #161B22; }
            QPushButton#btn_play { background-color: #238636; border-color: #2EA043; }
            QPushButton#btn_play:hover { background-color: #2EA043; }
            QPushButton#btn_pause { background-color: #D29922; border-color: #BB8009; color: #0D1117; }
            QPushButton#btn_reset { background-color: #DA3633; border-color: #F85149; }
            QSlider::groove:horizontal { border: 1px solid #30363D; height: 6px; background: #161B22; border-radius: 3px; }
            QSlider::handle:horizontal { background: #58A6FF; border: 1px solid #1F6FEB; width: 14px; margin: -5px 0; border-radius: 7px; }
            QDoubleSpinBox, QComboBox { background-color: #161B22; border: 1px solid #30363D; border-radius: 4px; padding: 3px; color: #E6EDF3; }
            QComboBox QAbstractItemView { background-color: #161B22; selection-background-color: #1F6FEB; color: #E6EDF3; }
            QTabWidget::pane { border: 1px solid #30363D; border-radius: 6px; }
            QTabBar::tab { background: #161B22; border: 1px solid #30363D; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #21262D; border-bottom: 2px solid #58A6FF; color: #58A6FF; }
        """)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left Column: Tabbed Multi-Chart Viewport
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.tabs = QTabWidget()

        # Tab 1: XY Trajectory
        self.fig_xy = Figure(facecolor="#0D1117")
        self.canvas_xy = FigureCanvas(self.fig_xy)
        self.ax_xy = self.fig_xy.add_subplot(111)
        self.tabs.addTab(self.canvas_xy, "1. XY Physical Trajectory")

        # Tab 2: Velocity & Energy History
        self.fig_hist = Figure(facecolor="#0D1117")
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.ax_hist_v = self.fig_hist.add_subplot(211)
        self.ax_hist_e = self.fig_hist.add_subplot(212)
        self.tabs.addTab(self.canvas_hist, "2. Velocity & Energy Curves")

        # Tab 3: Airfoil Geometry & Cp Pressure Distribution
        self.fig_foil = Figure(facecolor="#0D1117")
        self.canvas_foil = FigureCanvas(self.fig_foil)
        self.ax_foil_shape = self.fig_foil.add_subplot(211)
        self.ax_foil_cp = self.fig_foil.add_subplot(212)
        self.tabs.addTab(self.canvas_foil, "3. Airfoil & Cp Distribution")

        # Tab 4: Orbital Map & Hohmann Transfer
        self.fig_orbit = Figure(facecolor="#0D1117")
        self.canvas_orbit = FigureCanvas(self.fig_orbit)
        self.ax_orbit = self.fig_orbit.add_subplot(111)
        self.tabs.addTab(self.canvas_orbit, "4. Orbital Hohmann Transfer")

        left_layout.addWidget(self.tabs, stretch=1)

        # Transport & Control Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 4px;")
        t_layout = QHBoxLayout(toolbar)
        t_layout.setContentsMargins(6, 4, 6, 4)

        self.btn_play = QPushButton("Launch / Resume")
        self.btn_play.setObjectName("btn_play")
        self.btn_play.clicked.connect(self.play_simulation)
        t_layout.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.clicked.connect(self.pause_simulation)
        t_layout.addWidget(self.btn_pause)

        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self.step_once)
        t_layout.addWidget(self.btn_step)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self.reset_simulation)
        t_layout.addWidget(self.btn_reset)

        t_layout.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 50)
        self.slider_speed.setValue(10)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        t_layout.addWidget(self.slider_speed)
        self.lbl_speed_val = QLabel("1.0x")
        t_layout.addWidget(self.lbl_speed_val)

        self.btn_units = QPushButton("Units: Metric")
        self.btn_units.clicked.connect(self.toggle_units)
        t_layout.addWidget(self.btn_units)

        self.btn_save_run = QPushButton("+ Save Run Overlay")
        self.btn_save_run.clicked.connect(self.save_current_run_overlay)
        t_layout.addWidget(self.btn_save_run)

        self.btn_launch_3d = QPushButton("Launch 3D Simulator")
        self.btn_launch_3d.setStyleSheet("background-color: #1F6FEB; color: white; font-weight: bold; border-color: #388BFD;")
        self.btn_launch_3d.clicked.connect(self.launch_3d_sandbox)
        t_layout.addWidget(self.btn_launch_3d)

        self.btn_help = QPushButton("Formulations & Guide")
        self.btn_help.clicked.connect(self.show_help_dialog)
        t_layout.addWidget(self.btn_help)

        left_layout.addWidget(toolbar)
        main_layout.addWidget(left_panel, stretch=7)

        # Right Column: Controls, Telemetry & Scenario Parameters
        right_panel = QWidget()
        right_panel.setFixedWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 1. Scenario Selector
        grp_scenario = QGroupBox("Physical Scenario Mode")
        scen_layout = QVBoxLayout(grp_scenario)
        self.cmb_scenario = QComboBox()
        self.cmb_scenario.addItems([
            "1. Earth Compressible Ballistics & Drag",
            "2. Fighter Jet 6-DOF Flight Envelope",
            "3. Interactive NACA Airfoil & Cp Polar",
            "4. Multi-Stage Space Rocket Gravity Turn",
            "5. Orbital Mechanics & Hohmann Transfer",
            "6. Chaotic Double Pendulum (Lagrange)",
            "7. Lorentz Force & Particle Cyclotron",
            "8. Viscoelastic Spring-Damper Contact"
        ])
        self.cmb_scenario.currentIndexChanged.connect(self._on_scenario_changed)
        scen_layout.addWidget(self.cmb_scenario)
        right_layout.addWidget(grp_scenario)

        # 2. Digital Speedometer & Live Telemetry
        grp_telemetry = QGroupBox("Live Flight & Engineering Telemetry")
        tel_layout = QGridLayout(grp_telemetry)
        tel_layout.setVerticalSpacing(4)

        self.lbl_spd = QLabel("0.0 km/h")
        self.lbl_spd.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        self.lbl_spd.setStyleSheet("color: #3FB950;")
        tel_layout.addWidget(QLabel("Airspeed / Velocity:"), 0, 0)
        tel_layout.addWidget(self.lbl_spd, 0, 1)

        self.lbl_mach = QLabel("M 0.00")
        self.lbl_mach.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.lbl_mach.setStyleSheet("color: #58A6FF;")
        tel_layout.addWidget(QLabel("Mach Number:"), 1, 0)
        tel_layout.addWidget(self.lbl_mach, 1, 1)

        self.lbl_alt = QLabel("0.0 m")
        self.lbl_alt.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        self.lbl_alt.setStyleSheet("color: #58A6FF;")
        tel_layout.addWidget(QLabel("Altitude / Y-Coord:"), 2, 0)
        tel_layout.addWidget(self.lbl_alt, 2, 1)

        self.lbl_range = QLabel("0.0 m")
        self.lbl_range.setFont(QFont("Consolas", 12))
        tel_layout.addWidget(QLabel("Downrange / X-Coord:"), 3, 0)
        tel_layout.addWidget(self.lbl_range, 3, 1)

        self.lbl_q = QLabel("0 Pa")
        self.lbl_q.setFont(QFont("Consolas", 11))
        tel_layout.addWidget(QLabel("Dynamic Pressure (q):"), 4, 0)
        tel_layout.addWidget(self.lbl_q, 4, 1)

        self.lbl_energy = QLabel("0.0 kJ")
        self.lbl_energy.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self.lbl_energy.setStyleSheet("color: #E3B341;")
        tel_layout.addWidget(QLabel("Total Mechanical Energy:"), 5, 0)
        tel_layout.addWidget(self.lbl_energy, 5, 1)

        right_layout.addWidget(grp_telemetry)

        # 3. Interactive Physics Sliders
        grp_params = QGroupBox("Physical Parameters & Knobs")
        param_layout = QGridLayout(grp_params)
        param_layout.setVerticalSpacing(4)

        # Launch Speed
        param_layout.addWidget(QLabel("Launch Speed v₀ (m/s):"), 0, 0)
        self.spin_v0 = QDoubleSpinBox()
        self.spin_v0.setRange(0.0, 10000.0)
        self.spin_v0.setValue(320.0)
        self.spin_v0.setSingleStep(10.0)
        param_layout.addWidget(self.spin_v0, 0, 1)

        # Launch Angle / Airfoil AoA
        param_layout.addWidget(QLabel("Angle θ / AoA α (°):"), 1, 0)
        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setRange(-90.0, 90.0)
        self.spin_theta.setValue(45.0)
        param_layout.addWidget(self.spin_theta, 1, 1)

        # Mass
        param_layout.addWidget(QLabel("Mass m (kg):"), 2, 0)
        self.spin_mass = QDoubleSpinBox()
        self.spin_mass.setRange(0.001, 1000000.0)
        self.spin_mass.setValue(15.0)
        param_layout.addWidget(self.spin_mass, 2, 1)

        # Drag Coefficient / Camber
        param_layout.addWidget(QLabel("Drag Coeff Cd / Camber:"), 3, 0)
        self.spin_cd = QDoubleSpinBox()
        self.spin_cd.setRange(0.0, 5.0)
        self.spin_cd.setValue(0.30)
        self.spin_cd.setSingleStep(0.02)
        param_layout.addWidget(self.spin_cd, 3, 1)

        # Area / Thickness
        param_layout.addWidget(QLabel("Area A (m²) / Thickness:"), 4, 0)
        self.spin_area = QDoubleSpinBox()
        self.spin_area.setRange(0.001, 100.0)
        self.spin_area.setValue(0.08)
        param_layout.addWidget(self.spin_area, 4, 1)

        # Crosswind / Magnetic Field
        param_layout.addWidget(QLabel("Wind / Magnetic B (T):"), 5, 0)
        self.spin_wind = QDoubleSpinBox()
        self.spin_wind.setRange(-200.0, 200.0)
        self.spin_wind.setValue(0.0)
        param_layout.addWidget(self.spin_wind, 5, 1)

        # Continuous Thrust / Electric E
        param_layout.addWidget(QLabel("Thrust (N) / Electric E:"), 6, 0)
        self.spin_thrust = QDoubleSpinBox()
        self.spin_thrust.setRange(0.0, 5000000.0)
        self.spin_thrust.setValue(0.0)
        self.spin_thrust.setSingleStep(500.0)
        param_layout.addWidget(self.spin_thrust, 6, 1)

        right_layout.addWidget(grp_params)

        # 4. Data Export & Actions
        grp_export = QGroupBox("Data Export & Run Management")
        exp_layout = QHBoxLayout(grp_export)
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.export_csv)
        exp_layout.addWidget(self.btn_export)
        self.btn_clear = QPushButton("Clear Overlays")
        self.btn_clear.clicked.connect(self.clear_overlays)
        right_layout.addStretch(1)
        main_layout.addWidget(right_panel, stretch=3)

        # Connect tab signal after entire UI is fully instantiated
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_scenario_changed(self, idx: int):
        """Switches physical models, defaults, and activates relevant tabs."""
        if idx == 0:  # Ballistics
            self.tabs.setCurrentIndex(0)
            self.spin_v0.setValue(320.0)
            self.spin_theta.setValue(45.0)
            self.spin_mass.setValue(15.0)
            self.spin_cd.setValue(0.30)
            self.spin_area.setValue(0.08)
            self.spin_thrust.setValue(0.0)
        elif idx == 1:  # Fighter Jet
            self.tabs.setCurrentIndex(0)
            self.spin_v0.setValue(180.0)
            self.spin_theta.setValue(15.0)
            self.spin_mass.setValue(12000.0)
            self.spin_cd.setValue(0.028)
            self.spin_area.setValue(28.0)
            self.spin_thrust.setValue(65000.0)
        elif idx == 2:  # NACA Airfoil
            self.tabs.setCurrentIndex(2)
            self.spin_theta.setValue(5.0)  # AoA = 5 deg
            self.spin_cd.setValue(0.02)    # 2% Camber (NACA 2412)
            self.spin_area.setValue(0.12)  # 12% Thickness (NACA 2412)
        elif idx == 3:  # Multi-Stage Rocket
            self.tabs.setCurrentIndex(0)
            self.spin_v0.setValue(10.0)
            self.spin_theta.setValue(88.0)
            self.spin_mass.setValue(8500.0)
            self.spin_cd.setValue(0.20)
            self.spin_area.setValue(4.5)
            self.spin_thrust.setValue(140000.0)
        elif idx == 4:  # Orbital Hohmann Transfer
            self.tabs.setCurrentIndex(3)
            self.spin_v0.setValue(7700.0)  # LEO orbital speed
            self.spin_theta.setValue(0.0)
        elif idx == 5:  # Chaotic Double Pendulum
            self.tabs.setCurrentIndex(0)
            self.pendulum_state = np.array([np.pi / 2.0, 0.0, np.pi / 2.0, 0.0])
            self.pendulum_trail_x.clear()
            self.pendulum_trail_y.clear()
        elif idx == 6:  # Lorentz Particle Cyclotron
            self.tabs.setCurrentIndex(0)
            self.lorentz_pos = np.array([0.0, 0.0, 0.0])
            self.lorentz_vel = np.array([50.0, 20.0, 0.0])
            self.lorentz_trail_x.clear()
            self.lorentz_trail_y.clear()
            self.spin_wind.setValue(1.5)    # 1.5 Tesla Magnetic Field
            self.spin_thrust.setValue(0.0)  # Zero Electric Field
        elif idx == 7:  # Spring-Damper
            self.tabs.setCurrentIndex(0)
            self.spin_v0.setValue(40.0)
            self.spin_theta.setValue(60.0)
            self.spin_mass.setValue(5.0)

        self.reset_simulation()

    def _on_tab_changed(self, idx: int):
        self._redraw_charts()

    def _on_speed_changed(self, val: int):
        self.time_scale = val / 10.0
        self.lbl_speed_val.setText(f"{self.time_scale:.1f}x")

    def toggle_units(self):
        self.use_metric = not self.use_metric
        self.btn_units.setText("Units: Metric (m, km/h)" if self.use_metric else "Units: Imperial (ft, kt)")
        self._redraw_charts()

    def show_help_dialog(self):
        dlg = PhysicsHelpDialog(self)
        dlg.exec()

    def launch_3d_sandbox(self):
        """Launches the full 3D Panda3D mountain flight simulator sandbox in a separate process."""
        import subprocess
        main_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main.py"))
        try:
            subprocess.Popen([sys.executable, main_py, "--3d"])
        except Exception as e:
            QMessageBox.warning(self, "3D Simulator Launch Error", f"Failed to launch 3D simulator:\n{e}")

    def play_simulation(self):
        self.is_running = True
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def pause_simulation(self):
        self.is_running = False
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def step_once(self):
        self.pause_simulation()
        self._integrate_step(self.dt)
        self._redraw_charts()

    def save_current_run_overlay(self):
        """Saves current trajectory run for side-by-side comparison overlay."""
        if len(self.trajectory_x) < 2:
            return
        run_data = {
            "x": list(self.trajectory_x),
            "y": list(self.trajectory_y),
            "label": f"Run #{len(self.saved_runs) + 1} (v0={self.spin_v0.value():.0f}, Cd={self.spin_cd.value():.2f})",
            "color": ["#BC8CFF", "#F0883E", "#E3B341", "#3FB950", "#58A6FF"][len(self.saved_runs) % 5]
        }
        self.saved_runs.append(run_data)
        self._redraw_charts()

    def clear_overlays(self):
        self.saved_runs.clear()
        self.trajectory_x.clear()
        self.trajectory_y.clear()
        self.trajectory_t.clear()
        self.trajectory_v.clear()
        self.pendulum_trail_x.clear()
        self.pendulum_trail_y.clear()
        self.lorentz_trail_x.clear()
        self.lorentz_trail_y.clear()
        self.reset_simulation()

    def reset_simulation(self):
        self.is_running = False
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

        self.sim_time = 0.0
        self.pos_x = 0.0
        self.pos_y = 1.0
        
        v0 = self.spin_v0.value()
        theta_rad = np.radians(self.spin_theta.value())
        self.vel_x = v0 * np.cos(theta_rad)
        self.vel_y = v0 * np.sin(theta_rad)
        self.current_mass = self.spin_mass.value()
        self.has_landed = False

        self.trajectory_x = [self.pos_x]
        self.trajectory_y = [self.pos_y]
        self.trajectory_t = [self.sim_time]
        self.trajectory_v = [v0]

        # Reset Pendulum
        self.pendulum_state = np.array([np.pi / 2.0, 0.0, np.pi / 2.0, 0.0])
        self.pendulum_trail_x.clear()
        self.pendulum_trail_y.clear()

        # Reset Lorentz
        self.lorentz_pos = np.array([0.0, 0.0, 0.0])
        self.lorentz_vel = np.array([v0 * np.cos(theta_rad), v0 * np.sin(theta_rad), 0.0])
        self.lorentz_trail_x.clear()
        self.lorentz_trail_y.clear()

        self._redraw_charts()

    def _simulation_tick(self):
        if not self.is_running:
            return

        steps = int(np.clip(self.time_scale * 2, 1, 10))
        sub_dt = (self.dt * self.time_scale) / steps

        for _ in range(steps):
            if not self.has_landed:
                self._integrate_step(sub_dt)

        self._redraw_charts()

    def _integrate_step(self, dt: float):
        scen_idx = self.cmb_scenario.currentIndex()

        if scen_idx == 5:  # Double Pendulum
            self.pendulum_state = self.pendulum_solver.rk4_step(self.pendulum_state, dt)
            x1, y1, x2, y2 = self.pendulum_solver.get_cartesian_positions(self.pendulum_state)
            self.pos_x, self.pos_y = x2, y2
            self.vel_x, self.vel_y = self.pendulum_state[1], self.pendulum_state[3]
            self.pendulum_trail_x.append(x2)
            self.pendulum_trail_y.append(y2)
            if len(self.pendulum_trail_x) > 500:
                self.pendulum_trail_x.pop(0)
                self.pendulum_trail_y.pop(0)
            self.sim_time += dt
            self.lbl_spd.setText(f"{abs(self.pendulum_state[1]):.2f} rad/s")
            self.lbl_alt.setText(f"{y2:.2f} m")
            self.lbl_range.setText(f"{x2:.2f} m")
            return

        elif scen_idx == 6:  # Lorentz Force Particle
            b_field = np.array([0.0, 0.0, self.spin_wind.value()])
            e_field = np.array([0.0, self.spin_thrust.value() * 0.001, 0.0])
            self.lorentz_pos, self.lorentz_vel = LorentzParticleSolver.step_boris(
                self.lorentz_pos, self.lorentz_vel, q=1.0, m=self.current_mass * 0.01,
                E=e_field, B=b_field, dt=dt
            )
            self.pos_x, self.pos_y = self.lorentz_pos[0], self.lorentz_pos[1]
            self.vel_x, self.vel_y = self.lorentz_vel[0], self.lorentz_vel[1]
            self.lorentz_trail_x.append(self.pos_x)
            self.lorentz_trail_y.append(self.pos_y)
            if len(self.lorentz_trail_x) > 600:
                self.lorentz_trail_x.pop(0)
                self.lorentz_trail_y.pop(0)
            self.sim_time += dt
            spd = float(np.hypot(self.vel_x, self.vel_y))
            self.lbl_spd.setText(f"{spd:.1f} m/s")
            self.lbl_alt.setText(f"{self.pos_y:.2f} m")
            self.lbl_range.setText(f"{self.pos_x:.2f} m")
            return

        # Atmospheric & Ballistic Integration (Scenarios 0, 1, 3, 7)
        T, P, rho, a, mu = StandardAtmosphere.get_properties(self.pos_y)
        g_y = self.gravity_model.get_gravity(self.pos_y)

        wind_x = self.spin_wind.value()
        v_rel_x = self.vel_x - wind_x
        v_rel_y = self.vel_y
        speed_rel = float(np.hypot(v_rel_x, v_rel_y))
        mach = speed_rel / max(100.0, a)

        cd_val = self.spin_cd.value()
        if 0.75 < mach < 1.3:
            cd_val *= (1.0 + 1.5 * np.exp(-((mach - 1.0) / 0.15) ** 2))
        elif mach >= 1.3:
            cd_val *= (1.0 + 0.4 / np.sqrt(mach ** 2 - 1.0))

        area = self.spin_area.value()
        f_drag_mag = 0.5 * rho * (speed_rel ** 2) * cd_val * area
        f_drag_x = -f_drag_mag * (v_rel_x / speed_rel) if speed_rel > 1e-4 else 0.0
        f_drag_y = -f_drag_mag * (v_rel_y / speed_rel) if speed_rel > 1e-4 else 0.0

        thrust_mag = self.spin_thrust.value()

        if scen_idx == 3:  # Rocket Gravity Turn
            turn_angle = np.radians(max(10.0, 90.0 - (self.pos_y / 1500.0) * 80.0))
            f_thrust_x = thrust_mag * np.cos(turn_angle)
            f_thrust_y = thrust_mag * np.sin(turn_angle)
        elif scen_idx == 1:  # Fighter Jet Lift Polar
            c_l = 0.50
            f_lift_mag = 0.5 * rho * (speed_rel ** 2) * area * c_l
            f_lift_x = -f_lift_mag * (v_rel_y / speed_rel) if speed_rel > 1e-4 else 0.0
            f_lift_y = f_lift_mag * (v_rel_x / speed_rel) if speed_rel > 1e-4 else 0.0
            f_thrust_x = thrust_mag * (v_rel_x / speed_rel) if speed_rel > 1e-4 else thrust_mag
            f_thrust_y = thrust_mag * (v_rel_y / speed_rel) if speed_rel > 1e-4 else 0.0
            f_drag_x += f_lift_x
            f_drag_y += f_lift_y
        else:
            f_thrust_x = thrust_mag * (self.vel_x / speed_rel) if speed_rel > 1e-4 else 0.0
            f_thrust_y = thrust_mag * (self.vel_y / speed_rel) if speed_rel > 1e-4 else 0.0

        m = self.current_mass
        ax = (f_drag_x + f_thrust_x) / m
        ay = (f_drag_y + f_thrust_y) / m - g_y

        self.vel_x += ax * dt
        self.vel_y += ay * dt
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt
        self.sim_time += dt

        # Ground Collision & Restitution
        if self.pos_y <= 0.0:
            self.pos_y = 0.0
            if abs(self.vel_y) > 2.0:
                restitution = 0.60 if scen_idx == 7 else 0.25
                self.vel_y = -self.vel_y * restitution
                self.vel_x *= 0.85
            else:
                self.vel_y = 0.0
                self.vel_x *= 0.90
                if abs(self.vel_x) < 0.1:
                    self.has_landed = True
                    self.is_running = False
                    self.btn_play.setEnabled(True)
                    self.btn_pause.setEnabled(False)

        self.trajectory_x.append(self.pos_x)
        self.trajectory_y.append(self.pos_y)
        self.trajectory_t.append(self.sim_time)
        self.trajectory_v.append(float(np.hypot(self.vel_x, self.vel_y)))

        q_pa = 0.5 * rho * (speed_rel ** 2)
        e_tot = (0.5 * m * (speed_rel ** 2) + m * g_y * self.pos_y) / 1000.0

        if self.use_metric:
            self.lbl_spd.setText(f"{speed_rel * 3.6:.1f} km/h")
            self.lbl_alt.setText(f"{self.pos_y:.1f} m")
            self.lbl_range.setText(f"{self.pos_x:.1f} m")
        else:
            self.lbl_spd.setText(f"{speed_rel * 1.94384:.1f} kt")
            self.lbl_alt.setText(f"{self.pos_y * 3.28084:.1f} ft")
            self.lbl_range.setText(f"{self.pos_x * 3.28084:.1f} ft")

        self.lbl_mach.setText(f"M {mach:.2f}")
        self.lbl_q.setText(f"{int(q_pa):,d} Pa")
        self.lbl_energy.setText(f"{e_tot:.1f} kJ")

    def _redraw_charts(self):
        if not hasattr(self, "cmb_scenario") or not hasattr(self, "tabs"):
            return
        curr_tab = self.tabs.currentIndex()

        if curr_tab == 0:  # XY Trajectory
            self._redraw_xy_tab()
        elif curr_tab == 1:  # Time History & Energy
            self._redraw_history_tab()
        elif curr_tab == 2:  # Airfoil Cp Distribution
            self._redraw_airfoil_tab()
        elif curr_tab == 3:  # Orbital Hohmann Map
            self._redraw_orbital_tab()

    def _redraw_xy_tab(self):
        self.ax_xy.clear()
        self.ax_xy.set_facecolor("#161B22")
        self.ax_xy.grid(True, linestyle="--", alpha=0.35, color="#30363D")

        scen_idx = self.cmb_scenario.currentIndex()

        if scen_idx == 5:  # Double Pendulum Drawing
            self.ax_xy.set_title("Chaotic Double Pendulum — Lagrangian Mechanics", color="#58A6FF", fontweight="bold")
            x1, y1, x2, y2 = self.pendulum_solver.get_cartesian_positions(self.pendulum_state)
            self.ax_xy.plot([0, x1, x2], [0, y1, y2], "o-", color="#C9D1D9", lw=3.0, markersize=8)
            if len(self.pendulum_trail_x) > 1:
                self.ax_xy.plot(self.pendulum_trail_x, self.pendulum_trail_y, "-", color="#BC8CFF", lw=1.5, alpha=0.8)
            self.ax_xy.set_xlim(-2.5, 2.5)
            self.ax_xy.set_ylim(-2.5, 2.5)
            self.canvas_xy.draw_idle()
            return

        elif scen_idx == 6:  # Lorentz Particle
            self.ax_xy.set_title("Lorentz Electromagnetic Field — Cyclotron Gyromotion", color="#58A6FF", fontweight="bold")
            if len(self.lorentz_trail_x) > 1:
                self.ax_xy.plot(self.lorentz_trail_x, self.lorentz_trail_y, "-", color="#58A6FF", lw=2.0)
            self.ax_xy.plot(self.pos_x, self.pos_y, "o", color="#3FB950", markersize=8)
            self.ax_xy.set_xlabel("Position X [m]", color="#C9D1D9")
            self.ax_xy.set_ylabel("Position Y [m]", color="#C9D1D9")
            self.ax_xy.tick_params(colors="#8B949E")
            self.canvas_xy.draw_idle()
            return

        # Ground Line
        self.ax_xy.axhline(0.0, color="#238636", linewidth=2.0, label="Ground Surface (y=0)")

        # Saved Run Overlays
        for run in self.saved_runs:
            self.ax_xy.plot(run["x"], run["y"], "--", color=run["color"], alpha=0.7, label=run["label"])

        # Current Active Trajectory
        if len(self.trajectory_x) > 1:
            self.ax_xy.plot(self.trajectory_x, self.trajectory_y, "-", color="#58A6FF", linewidth=2.5, label="Active Run (PyAero3D)")

        self.ax_xy.plot(self.pos_x, self.pos_y, "o", color="#3FB950", markersize=9, zorder=5)

        # Force Vectors
        spd = float(np.hypot(self.vel_x, self.vel_y))
        if spd > 1e-3:
            scale_v = max(10.0, np.max(self.trajectory_x) if self.trajectory_x else 50.0) * 0.08 / max(1.0, spd)
            self.ax_xy.annotate("", xy=(self.pos_x + self.vel_x * scale_v, self.pos_y + self.vel_y * scale_v),
                                xytext=(self.pos_x, self.pos_y), arrowprops=dict(arrowstyle="->", color="#58A6FF", lw=2.2))
            self.ax_xy.annotate("", xy=(self.pos_x - self.vel_x * scale_v * 0.5, self.pos_y - self.vel_y * scale_v * 0.5),
                                xytext=(self.pos_x, self.pos_y), arrowprops=dict(arrowstyle="->", color="#F85149", lw=1.8))

        unit_dist = "m" if self.use_metric else "ft"
        self.ax_xy.set_xlabel(f"Downrange Distance X [{unit_dist}]", color="#C9D1D9", fontsize=11, fontweight="bold")
        self.ax_xy.set_ylabel(f"Altitude Y [{unit_dist}]", color="#C9D1D9", fontsize=11, fontweight="bold")
        self.ax_xy.tick_params(colors="#8B949E")
        self.ax_xy.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=9)

        max_x = max(50.0, np.max(self.trajectory_x) if self.trajectory_x else 50.0)
        max_y = max(20.0, np.max(self.trajectory_y) if self.trajectory_y else 20.0)
        self.ax_xy.set_xlim(-5.0, max_x * 1.15)
        self.ax_xy.set_ylim(-2.0, max_y * 1.25)
        self.canvas_xy.draw_idle()

    def _redraw_history_tab(self):
        self.ax_hist_v.clear()
        self.ax_hist_e.clear()
        self.ax_hist_v.set_facecolor("#161B22")
        self.ax_hist_e.set_facecolor("#161B22")
        self.ax_hist_v.grid(True, linestyle="--", alpha=0.3, color="#30363D")
        self.ax_hist_e.grid(True, linestyle="--", alpha=0.3, color="#30363D")

        if len(self.trajectory_t) > 1:
            unit_spd = "km/h" if self.use_metric else "kt"
            conv_v = 3.6 if self.use_metric else 1.94384
            v_arr = np.array(self.trajectory_v) * conv_v
            t_arr = np.array(self.trajectory_t)

            self.ax_hist_v.plot(t_arr, v_arr, color="#3FB950", lw=2.0)
            self.ax_hist_v.set_ylabel(f"Velocity [{unit_spd}]", color="#3FB950", fontweight="bold")
            self.ax_hist_v.tick_params(colors="#8B949E")

            m = self.current_mass
            e_k = 0.5 * m * (np.array(self.trajectory_v) ** 2) / 1000.0
            e_p = m * 9.81 * np.array(self.trajectory_y) / 1000.0
            self.ax_hist_e.plot(t_arr, e_k, label="Kinetic (Ek)", color="#58A6FF", lw=1.8)
            self.ax_hist_e.plot(t_arr, e_p, label="Potential (Ep)", color="#E3B341", lw=1.8)
            self.ax_hist_e.plot(t_arr, e_k + e_p, "--", label="Total (E_tot)", color="#F85149", lw=1.5)
            self.ax_hist_e.set_xlabel("Time t [s]", color="#C9D1D9", fontweight="bold")
            self.ax_hist_e.set_ylabel("Energy [kJ]", color="#E3B341", fontweight="bold")
            self.ax_hist_e.tick_params(colors="#8B949E")
            self.ax_hist_e.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=8)

        self.canvas_hist.draw_idle()

    def _redraw_airfoil_tab(self):
        """Renders interactive NACA 4-digit airfoil geometry and Cp pressure curves."""
        self.ax_foil_shape.clear()
        self.ax_foil_cp.clear()
        self.ax_foil_shape.set_facecolor("#161B22")
        self.ax_foil_cp.set_facecolor("#161B22")
        self.ax_foil_shape.grid(True, linestyle="--", alpha=0.3, color="#30363D")
        self.ax_foil_cp.grid(True, linestyle="--", alpha=0.3, color="#30363D")

        alpha_val = self.spin_theta.value()
        camber_val = max(0.0, self.spin_cd.value())
        thick_val = max(0.04, self.spin_area.value())

        xu, yu, xl, yl = NACA4AirfoilSolver.generate_airfoil_coordinates(m_camber=camber_val, p_camber_pos=0.4, t_thickness=thick_val)
        self.ax_foil_shape.plot(xu, yu, color="#58A6FF", lw=2.0, label="Upper Surface")
        self.ax_foil_shape.plot(xl, yl, color="#3FB950", lw=2.0, label="Lower Surface")
        self.ax_foil_shape.fill_between(xu, yu, yl, color="#58A6FF", alpha=0.15)
        self.ax_foil_shape.set_title(f"NACA Airfoil Profile (Camber: {camber_val*100:.1f}%, Thick: {thick_val*100:.1f}%, AoA: {alpha_val:.1f}°)", color="#58A6FF", fontweight="bold")
        self.ax_foil_shape.set_xlim(-0.05, 1.05)
        self.ax_foil_shape.set_ylim(-0.35, 0.35)
        self.ax_foil_shape.set_aspect("equal", adjustable="box")
        self.ax_foil_shape.tick_params(colors="#8B949E")
        self.ax_foil_shape.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=8)

        # Cp distribution
        x_cp, cp_u, cp_l = NACA4AirfoilSolver.compute_pressure_distribution(alpha_deg=alpha_val, m_camber=camber_val, t_thickness=thick_val)
        self.ax_foil_cp.plot(x_cp, cp_u, color="#F85149", lw=2.0, label="Cp Upper (Suction)")
        self.ax_foil_cp.plot(x_cp, cp_l, color="#3FB950", lw=2.0, label="Cp Lower (Pressure)")
        self.ax_foil_cp.invert_yaxis()  # Standard aerodynamic convention: negative Cp suction upwards
        self.ax_foil_cp.set_xlabel("Normalized Chord x / c", color="#C9D1D9", fontweight="bold")
        self.ax_foil_cp.set_ylabel("Pressure Coeff Cp", color="#C9D1D9", fontweight="bold")
        self.ax_foil_cp.tick_params(colors="#8B949E")
        self.ax_foil_cp.legend(loc="lower right", facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=8)

        self.canvas_foil.draw_idle()

    def _redraw_orbital_tab(self):
        """Renders 2D Keplerian Earth orbit and Hohmann transfer ellipse."""
        self.ax_orbit.clear()
        self.ax_orbit.set_facecolor("#161B22")
        self.ax_orbit.grid(True, linestyle="--", alpha=0.3, color="#30363D")

        res = OrbitalMechanicsSolver.calculate_hohmann_transfer(r1_alt_km=400.0, r2_alt_km=35786.0)

        # Draw Earth
        earth_circle = patches.Circle((0, 0), radius=EARTH_RADIUS / 1000.0, color="#1F6FEB", alpha=0.6, label="Earth (R=6,371 km)")
        self.ax_orbit.add_patch(earth_circle)

        # LEO Orbit
        x_leo, y_leo = OrbitalMechanicsSolver.generate_orbit_points(res["r1_km"], 0.0)
        self.ax_orbit.plot(x_leo, y_leo, "--", color="#3FB950", label=f"LEO 400km (v={res['v1_mps']:.0f} m/s)")

        # GEO Orbit
        x_geo, y_geo = OrbitalMechanicsSolver.generate_orbit_points(res["r2_km"], 0.0)
        self.ax_orbit.plot(x_geo, y_geo, "--", color="#E3B341", label=f"GEO 35,786km (v={res['v2_mps']:.0f} m/s)")

        # Hohmann Transfer Ellipse
        a_trans = (res["r1_km"] + res["r2_km"]) * 0.5
        e_trans = (res["r2_km"] - res["r1_km"]) / (res["r1_km"] + res["r2_km"])
        x_trans, y_trans = OrbitalMechanicsSolver.generate_orbit_points(a_trans, e_trans)
        self.ax_orbit.plot(x_trans, y_trans, "-", color="#F85149", lw=2.2, label=f"Hohmann Transfer (Total Δv: {res['dv_total_mps']:.0f} m/s, Duration: {res['transfer_time_hours']:.1f}h)")

        self.ax_orbit.set_title("Keplerian Orbit Map & Hohmann Transfer Orbit", color="#58A6FF", fontweight="bold")
        self.ax_orbit.set_xlabel("X-Axis [km]", color="#C9D1D9", fontweight="bold")
        self.ax_orbit.set_ylabel("Y-Axis [km]", color="#C9D1D9", fontweight="bold")
        self.ax_orbit.tick_params(colors="#8B949E")
        self.ax_orbit.set_aspect("equal", adjustable="box")
        self.ax_orbit.legend(loc="upper right", facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", fontsize=8)

        self.canvas_orbit.draw_idle()

    def export_csv(self):
        if not self.trajectory_x:
            QMessageBox.warning(self, "Export Trajectory", "No trajectory data recorded yet. Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Trajectory CSV", "pyaero3d_trajectory.csv", "CSV Files (*.csv)")
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "pos_x_m", "pos_y_m", "speed_mps"])
                for t, x, y, v in zip(self.trajectory_t, self.trajectory_x, self.trajectory_y, self.trajectory_v):
                    writer.writerow([f"{t:.4f}", f"{x:.4f}", f"{y:.4f}", f"{v:.4f}"])
            QMessageBox.information(self, "Export Success", f"Successfully exported {len(self.trajectory_t)} data rows to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save CSV:\n{e}")


def launch_graph_studio():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PyAero3D Physical Graph Studio")
    studio = PyAero3DGraphStudio()
    studio.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_graph_studio())
