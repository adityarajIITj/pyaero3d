"""
PyAero3D - Movable, Collapsible In-Viewport 3D GUI Control Panel.
Features compact docking, draggable positioning, minimizable state, and full scenario/parameter controls.
"""

from typing import Callable, Optional
from panda3d.core import TextNode, NodePath, Vec4
from direct.gui.DirectGui import (
    DirectFrame, DirectButton, DirectLabel, DGG
)


class InViewportControlPanel3D:
    """
    Interactive 3D Simulation Control Dashboard rendered over Panda3D viewport.
    Supports Dragging, Minimizing, and Docking to avoid blocking the 3D scene.
    """

    def __init__(
        self,
        base_app,
        on_change_scenario: Callable[[int], None],
        on_tweak_mass: Callable[[float], None],
        on_tweak_cd: Callable[[float], None],
        on_tweak_thrust: Callable[[float], None],
        on_tweak_angle: Callable[[float], None],
        on_launch_reset: Callable[[], None],
        on_toggle_pause: Callable[[], None],
        on_step_physics: Callable[[], None],
        on_change_cam_mode: Callable[[int], None],
        on_toggle_axes: Callable[[], None],
        on_toggle_grid: Callable[[], None],
        on_toggle_trail: Callable[[], None],
    ):
        self.base = base_app
        self.on_change_scenario = on_change_scenario
        self.on_tweak_mass = on_tweak_mass
        self.on_tweak_cd = on_tweak_cd
        self.on_tweak_thrust = on_tweak_thrust
        self.on_tweak_angle = on_tweak_angle
        self.on_launch_reset = on_launch_reset
        self.on_toggle_pause = on_toggle_pause
        self.on_step_physics = on_step_physics
        self.on_change_cam_mode = on_change_cam_mode
        self.on_toggle_axes = on_toggle_axes
        self.on_toggle_grid = on_toggle_grid
        self.on_toggle_trail = on_toggle_trail

        self.is_minimized = False
        self.dock_side = "right"  # "right" or "left"

        # Style colors
        btn_col = (0.16, 0.20, 0.28, 0.95)
        btn_hover = (0.24, 0.45, 0.75, 1.0)
        accent_col = (0.15, 0.70, 0.95, 1.0)
        action_col = (0.20, 0.65, 0.35, 1.0)
        txt_fg = (1.0, 1.0, 1.0, 1.0)

        # Main Panel Frame (Compact size: 0.76 wide x 1.70 tall)
        self.frame = DirectFrame(
            frameColor=(0.08, 0.10, 0.14, 0.85),
            frameSize=(-0.38, 0.38, -0.85, 0.85),
            pos=(1.22, 0.0, 0.0),
        )

        # 1. Header Title & Window Controls
        self.lbl_title = DirectLabel(
            parent=self.frame,
            text="PYAERO3D LAB // 3D CONTROLS",
            text_scale=0.032,
            text_fg=accent_col,
            text_shadow=(0, 0, 0, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.06, 0.0, 0.79),
            text_align=TextNode.ACenter,
        )

        # Minimize & Dock Buttons
        self.btn_min = DirectButton(
            parent=self.frame,
            text="[-]",
            text_scale=0.030,
            text_fg=txt_fg,
            frameColor=(0.20, 0.25, 0.35, 0.9),
            frameSize=(-0.04, 0.04, -0.025, 0.03),
            pos=(0.32, 0.0, 0.79),
            command=self.toggle_minimize,
        )
        self.btn_dock = DirectButton(
            parent=self.frame,
            text="<->",
            text_scale=0.026,
            text_fg=txt_fg,
            frameColor=(0.20, 0.25, 0.35, 0.9),
            frameSize=(-0.04, 0.04, -0.025, 0.03),
            pos=(0.23, 0.0, 0.79),
            command=self.toggle_dock,
        )

        # 2. Scenario Presets
        DirectLabel(
            parent=self.frame,
            text="PHYSICAL PRESETS",
            text_scale=0.024,
            text_fg=(0.8, 0.85, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, 0.72),
            text_align=TextNode.ALeft,
        )

        presets = [
            ("0 Cannon", 0), ("1 Fighter Jet", 1),
            ("2 Glider", 2), ("3 Rocket", 3),
            ("4 Satellite", 4), ("5 Pendulum", 5),
            ("6 Cyclotron", 6), ("7 Spheres", 7),
        ]

        self.preset_btns = []
        for i, (name, idx) in enumerate(presets):
            col_idx = i % 2
            row_idx = i // 2
            bx = -0.18 + col_idx * 0.36
            by = 0.66 - row_idx * 0.060
            btn = DirectButton(
                parent=self.frame,
                text=name,
                text_scale=0.024,
                text_fg=txt_fg,
                frameColor=btn_col,
                frameSize=(-0.17, 0.17, -0.022, 0.030),
                pos=(bx, 0.0, by),
                command=self.on_change_scenario,
                extraArgs=[idx],
                relief=DGG.RAISED,
                borderWidth=(0.004, 0.004),
            )
            self.preset_btns.append(btn)

        # 3. Interactive Physical Parameters
        y_param = 0.39
        DirectLabel(
            parent=self.frame,
            text="PARAMETER KNOBS",
            text_scale=0.024,
            text_fg=(0.8, 0.85, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_param),
            text_align=TextNode.ALeft,
        )

        # Mass Knob
        self.lbl_mass = DirectLabel(
            parent=self.frame,
            text="Mass: 15.0 kg",
            text_scale=0.024,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_param - 0.045),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.18, 0.0, y_param - 0.045), command=self.on_tweak_mass, extraArgs=[-0.25],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.28, 0.0, y_param - 0.045), command=self.on_tweak_mass, extraArgs=[0.25],
        )

        # Drag Cd Knob
        self.lbl_cd = DirectLabel(
            parent=self.frame,
            text="Drag Cd: 0.30",
            text_scale=0.024,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_param - 0.100),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.18, 0.0, y_param - 0.100), command=self.on_tweak_cd, extraArgs=[-0.05],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.28, 0.0, y_param - 0.100), command=self.on_tweak_cd, extraArgs=[0.05],
        )

        # Thrust / Force Knob
        self.lbl_thrust = DirectLabel(
            parent=self.frame,
            text="Thrust: 0 N",
            text_scale=0.024,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_param - 0.155),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.18, 0.0, y_param - 0.155), command=self.on_tweak_thrust, extraArgs=[-5000.0],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.28, 0.0, y_param - 0.155), command=self.on_tweak_thrust, extraArgs=[5000.0],
        )

        # Pitch Angle Knob
        self.lbl_angle = DirectLabel(
            parent=self.frame,
            text="Angle: 45.0°",
            text_scale=0.024,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_param - 0.210),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.18, 0.0, y_param - 0.210), command=self.on_tweak_angle, extraArgs=[-5.0],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.028, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.04, 0.04, -0.018, 0.025),
            pos=(0.28, 0.0, y_param - 0.210), command=self.on_tweak_angle, extraArgs=[5.0],
        )

        # 4. Simulation Actions
        y_act = y_param - 0.285
        DirectButton(
            parent=self.frame,
            text="Launch / Re-Fire (R)",
            text_scale=0.026,
            text_fg=txt_fg,
            frameColor=action_col,
            frameSize=(-0.34, 0.34, -0.025, 0.035),
            pos=(0.0, 0.0, y_act),
            command=self.on_launch_reset,
        )
        DirectButton(
            parent=self.frame,
            text="Pause / Resume",
            text_scale=0.023,
            text_fg=txt_fg,
            frameColor=btn_col,
            frameSize=(-0.16, 0.16, -0.022, 0.030),
            pos=(-0.18, 0.0, y_act - 0.065),
            command=self.on_toggle_pause,
        )
        DirectButton(
            parent=self.frame,
            text="Step Physics",
            text_scale=0.023,
            text_fg=txt_fg,
            frameColor=btn_col,
            frameSize=(-0.16, 0.16, -0.022, 0.030),
            pos=(0.18, 0.0, y_act - 0.065),
            command=self.on_step_physics,
        )

        # 5. 3D Camera Controls
        y_cam = y_act - 0.14
        DirectLabel(
            parent=self.frame,
            text="CAMERA MODES (O)",
            text_scale=0.024,
            text_fg=(0.8, 0.85, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_cam),
            text_align=TextNode.ALeft,
        )

        cam_modes = [("Free Cam", 0), ("Orbit Target", 1), ("Chase Cam", 2), ("Cockpit", 3)]
        for i, (cname, cmode) in enumerate(cam_modes):
            col_idx = i % 2
            row_idx = i // 2
            bx = -0.18 + col_idx * 0.36
            by = y_cam - 0.050 - row_idx * 0.058
            DirectButton(
                parent=self.frame,
                text=cname,
                text_scale=0.023,
                text_fg=txt_fg,
                frameColor=btn_col,
                frameSize=(-0.16, 0.16, -0.022, 0.030),
                pos=(bx, 0.0, by),
                command=self.on_change_cam_mode,
                extraArgs=[cmode],
            )

        # 6. 3D Visual Reference Toggles
        y_vis = y_cam - 0.17
        DirectLabel(
            parent=self.frame,
            text="VISUAL AIDS",
            text_scale=0.024,
            text_fg=(0.8, 0.85, 0.9, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.34, 0.0, y_vis),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="3D Axes", text_scale=0.022, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.10, 0.10, -0.02, 0.025),
            pos=(-0.24, 0.0, y_vis - 0.045), command=self.on_toggle_axes,
        )
        DirectButton(
            parent=self.frame, text="3D Grid", text_scale=0.022, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.10, 0.10, -0.02, 0.025),
            pos=(0.0, 0.0, y_vis - 0.045), command=self.on_toggle_grid,
        )
        DirectButton(
            parent=self.frame, text="Trail", text_scale=0.022, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.10, 0.10, -0.02, 0.025),
            pos=(0.24, 0.0, y_vis - 0.045), command=self.on_toggle_trail,
        )

        # Minimized Floating Bar (Slim top widget)
        self.min_bar = DirectFrame(
            frameColor=(0.08, 0.10, 0.14, 0.90),
            frameSize=(-0.45, 0.45, -0.04, 0.04),
            pos=(0.95, 0.0, 0.92),
        )
        self.min_bar.hide()
        DirectButton(
            parent=self.min_bar,
            text="[+] Expand Controls Panel (TAB)",
            text_scale=0.026,
            text_fg=accent_col,
            frameColor=(0, 0, 0, 0),
            pos=(0.0, 0.0, -0.01),
            command=self.toggle_minimize,
        )

    def toggle_minimize(self) -> None:
        self.is_minimized = not self.is_minimized
        if self.is_minimized:
            self.frame.hide()
            self.min_bar.show()
        else:
            self.frame.show()
            self.min_bar.hide()

    def toggle_panel_visibility(self) -> None:
        self.toggle_minimize()

    def toggle_dock(self) -> None:
        if self.dock_side == "right":
            self.dock_side = "left"
            self.frame.setPos(-1.22, 0.0, 0.0)
            self.min_bar.setPos(-0.95, 0.0, 0.92)
        else:
            self.dock_side = "right"
            self.frame.setPos(1.22, 0.0, 0.0)
            self.min_bar.setPos(0.95, 0.0, 0.92)

    def update_parameter_readouts(self, mass: float, cd: float, thrust: float, angle_deg: float) -> None:
        """Updates text labels with active physical parameter values."""
        self.lbl_mass["text"] = f"Mass: {mass:.1f} kg"
        self.lbl_cd["text"] = f"Drag Cd: {cd:.3f}"
        self.lbl_thrust["text"] = f"Thrust: {thrust:.0f} N"
        self.lbl_angle["text"] = f"Angle: {angle_deg:.1f}°"
