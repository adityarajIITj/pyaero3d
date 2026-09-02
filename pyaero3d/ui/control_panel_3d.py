"""
PyAero3D - Interactive In-Viewport 3D GUI Control Panel.
Enables real-time parameter tweaking, preset switching, camera modes, and simulation controls directly inside the 3D viewport.
"""

from typing import Callable, Optional
from panda3d.core import TextNode, NodePath, Vec4
from direct.gui.DirectGui import (
    DirectFrame, DirectButton, DirectLabel, DGG
)


class InViewportControlPanel3D:
    """
    Interactive 3D Simulation Control Dashboard rendered over Panda3D viewport.
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

        self.is_visible = True

        # Main Panel Frame (Docked on Right Side)
        self.frame = DirectFrame(
            frameColor=(0.08, 0.10, 0.14, 0.88),
            frameSize=(-0.45, 0.45, -0.92, 0.92),
            pos=(1.15, 0.0, 0.0),
        )

        # Style colors
        btn_col = (0.16, 0.20, 0.28, 0.95)
        btn_hover = (0.24, 0.45, 0.75, 1.0)
        accent_col = (0.15, 0.70, 0.95, 1.0)
        action_col = (0.20, 0.65, 0.35, 1.0)
        txt_fg = (1.0, 1.0, 1.0, 1.0)

        # 1. Header Title
        DirectLabel(
            parent=self.frame,
            text="PYAERO3D LAB // 3D CONTROLS",
            text_scale=0.038,
            text_fg=accent_col,
            text_shadow=(0, 0, 0, 1),
            frameColor=(0, 0, 0, 0),
            pos=(0.0, 0.0, 0.85),
        )

        # 2. Scenario Presets
        DirectLabel(
            parent=self.frame,
            text="PHYSICAL SCENARIOS",
            text_scale=0.026,
            text_fg=(0.8, 0.8, 0.8, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, 0.78),
            text_align=TextNode.ALeft,
        )

        presets = [
            ("0 Ballistics", 0), ("1 Fighter Jet", 1),
            ("2 NACA Glider", 2), ("3 Space Rocket", 3),
            ("4 Satellite", 4), ("5 Double Pendulum", 5),
            ("6 Cyclotron", 6), ("7 Bouncing Spheres", 7),
        ]

        self.preset_btns = []
        for i, (name, idx) in enumerate(presets):
            col_idx = i % 2
            row_idx = i // 2
            bx = -0.22 + col_idx * 0.44
            by = 0.72 - row_idx * 0.065
            btn = DirectButton(
                parent=self.frame,
                text=name,
                text_scale=0.025,
                text_fg=txt_fg,
                frameColor=btn_col,
                frameSize=(-0.20, 0.20, -0.025, 0.035),
                pos=(bx, 0.0, by),
                command=self.on_change_scenario,
                extraArgs=[idx],
                relief=DGG.RAISED,
                borderWidth=(0.005, 0.005),
            )
            self.preset_btns.append(btn)

        # 3. Interactive Physical Parameters
        y_param = 0.42
        DirectLabel(
            parent=self.frame,
            text="PHYSICAL PARAMETERS & KNOBS",
            text_scale=0.026,
            text_fg=(0.8, 0.8, 0.8, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_param),
            text_align=TextNode.ALeft,
        )

        # Mass Knob
        self.lbl_mass = DirectLabel(
            parent=self.frame,
            text="Mass: 15.0 kg",
            text_scale=0.026,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_param - 0.05),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.20, 0.0, y_param - 0.05), command=self.on_tweak_mass, extraArgs=[-0.25],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.33, 0.0, y_param - 0.05), command=self.on_tweak_mass, extraArgs=[0.25],
        )

        # Drag Cd Knob
        self.lbl_cd = DirectLabel(
            parent=self.frame,
            text="Drag Cd: 0.30",
            text_scale=0.026,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_param - 0.11),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.20, 0.0, y_param - 0.11), command=self.on_tweak_cd, extraArgs=[-0.05],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.33, 0.0, y_param - 0.11), command=self.on_tweak_cd, extraArgs=[0.05],
        )

        # Thrust / Force Knob
        self.lbl_thrust = DirectLabel(
            parent=self.frame,
            text="Thrust: 0 N",
            text_scale=0.026,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_param - 0.17),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.20, 0.0, y_param - 0.17), command=self.on_tweak_thrust, extraArgs=[-5000.0],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.33, 0.0, y_param - 0.17), command=self.on_tweak_thrust, extraArgs=[5000.0],
        )

        # Pitch Angle Knob
        self.lbl_angle = DirectLabel(
            parent=self.frame,
            text="Launch Angle: 45.0°",
            text_scale=0.026,
            text_fg=txt_fg,
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_param - 0.23),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="-", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.20, 0.0, y_param - 0.23), command=self.on_tweak_angle, extraArgs=[-5.0],
        )
        DirectButton(
            parent=self.frame, text="+", text_scale=0.03, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.05, 0.05, -0.02, 0.03),
            pos=(0.33, 0.0, y_param - 0.23), command=self.on_tweak_angle, extraArgs=[5.0],
        )

        # 4. Simulation Play / Pause / Re-launch Actions
        y_act = y_param - 0.31
        DirectButton(
            parent=self.frame,
            text="Launch / Re-Fire (R)",
            text_scale=0.028,
            text_fg=txt_fg,
            frameColor=action_col,
            frameSize=(-0.40, 0.40, -0.03, 0.04),
            pos=(0.0, 0.0, y_act),
            command=self.on_launch_reset,
        )
        DirectButton(
            parent=self.frame,
            text="Pause / Resume",
            text_scale=0.025,
            text_fg=txt_fg,
            frameColor=btn_col,
            frameSize=(-0.19, 0.19, -0.025, 0.035),
            pos=(-0.21, 0.0, y_act - 0.075),
            command=self.on_toggle_pause,
        )
        DirectButton(
            parent=self.frame,
            text="Step Physics",
            text_scale=0.025,
            text_fg=txt_fg,
            frameColor=btn_col,
            frameSize=(-0.19, 0.19, -0.025, 0.035),
            pos=(0.21, 0.0, y_act - 0.075),
            command=self.on_step_physics,
        )

        # 5. 3D Camera Controls
        y_cam = y_act - 0.16
        DirectLabel(
            parent=self.frame,
            text="3D CAMERA MODES",
            text_scale=0.026,
            text_fg=(0.8, 0.8, 0.8, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_cam),
            text_align=TextNode.ALeft,
        )

        cam_modes = [("Free Cam", 0), ("Orbit Target", 1), ("Chase Cam", 2), ("Cockpit", 3)]
        for i, (cname, cmode) in enumerate(cam_modes):
            col_idx = i % 2
            row_idx = i // 2
            bx = -0.21 + col_idx * 0.42
            by = y_cam - 0.055 - row_idx * 0.065
            DirectButton(
                parent=self.frame,
                text=cname,
                text_scale=0.025,
                text_fg=txt_fg,
                frameColor=btn_col,
                frameSize=(-0.19, 0.19, -0.025, 0.035),
                pos=(bx, 0.0, by),
                command=self.on_change_cam_mode,
                extraArgs=[cmode],
            )

        # 6. 3D Visual Reference Toggles
        y_vis = y_cam - 0.19
        DirectLabel(
            parent=self.frame,
            text="3D VISUAL AIDS",
            text_scale=0.026,
            text_fg=(0.8, 0.8, 0.8, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.40, 0.0, y_vis),
            text_align=TextNode.ALeft,
        )
        DirectButton(
            parent=self.frame, text="Toggle 3D Axes", text_scale=0.023, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.13, 0.13, -0.02, 0.03),
            pos=(-0.28, 0.0, y_vis - 0.05), command=self.on_toggle_axes,
        )
        DirectButton(
            parent=self.frame, text="Toggle 3D Grid", text_scale=0.023, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.13, 0.13, -0.02, 0.03),
            pos=(0.0, 0.0, y_vis - 0.05), command=self.on_toggle_grid,
        )
        DirectButton(
            parent=self.frame, text="Toggle Trail", text_scale=0.023, text_fg=txt_fg,
            frameColor=btn_col, frameSize=(-0.13, 0.13, -0.02, 0.03),
            pos=(0.28, 0.0, y_vis - 0.05), command=self.on_toggle_trail,
        )

        # Hide/Show Panel Toggle Button (Docked on Top Right)
        self.btn_toggle_panel = DirectButton(
            text="[ Panel: TAB ]",
            text_scale=0.028,
            text_fg=txt_fg,
            frameColor=(0.10, 0.14, 0.20, 0.90),
            frameSize=(-0.15, 0.15, -0.025, 0.035),
            pos=(0.80, 0.0, 0.92),
            command=self.toggle_panel_visibility,
        )

    def toggle_panel_visibility(self) -> None:
        self.is_visible = not self.is_visible
        if self.is_visible:
            self.frame.show()
        else:
            self.frame.hide()

    def update_parameter_readouts(self, mass: float, cd: float, thrust: float, angle_deg: float) -> None:
        """Updates text labels with active physical parameter values."""
        self.lbl_mass["text"] = f"Mass: {mass:.1f} kg"
        self.lbl_cd["text"] = f"Drag Cd: {cd:.3f}"
        self.lbl_thrust["text"] = f"Thrust: {thrust:.0f} N"
        self.lbl_angle["text"] = f"Launch Angle: {angle_deg:.1f}°"
