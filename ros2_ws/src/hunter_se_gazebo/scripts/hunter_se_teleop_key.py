#!/usr/bin/env python3
"""
Unity-style axis-based keyboard teleop for Hunter SE (Ackermann steering).

This implementation uses a focused Tk window for real key press / release
events. That avoids the terminal auto-repeat delay problem and also avoids
global desktop key capture.

Mirrors Unity VehicleController.cs intent:
  Input.GetAxis("Vertical")    W / Up    = +1, S / Down  = -1
  Input.GetAxis("Horizontal")  A / Left  = +1, D / Right = -1

Key pressed  -> axis ramps toward +/-1 at SENSITIVITY units/s
Key released -> axis decays toward 0 at GRAVITY units/s

Note on braking:
Unity's VehicleController applies brake torque when throttle reaches zero.
Gazebo's AckermannSteering plugin tracks target velocity instead, so sending
linear.x = 0 produces similar user experience via a different physics path.
"""

from __future__ import annotations

import math
import signal
import sys
import threading
import time
import tkinter as tk
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


LINEAR_SENSITIVITY: float = 3.0
LINEAR_GRAVITY: float = 3.0
STEERING_SENSITIVITY: float = 5.0
STEERING_GRAVITY: float = 12.0
DEAD_ZONE: float = 0.01
PUBLISH_HZ: float = 20.0
HUNTER_SE_MAX_SPEED_MPS: float = 1.333
HUNTER_SE_CENTER_STEER_LIMIT_DEG: float = 21.58
HUNTER_SE_MAX_STEERING_RAD: float = math.radians(HUNTER_SE_CENTER_STEER_LIMIT_DEG)

_UP = "UP"
_DOWN = "DOWN"
_LEFT = "LEFT"
_RIGHT = "RIGHT"
_STOP = "STOP"

_SPECIAL_MAP = {
    "Up": _UP,
    "Down": _DOWN,
    "Left": _LEFT,
    "Right": _RIGHT,
    "space": _STOP,
}

_MOVEMENT_KEYS = {_UP, _DOWN, _LEFT, _RIGHT, "w", "s", "a", "d"}
_SPEED_KEYS = {"q", "z", "e", "c"}

HELP = """Hunter SE Teleop (Unity-Axis Style)

Focus this window, then use:
  W / Up      forward
  S / Down    backward
  A / Left    steer left
  D / Right   steer right

  Q / Z       linear speed +/-10%
  E / C       steering range +/-10%
  Space / K   immediate stop
  Esc         quit

Hold key  -> smooth ramp toward +/-1
Release   -> smooth decay toward 0
"""


class KeyboardState:
    """Focused-window keyboard state with repeat-safe press/release handling."""

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._speed_down: set[str] = set()
        self._events: deque[str] = deque()
        self._quit = False
        self._lock = threading.Lock()
        self._release_jobs: dict[str, str] = {}

        self._root = tk.Tk()
        self._root.title("Hunter SE Teleop")
        self._root.geometry("420x220")
        self._root.resizable(False, False)
        self._root.configure(padx=14, pady=14)

        self._help_var = tk.StringVar(value=HELP)
        self._status_var = tk.StringVar(value="Click this window to capture keyboard input.")

        tk.Label(
            self._root,
            textvariable=self._help_var,
            justify="left",
            anchor="w",
            font=("TkFixedFont", 11),
        ).pack(fill="both", expand=True)
        tk.Label(
            self._root,
            textvariable=self._status_var,
            justify="left",
            anchor="w",
            font=("TkFixedFont", 10),
        ).pack(fill="x", pady=(10, 0))

        self._root.bind("<KeyPress>", self._on_press)
        self._root.bind("<KeyRelease>", self._on_release)
        self._root.protocol("WM_DELETE_WINDOW", self.request_quit)
        self._root.after(50, self._pump_quit)
        self._root.after(100, self._focus_window)

    @staticmethod
    def _normalize(event: tk.Event) -> str | None:
        keysym = event.keysym
        if keysym in _SPECIAL_MAP:
            return _SPECIAL_MAP[keysym]
        if len(keysym) == 1:
            return keysym.lower()
        if keysym == "Escape":
            return "escape"
        return None

    def _focus_window(self) -> None:
        try:
            self._root.lift()
            self._root.focus_force()
        except tk.TclError:
            pass

    def _cancel_pending_release(self, name: str) -> None:
        job = self._release_jobs.pop(name, None)
        if job is not None:
            try:
                self._root.after_cancel(job)
            except tk.TclError:
                pass

    def _finalize_release(self, name: str) -> None:
        self._release_jobs.pop(name, None)
        with self._lock:
            self._pressed.discard(name)
            self._speed_down.discard(name)

    def _on_press(self, event: tk.Event) -> None:
        name = self._normalize(event)
        if name is None:
            return

        self._cancel_pending_release(name)

        if name == "escape":
            self.request_quit()
            return

        with self._lock:
            if name in _MOVEMENT_KEYS:
                self._pressed.add(name)
            elif name in _SPEED_KEYS:
                if name not in self._speed_down:
                    self._events.append(name)
                    self._speed_down.add(name)
            elif name in {_STOP, "k"}:
                self._pressed.clear()
                self._events.append(_STOP)

    def _on_release(self, event: tk.Event) -> None:
        name = self._normalize(event)
        if name is None or name in {"escape", _STOP, "k"}:
            return

        # Tk auto-repeat may emit a release immediately before the next repeated
        # press. Defer release until idle so a following press can cancel it.
        self._cancel_pending_release(name)
        self._release_jobs[name] = self._root.after_idle(
            lambda n=name: self._finalize_release(n)
        )

    def _pump_quit(self) -> None:
        if self.quit_requested:
            try:
                self._root.quit()
            except tk.TclError:
                pass
            return
        self._root.after(50, self._pump_quit)

    def run(self) -> None:
        self._root.mainloop()

    def request_quit(self) -> None:
        with self._lock:
            self._quit = True
            self._pressed.clear()
            self._speed_down.clear()

    def is_held(self, *keys: str) -> bool:
        with self._lock:
            return any(k in self._pressed for k in keys)

    def pop_event(self) -> str | None:
        with self._lock:
            return self._events.popleft() if self._events else None

    @property
    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit

    def set_status(self, text: str) -> None:
        try:
            self._root.after(0, lambda: self._status_var.set(text))
        except tk.TclError:
            pass


class TeleopNode(Node):
    """Publishes smoothed Twist commands derived from Unity-like axis values."""

    def __init__(self) -> None:
        super().__init__("hunter_se_teleop_key")
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._vertical = 0.0
        self._horizontal = 0.0
        self.lin_spd = HUNTER_SE_MAX_SPEED_MPS
        self.ang_spd = HUNTER_SE_MAX_STEERING_RAD
        self._dt = 1.0 / PUBLISH_HZ
        self._kb: KeyboardState | None = None
        self.create_timer(self._dt, self._tick)

    def set_keyboard(self, kb: KeyboardState) -> None:
        self._kb = kb

    def _smooth(self, current: float, target: float, sensitivity: float, gravity: float) -> float:
        rate = sensitivity if target != 0.0 else gravity
        step = rate * self._dt
        delta = target - current
        if abs(delta) <= step:
            return target
        return current + step * (1.0 if delta > 0.0 else -1.0)

    def _tick(self) -> None:
        kb = self._kb
        if kb is None:
            return

        while (ev := kb.pop_event()) is not None:
            if ev == "q":
                self.lin_spd = round(min(self.lin_spd * 1.1, HUNTER_SE_MAX_SPEED_MPS), 3)
            elif ev == "z":
                self.lin_spd = round(max(self.lin_spd * 0.9, 0.1), 3)
            elif ev == "e":
                self.ang_spd = round(min(self.ang_spd * 1.1, HUNTER_SE_MAX_STEERING_RAD), 3)
            elif ev == "c":
                self.ang_spd = round(max(self.ang_spd * 0.9, 0.1), 3)
            elif ev == _STOP:
                self._vertical = 0.0
                self._horizontal = 0.0

        w = kb.is_held("w", _UP)
        s = kb.is_held("s", _DOWN)
        a = kb.is_held("a", _LEFT)
        d = kb.is_held("d", _RIGHT)

        v_target = 1.0 if w and not s else -1.0 if s and not w else 0.0
        h_target = 1.0 if a and not d else -1.0 if d and not a else 0.0

        self._vertical = self._smooth(
            self._vertical,
            v_target,
            LINEAR_SENSITIVITY,
            LINEAR_GRAVITY,
        )
        self._horizontal = self._smooth(
            self._horizontal,
            h_target,
            STEERING_SENSITIVITY,
            STEERING_GRAVITY,
        )

        if abs(self._vertical) < DEAD_ZONE:
            self._vertical = 0.0
        if abs(self._horizontal) < DEAD_ZONE:
            self._horizontal = 0.0

        msg = Twist()
        msg.linear.x = self._vertical * self.lin_spd

        # hunter_se_cmd_prefilter expects angular.z as target center steering
        # angle [rad]. Steering orientation should stay aligned with the
        # pressed left/right key in both forward and reverse.
        msg.angular.z = self._horizontal * self.ang_spd
        self._pub.publish(msg)

        kb.set_status(
            f"linear={msg.linear.x:+.3f} m/s   steering={msg.angular.z:+.3f} rad   "
            f"speed={self.lin_spd:.2f} / steer={self.ang_spd:.2f}"
        )


def main() -> None:
    if not tk._default_root and not hasattr(tk, "Tk"):
        print("[hunter_se_teleop] tkinter is unavailable.", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = TeleopNode()
    kb = KeyboardState()
    node.set_keyboard(kb)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    def _sigint(_sig, _frame) -> None:
        kb.request_quit()

    signal.signal(signal.SIGINT, _sigint)

    try:
        kb.run()
    finally:
        kb.request_quit()
        stop = Twist()
        node._pub.publish(stop)
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
