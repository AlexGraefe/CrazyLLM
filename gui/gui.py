#!/usr/bin/env python3
"""Swarm Controller GUI — skeleton.

Three-panel layout:
  Left   – drone fleet / connection status
  Center – 3-D visualisation

Entry point: run with  python gui.py
Requires:  PyQt6  qasync
"""

import asyncio
import sys
from enum import Enum, auto
from typing import Callable

import qasync
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from gui.left_panel import LeftPanel
from gui.middle_panel import MiddlePanel
from hardware.swarm import Swarm, SwarmState


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class AppState(Enum):
    IDLE       = auto()
    CONNECTING = auto()
    READY      = auto()
    SIMULATING = auto()
    FLYING     = auto()
    LANDING    = auto()
    ERROR      = auto()


_StateListener = Callable[[AppState], None]


class StateMachine:
    """Async state machine that drives the swarm controller lifecycle."""

    def __init__(self) -> None:
        self._state: AppState = AppState.IDLE
        self._listeners: list[_StateListener] = []

    # -- Public API ----------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    def add_listener(self, cb: _StateListener) -> None:
        self._listeners.append(cb)

    async def transition(self, new_state: AppState) -> None:
        print(f"[SM] {self._state.name} -> {new_state.name}")
        self._state = new_state
        for cb in self._listeners:
            cb(new_state)

    # -- Main loop -----------------------------------------------------------

    async def run(self) -> None:
        """Drive state transitions.
        Expand with guard conditions, timers, and backend events later.
        """
        await self.transition(AppState.IDLE)

        # TODO: react to UI events and backend callbacks, e.g.:
        #   await self.transition(AppState.CONNECTING)
        #   await self._wait_for_all_connected()
        #   await self.transition(AppState.READY)
        #   ...


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, sm: StateMachine) -> None:
        super().__init__()
        self._sm = sm
        self._sm.add_listener(self._on_state_changed)
        self._transition_task: asyncio.Task | None = None
        self._swarm = Swarm(self)

        self.setWindowTitle("Swarm Controller")
        self.resize(1600, 900)

        self._setup_ui()
        self._update_status_bar(sm.state)

    # -- UI setup ------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.panel_left   = LeftPanel()
        self.panel_left.connect_requested.connect(self._on_connect_clicked)
        self.panel_left.disconnect_requested.connect(self._on_disconnect_clicked)
        self.panel_left.takeoff_requested.connect(self._on_takeoff_clicked)
        self.panel_left.land_requested.connect(self._on_land_clicked)
        self.panel_left.emergency_land_requested.connect(self._on_emergency_land_clicked)
        self.panel_left.goto_requested.connect(self._on_goto_requested)
        self.panel_center = MiddlePanel()

        root.addWidget(self.panel_left,   stretch=1)
        root.addWidget(self.panel_center, stretch=3)

        # Status bar driven by the state machine
        self._status_label = QLabel()
        status_bar = QStatusBar()
        status_bar.addPermanentWidget(self._status_label)
        self.setStatusBar(status_bar)

    @staticmethod
    def _make_panel(title: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Raised)

        # Placeholder title — remove once real content is added
        label = QLabel(title, frame)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: gray; font-size: 18px;")
        label.setGeometry(0, 0, 200, 40)          # resized in resizeEvent later
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        return frame

    # -- State machine callback ---------------------------------------------

    def _on_connect_clicked(self, base_address: str, num_drones: int) -> None:
        asyncio.create_task(self._connect_drones(base_address, num_drones))

    def _on_disconnect_clicked(self) -> None:
        asyncio.create_task(self._disconnect_and_set_idle())

    def _on_takeoff_clicked(self) -> None:
        self._swarm.takeoff()

    def _on_land_clicked(self) -> None:
        self._swarm.land()

    def _on_emergency_land_clicked(self) -> None:
        asyncio.create_task(self._emergency_land_drones())

    def _on_goto_requested(self, positions: list) -> None:
        """Forward target positions from LLM response to the swarm and visualiser."""
        self.panel_center.update_target_positions(positions)
        self._swarm.goto_positions(positions)

    async def _connect_drones(self, base_address: str, num_drones: int) -> None:
        await self._sm.transition(AppState.CONNECTING)
        self._swarm.connect(base_address, num_drones)
        # on_swarm_state_changed will drive the AppState transition to READY or ERROR.

    async def _disconnect_and_set_idle(self) -> None:
        await self._swarm.disconnect()
        # on_swarm_state_changed(UNCONNECTED) will transition AppState to IDLE.

    async def _emergency_land_drones(self) -> None:
        await self._swarm.emergency_land()

    # -- SwarmGUI callbacks --------------------------------------------------

    def on_swarm_state_changed(self, state: SwarmState) -> None:
        """Map SwarmState to AppState and drive the GUI state machine."""
        mapping = {
            SwarmState.UNCONNECTED: AppState.IDLE,
            SwarmState.CONNECTED:   AppState.READY,
            SwarmState.FLYING:      AppState.FLYING,
            SwarmState.LANDED:      AppState.LANDING,
            SwarmState.ERROR:       AppState.ERROR,
        }
        app_state = mapping.get(state)
        if app_state is not None:
            asyncio.create_task(self._sm.transition(app_state))

    def on_phase_changed(
        self,
        phase_name: str,
        splines: list,
        segment_duration: float,
    ) -> None:
        self.panel_center.set_live_phase_curves(phase_name, splines, segment_duration)

    def on_live_positions(self, positions: list) -> None:
        self.panel_center.update_live_positions(positions)

    def on_live_mode_started(self, n_drones: int) -> None:
        self.panel_center.start_live_mode(n_drones)

    def on_live_mode_stopped(self) -> None:
        self.panel_center.stop_live_mode()

    def on_fly_enabled(self, enabled: bool) -> None:
        pass

    def on_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _on_state_changed(self, state: AppState) -> None:
        """React to every state transition. Expand per-state logic here."""
        self._update_status_bar(state)

        if state == AppState.IDLE:
            pass  # TODO: reset UI
        elif state == AppState.CONNECTING:
            pass  # TODO: show spinner / disable controls
        elif state == AppState.READY:
            pass  # TODO: enable fly button
        elif state == AppState.SIMULATING:
            pass  # TODO: run simulation flow
        elif state == AppState.FLYING:
            pass  # TODO: start live visualisation
        elif state == AppState.LANDING:
            pass  # TODO: show landing progress
        elif state == AppState.ERROR:
            pass  # TODO: show error dialog

    def _update_status_bar(self, state: AppState) -> None:
        self._status_label.setText(f"State: {state.name}")


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------

async def _async_main() -> None:
    sm = StateMachine()
    window = MainWindow(sm)
    window.show()

    # Run the state machine as a concurrent background task
    asyncio.ensure_future(sm.run())


def main() -> None:
    app = QApplication(sys.argv)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_until_complete(_async_main())
        loop.run_forever()   # keeps running until the window is closed
