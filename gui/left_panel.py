#!/usr/bin/env python3
"""Left panel widget for drone address entry and swarm connection."""

import ast
import asyncio
import shutil
import time
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

NUM_DRONES = 3
TEMP_DIR = Path("temp")
POLL_INTERVAL = 0.2  # seconds between answer-file checks
TIMEOUT = 120        # seconds to wait for a response before giving up
RESET_COMMAND = "__RESET__"

# Drone names expected in LLM position responses (order maps to swarm drone index).
_DRONE_NAMES = ["red", "green", "blue"]


def _parse_positions(llm_response: str) -> list:
    """Parse drone target positions from an LLM response string.

    Accepts two dict formats::

        {'Red': {'x': 1.0, 'y': 2.0, 'z': 1.5}, 'Green': {...}, 'Blue': {...}}
        {'Red': [1.0, 2.0, 1.5], 'Green': [...], 'Blue': [...]}

    Returns a list of ``(x, y, z)`` tuples or ``None``, one per drone.
    Returns an all-``None`` list if parsing fails or the response is not a dict.
    """
    positions = [None, None, None]
    try:
        data = ast.literal_eval(llm_response.strip())
        if not isinstance(data, dict):
            return positions
        for i, name in enumerate(_DRONE_NAMES):
            # Accept both capitalised and lower-case keys.
            pos_data = data.get(name) or data.get(name.capitalize())
            if pos_data is None:
                continue
            if isinstance(pos_data, dict):
                if "x" in pos_data and "y" in pos_data and "z" in pos_data:
                    positions[i] = (
                        float(pos_data["x"]),
                        float(pos_data["y"]),
                        float(pos_data["z"]),
                    )
            elif isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                positions[i] = (
                    float(pos_data[0]),
                    float(pos_data[1]),
                    float(pos_data[2]),
                )
    except (ValueError, SyntaxError, KeyError, TypeError, IndexError):
        pass
    return positions


class LeftPanel(QFrame):
    connect_requested = pyqtSignal(str, int)
    disconnect_requested = pyqtSignal()
    takeoff_requested = pyqtSignal()
    land_requested = pyqtSignal()
    emergency_land_requested = pyqtSignal()
    goto_requested = pyqtSignal(list)  # list of (x,y,z)|None, one per drone

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Fleet")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        self._base_address_input = QLineEdit()
        self._base_address_input.setText("radio://0/75/2M/DB1F1010")

        connection_layout = QFormLayout()
        connection_layout.setSpacing(8)
        connection_layout.addRow("Base Address", self._base_address_input)
        layout.addLayout(connection_layout)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self._connect_btn)

        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.clicked.connect(self.disconnect_requested.emit)
        layout.addWidget(self._disconnect_btn)

        # -- Flight control buttons -----------------------------------------
        flight_btn_row = QHBoxLayout()
        flight_btn_row.setSpacing(6)

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self.takeoff_requested.emit)
        flight_btn_row.addWidget(self._start_btn)

        self._land_btn = QPushButton("Land")
        self._land_btn.clicked.connect(self.land_requested.emit)
        flight_btn_row.addWidget(self._land_btn)

        self._emergency_btn = QPushButton("Emergency Land")
        self._emergency_btn.setStyleSheet(
            "background-color: #cc2200; color: white; font-weight: bold;"
        )
        self._emergency_btn.clicked.connect(self.emergency_land_requested.emit)
        flight_btn_row.addWidget(self._emergency_btn)

        layout.addLayout(flight_btn_row)

        # -- Chat section ----------------------------------------------------
        chat_title = QLabel("LLM Chat")
        chat_title.setStyleSheet("font-size: 14px; font-weight: 600; margin-top: 6px;")
        layout.addWidget(chat_title)

        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setPlaceholderText("Chat history will appear here...")
        layout.addWidget(self._chat_display, 1)

        prompt_row = QHBoxLayout()
        self._prompt_input = QLineEdit()
        self._prompt_input.setPlaceholderText("Type a prompt and press Enter…")
        self._prompt_input.returnPressed.connect(self._on_send_clicked)
        prompt_row.addWidget(self._prompt_input)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._on_send_clicked)
        prompt_row.addWidget(self._send_btn)
        layout.addLayout(prompt_row)

        self._reset_btn = QPushButton("Reset Context")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._reset_btn)

    def _on_connect_clicked(self) -> None:
        base_address = self.get_base_address()
        if not base_address:
            QMessageBox.warning(self, "Invalid Base Address", "Base address cannot be empty.")
            return
        self.connect_requested.emit(base_address, NUM_DRONES)

    def get_base_address(self) -> str:
        return self._base_address_input.text().strip()

    # -- Chat ----------------------------------------------------------------

    def _on_send_clicked(self) -> None:
        prompt = self._prompt_input.text().strip()
        if not prompt:
            return
        self._prompt_input.clear()
        self._set_chat_busy(True)
        asyncio.ensure_future(self._send_and_receive(prompt))

    def _on_reset_clicked(self) -> None:
        self._set_chat_busy(True)
        asyncio.ensure_future(self._reset_context())

    def _set_chat_busy(self, busy: bool) -> None:
        self._prompt_input.setEnabled(not busy)
        self._send_btn.setEnabled(not busy)
        self._reset_btn.setEnabled(not busy)

    def _append_chat(self, role: str, text: str) -> None:
        self._chat_display.append(f"<b>{role}:</b> {text}")
        self._chat_display.append("")

    async def _send_and_receive(self, prompt: str) -> None:
        self._append_chat("You", prompt)
        try:
            answer = await self._send_raw(prompt)
            self._append_chat("LLM", answer)
            # Try to extract drone target positions from the response.
            positions = _parse_positions(answer)
            if any(p is not None for p in positions):
                self.goto_requested.emit(positions)
        except Exception as e:
            self._append_chat("Error", str(e))
        finally:
            self._set_chat_busy(False)

    async def _reset_context(self) -> None:
        try:
            answer = await self._send_raw(RESET_COMMAND)
            if answer.strip() == "__OK__":
                self._chat_display.clear()
                self._append_chat("System", "LLM context reset.")
            else:
                self._append_chat("System", f"Reset reply: {answer}")
        except Exception as e:
            self._append_chat("Error", str(e))
        finally:
            self._set_chat_busy(False)

    async def _send_raw(self, text: str) -> str:
        """Write text to the temp prompt file and wait for an answer."""
        TEMP_DIR.mkdir(exist_ok=True)
        answer_file = TEMP_DIR / "answer.txt"
        prompt_file = TEMP_DIR / "prompt.txt"

        written = False
        while not written:
            try:
                local_file = Path("prompt.txt")
                local_file.write_text(text)
                shutil.copy(local_file, str(prompt_file))
                written = True
            except Exception:
                await asyncio.sleep(0.1)

        if answer_file.exists():
            answer_file.unlink()

        deadline = time.monotonic() + TIMEOUT
        while not answer_file.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"No response from server after {TIMEOUT}s.")
            await asyncio.sleep(POLL_INTERVAL)

        return answer_file.read_text()

