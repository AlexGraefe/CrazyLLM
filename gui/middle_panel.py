#!/usr/bin/env python3
"""Middle panel widget for active phase swarm trajectory visualization."""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFrame, QVBoxLayout

DEFAULT_SEGMENT_DURATION = 0.1
POINTS_PER_SEGMENT = 10
DRONE_COLORS = ["#FF0000", "#00FF00", "#0000FF"]  # Red, Green, Blue – matches LED_COLORS in swarm.py
HOVER_SECONDS = 5.0
STITCH_POINT_REPEAT = 10
TESTBED_X_MIN = -4.0
TESTBED_X_MAX = 4.0
TESTBED_Y_MIN = -4.0
TESTBED_Y_MAX = 4.0
TESTBED_Z_MIN = 0.0
TESTBED_Z_MAX = 3.0


def interpolate_trajectory(
	waypoints_x: np.ndarray,
	waypoints_y: np.ndarray,
	waypoints_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Fit Bezier spline through waypoints and sample a smooth curve."""
	waypoints = np.column_stack((waypoints_x, waypoints_y, waypoints_z))
	spline = None

	n_segments = len(spline.beziers)
	t_fine = np.linspace(0, n_segments, n_segments * POINTS_PER_SEGMENT)
	points = np.array([spline.evaluate(t) for t in t_fine])
	return points[:, 0], points[:, 1], points[:, 2]


class MiddlePanel(QFrame):
	"""3D trajectory visualizer embedded in the middle GUI panel."""

	def __init__(self) -> None:
		super().__init__()
		self.setFrameShape(QFrame.Shape.StyledPanel)
		self.setFrameShadow(QFrame.Shadow.Raised)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)

		self._fig = plt.Figure(figsize=(8, 6))
		self._canvas = FigureCanvasQTAgg(self._fig)
		layout.addWidget(self._canvas)

		self._ax = self._fig.add_subplot(111, projection="3d")
		self._timer = QTimer(self)
		self._timer.timeout.connect(self._update_frame)

		self._frame_idx = 0
		self._n_frames = 0
		self._n_drones = 0
		self._seconds_per_frame = DEFAULT_SEGMENT_DURATION / POINTS_PER_SEGMENT
		self._frame_times = np.array([], dtype=float)
		self._sim_start_time = 0.0
		self._last_rendered_frame = -1
		self._all_xi: list[np.ndarray] = []
		self._all_yi: list[np.ndarray] = []
		self._all_zi: list[np.ndarray] = []
		self._trails = []
		self._markers = []
		self._target_markers = []
		self._labels = []
		self._trail_len = POINTS_PER_SEGMENT * 3
		self._mode = "simulation"
		self._phase_xi: list[np.ndarray] = []
		self._phase_yi: list[np.ndarray] = []
		self._phase_zi: list[np.ndarray] = []
		self._phase_durations: list[float] = []
		self._phase_start_time: float | None = None
		self._current_phase_name = ""
		self._target_positions_data: list = []  # (x,y,z)|None per drone from LLM goto

		self._show_message("Select a valid folder and click Simulate")

	def start_live_mode(self, n_drones: int) -> None:
		"""Switch panel to live measured-position rendering mode."""
		self._timer.stop()
		self._mode = "live"
		self._n_drones = n_drones
		self._all_xi = [np.array([], dtype=float) for _ in range(n_drones)]
		self._all_yi = [np.array([], dtype=float) for _ in range(n_drones)]
		self._all_zi = [np.array([], dtype=float) for _ in range(n_drones)]
		self._phase_xi = []
		self._phase_yi = []
		self._phase_zi = []
		self._phase_durations = []
		self._phase_start_time = None
		self._current_phase_name = ""
		self._redraw_live_scene()

	def set_live_phase_curves(
		self,
		phase_name: str,
		splines: list,
		segment_duration: float,
	) -> None:
		"""Plot the full trajectory curve for the currently flying phase."""
		if self._mode != "live":
			return

		phase_xi: list[np.ndarray] = []
		phase_yi: list[np.ndarray] = []
		phase_zi: list[np.ndarray] = []

		for spline in splines:
			n_segments = len(spline.beziers)
			if n_segments <= 0:
				phase_xi.append(np.array([], dtype=float))
				phase_yi.append(np.array([], dtype=float))
				phase_zi.append(np.array([], dtype=float))
				continue

			t_fine = np.linspace(0, n_segments, max(2, n_segments * POINTS_PER_SEGMENT + 1))
			points = np.array([spline.evaluate(t) for t in t_fine])
			phase_xi.append(points[:, 0])
			phase_yi.append(points[:, 1])
			phase_zi.append(points[:, 2])

		self._phase_xi = phase_xi
		self._phase_yi = phase_yi
		self._phase_zi = phase_zi
		self._phase_durations = [len(spline.beziers) * segment_duration for spline in splines]
		self._phase_start_time = time.perf_counter()
		self._current_phase_name = phase_name
		self._redraw_live_scene()

	def _redraw_live_scene(self) -> None:
		"""Rebuild live scene preserving measured trail data and current phase curve."""
		self._fig.clf()
		self._ax = self._fig.add_subplot(111, projection="3d")

		self._trails = []
		self._markers = []
		self._target_markers = []
		self._labels = []

		for n in range(self._n_drones):
			color = DRONE_COLORS[n % len(DRONE_COLORS)]
			if n < len(self._phase_xi) and len(self._phase_xi[n]) > 0:
				self._ax.plot(
					self._phase_xi[n],
					self._phase_yi[n],
					self._phase_zi[n],
					color=color,
					linewidth=0.8,
					alpha=0.25,
				)

			trail, = self._ax.plot([], [], [], color=color, linewidth=1.5)
			marker, = self._ax.plot(
				[],
				[],
				[],
				"o",
				color=color,
				markersize=8,
				label=f"Drone {n}",
			)
			target_marker, = self._ax.plot(
				[],
				[],
				[],
				"x",
				color=color,
				markersize=8,
				markeredgewidth=2.0,
			)
			label = self._ax.text(
				0, 0, 0,
				str(n),
				color=color,
				fontsize=8,
				visible=False,
			)
			self._trails.append(trail)
			self._markers.append(marker)
			self._target_markers.append(target_marker)
			self._labels.append(label)

			if n < len(self._all_xi) and len(self._all_xi[n]) > 0:
				start = max(0, len(self._all_xi[n]) - self._trail_len)
				trail.set_data_3d(
					self._all_xi[n][start:],
					self._all_yi[n][start:],
					self._all_zi[n][start:],
				)
				marker.set_data_3d(
					[self._all_xi[n][-1]],
					[self._all_yi[n][-1]],
					[self._all_zi[n][-1]],
				)
				label.set_position((self._all_xi[n][-1], self._all_yi[n][-1]))
				label.set_3d_properties(self._all_zi[n][-1] + 0.1, zdir="z")
				label.set_visible(True)

			if n < len(self._phase_xi) and len(self._phase_xi[n]) > 0:
				target_marker.set_data_3d(
					[self._phase_xi[n][0]],
					[self._phase_yi[n][0]],
					[self._phase_zi[n][0]],
				)

		self._ax.set_xlabel("X (m)")
		self._ax.set_ylabel("Y (m)")
		self._ax.set_zlabel("Z (m)")
		self._set_testbed_axes()

		# Draw LLM-commanded target positions as large orange X markers.
		for idx, pos in enumerate(self._target_positions_data):
			if pos is None:
				continue
			tx, ty, tz = pos
			color = DRONE_COLORS[idx % len(DRONE_COLORS)]
			self._ax.scatter(
				[tx], [ty], [tz],
				marker="x",
				s=200,
				color=color,
				linewidths=3,
				zorder=10,
				label=f"Target {idx}",
			)

		title = f"Live measured positions - {self._n_drones} drone(s)"
		if self._current_phase_name:
			title += f" | phase: {self._current_phase_name}"
		self._ax.set_title(title)
		self._ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

		self._fig.tight_layout()
		self._canvas.draw_idle()

	def update_live_positions(self, positions: list[tuple[float, float, float] | None]) -> None:
		"""Update live mode with newest measured positions for each drone."""
		if self._mode != "live" or not positions:
			return

		n_drones = min(self._n_drones, len(positions))
		for idx in range(n_drones):
			pos = positions[idx]
			if pos is None:
				continue

			x, y, z = pos
			self._all_xi[idx] = np.append(self._all_xi[idx], x)
			self._all_yi[idx] = np.append(self._all_yi[idx], y)
			self._all_zi[idx] = np.append(self._all_zi[idx], z)

			start = max(0, len(self._all_xi[idx]) - self._trail_len)
			self._trails[idx].set_data_3d(
				self._all_xi[idx][start:],
				self._all_yi[idx][start:],
				self._all_zi[idx][start:],
			)
			self._markers[idx].set_data_3d([x], [y], [z])
			self._labels[idx].set_position((x, y))
			self._labels[idx].set_3d_properties(z + 0.1, zdir="z")
			self._labels[idx].set_visible(True)

		if self._phase_start_time is not None:
			elapsed = max(0.0, time.perf_counter() - self._phase_start_time)
			for idx in range(min(n_drones, len(self._target_markers), len(self._phase_xi), len(self._phase_durations))):
				if len(self._phase_xi[idx]) == 0:
					continue

				duration = max(self._phase_durations[idx], 1e-6)
				progress = min(1.0, elapsed / duration)
				frame_idx = int(round(progress * (len(self._phase_xi[idx]) - 1)))
				self._target_markers[idx].set_data_3d(
					[self._phase_xi[idx][frame_idx]],
					[self._phase_yi[idx][frame_idx]],
					[self._phase_zi[idx][frame_idx]],
				)

		self._canvas.draw_idle()

	def stop_live_mode(self) -> None:
		"""Leave live mode and keep current view until next simulation starts."""
		self._mode = "simulation"
		self._phase_start_time = None

	def update_target_positions(self, positions: list) -> None:
		"""Show LLM-commanded target positions as X markers in the 3-D view.

		*positions* is a list of ``(x, y, z)`` tuples or ``None``, one entry per
		drone.  Triggers a full scene redraw when in live mode.
		"""
		self._target_positions_data = positions
		if self._mode == "live":
			self._redraw_live_scene()

	def start_simulation(
		self,
		takeoff_csv_path: Path,
		active_csv_path: Path,
		landing_csv_path: Path,
		dt_start: float,
		dt_show: float,
		num_trials: int,
		wait_after_takeoff: float = HOVER_SECONDS,
		wait_between_passes: float = HOVER_SECONDS,
		wait_before_landing: float = HOVER_SECONDS,
	) -> bool:
		"""Load all phase CSVs and (re)start the full show animation."""
		self._mode = "simulation"
		self._timer.stop()

		full_show = self._build_full_show_data(
			takeoff_csv_path,
			active_csv_path,
			landing_csv_path,
			dt_start,
			dt_show,
			num_trials,
			wait_after_takeoff,
			wait_between_passes,
			wait_before_landing,
		)
		if full_show is None:
			return False

		all_xi, all_yi, all_zi, frame_times = full_show

		if not all_xi:
			self._show_message("No trajectory points to render")
			return False

		self._all_xi = all_xi
		self._all_yi = all_yi
		self._all_zi = all_zi
		self._frame_times = frame_times
		self._n_drones = len(all_xi)
		self._n_frames = len(all_xi[0])
		self._frame_idx = 0

		self._build_scene()

		self._seconds_per_frame = max(dt_show / POINTS_PER_SEGMENT, 1e-6)
		self._sim_start_time = time.perf_counter()
		self._last_rendered_frame = -1

		# Timer cadence is decoupled from simulation cadence.
		# Each tick renders the frame that should be visible "now" and
		# naturally skips intermediate frames if rendering falls behind.
		self._timer.start(16)
		self._canvas.draw_idle()
		return True

	def _build_full_show_data(
		self,
		takeoff_csv_path: Path,
		active_csv_path: Path,
		landing_csv_path: Path,
		dt_start: float,
		dt_show: float,
		num_trials: int,
		wait_after_takeoff: float = HOVER_SECONDS,
		wait_between_passes: float = HOVER_SECONDS,
		wait_before_landing: float = HOVER_SECONDS,
	) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray] | None:
		"""Load all phase CSVs and assemble full-show sampled paths."""

		takeoff_xyz = self._load_phase_trajectories(takeoff_csv_path)
		if takeoff_xyz is None:
			return None

		takeoff_last_waypoints = self._load_phase_last_waypoints(takeoff_csv_path)
		if takeoff_last_waypoints is None:
			return None

		show_forward_xyz = self._load_phase_trajectories(
			active_csv_path,
			prepend_points=takeoff_last_waypoints,
		)
		if show_forward_xyz is None:
			return None

		show_last_waypoints = self._load_phase_last_waypoints(active_csv_path)
		if show_last_waypoints is None:
			return None

		landing_xyz = self._load_phase_trajectories(
			landing_csv_path,
			prepend_points=show_last_waypoints,
		)
		if landing_xyz is None:
			return None

		n_drones = len(show_forward_xyz[0])
		if len(takeoff_xyz[0]) != n_drones or len(landing_xyz[0]) != n_drones:
			self._show_message("Phase CSV files contain different drone counts")
			return None

		show_backward_xyz = tuple(
			[np.flip(arr, axis=0) for arr in axis_set]
			for axis_set in show_forward_xyz
		)

		phase_durations = {
			"takeoff": max(dt_start / POINTS_PER_SEGMENT, 1e-6),
			"show": max(dt_show / POINTS_PER_SEGMENT, 1e-6),
			"landing": max(dt_start / POINTS_PER_SEGMENT, 1e-6),
		}

		all_xi, all_yi, all_zi, frame_times = self._compose_full_show(
			takeoff_xyz,
			show_forward_xyz,
			show_backward_xyz,
			landing_xyz,
			phase_durations,
			max(1, num_trials),
			wait_after_takeoff,
			wait_between_passes,
			wait_before_landing,
		)

		if not all_xi:
			return None

		return all_xi, all_yi, all_zi, frame_times

	def _load_phase_last_waypoints(self, csv_path: Path) -> list[np.ndarray] | None:
		try:
			df = pd.read_csv(csv_path)
		except OSError:
			self._show_message(f"Could not open {csv_path.name}")
			return None

		x_columns = sorted(
			(col for col in df.columns if col.startswith("x")),
			key=lambda name: int(name[1:]) if name[1:].isdigit() else 10**9,
		)
		n_drones = len(x_columns)
		if n_drones == 0:
			self._show_message(f"No xN,yN,zN columns found in {csv_path.name}")
			return None

		last_waypoints: list[np.ndarray] = []
		for n in range(n_drones):
			try:
				waypoints = np.column_stack(
					(
						df[f"x{n}"].values,
						df[f"y{n}"].values,
						df[f"z{n}"].values,
					)
				)
			except KeyError:
				self._show_message(f"Missing axis columns for drone index {n} in {csv_path.name}")
				return None

			if len(waypoints) == 0:
				self._show_message(f"No waypoints found for drone index {n} in {csv_path.name}")
				return None

			last_waypoints.append(waypoints[-1])

		return last_waypoints

	def _load_phase_trajectories(
		self,
		csv_path: Path,
		prepend_points: list[np.ndarray] | None = None,
	) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
		try:
			df = pd.read_csv(csv_path)
		except OSError:
			self._show_message(f"Could not open {csv_path.name}")
			return None

		x_columns = sorted(
			(col for col in df.columns if col.startswith("x")),
			key=lambda name: int(name[1:]) if name[1:].isdigit() else 10**9,
		)
		n_drones = len(x_columns)
		if n_drones == 0:
			self._show_message(f"No xN,yN,zN columns found in {csv_path.name}")
			return None

		if prepend_points is not None and len(prepend_points) != n_drones:
			self._show_message("Phase CSV files contain different drone counts")
			return None

		all_xi: list[np.ndarray] = []
		all_yi: list[np.ndarray] = []
		all_zi: list[np.ndarray] = []

		for n in range(n_drones):
			try:
				waypoints = np.column_stack(
					(
						df[f"x{n}"].values,
						df[f"y{n}"].values,
						df[f"z{n}"].values,
					)
				)
				if prepend_points is not None:
					repeated_prepend = np.repeat(
						prepend_points[n][np.newaxis, :],
						STITCH_POINT_REPEAT,
						axis=0,
					)
					waypoints = np.vstack((repeated_prepend, waypoints))

				repeated_point = np.repeat(
					waypoints[-1][np.newaxis, :],
					STITCH_POINT_REPEAT,
					axis=0,
				)
				waypoints = np.vstack((waypoints, repeated_point))

				xi, yi, zi = interpolate_trajectory(
					waypoints[:, 0],
					waypoints[:, 1],
					waypoints[:, 2],
				)
			except KeyError:
				self._show_message(f"Missing axis columns for drone index {n} in {csv_path.name}")
				return None

			all_xi.append(xi)
			all_yi.append(yi)
			all_zi.append(zi)

		return all_xi, all_yi, all_zi

	def _compose_full_show(
		self,
		takeoff_xyz: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
		show_forward_xyz: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
		show_backward_xyz: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
		landing_xyz: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
		phase_durations: dict[str, float],
		num_trials: int,
		wait_after_takeoff: float = HOVER_SECONDS,
		wait_between_passes: float = HOVER_SECONDS,
		wait_before_landing: float = HOVER_SECONDS,
	) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
		n_drones = len(show_forward_xyz[0])
		combined_x: list[list[np.ndarray]] = [[] for _ in range(n_drones)]
		combined_y: list[list[np.ndarray]] = [[] for _ in range(n_drones)]
		combined_z: list[list[np.ndarray]] = [[] for _ in range(n_drones)]
		frame_times_parts: list[np.ndarray] = []
		last_time: float | None = None

		def append_phase(
			xyz: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
			seconds_per_frame: float,
			drop_first: bool,
		) -> None:
			nonlocal last_time
			n_points = len(xyz[0][0])
			if n_points == 0:
				return

			start_idx = 1 if drop_first and n_points > 1 else 0
			if start_idx >= n_points:
				return

			for idx in range(n_drones):
				combined_x[idx].append(xyz[0][idx][start_idx:])
				combined_y[idx].append(xyz[1][idx][start_idx:])
				combined_z[idx].append(xyz[2][idx][start_idx:])

			n_appended = n_points - start_idx
			if last_time is None:
				times = np.arange(n_appended, dtype=float) * seconds_per_frame
			else:
				times = last_time + (np.arange(1, n_appended + 1, dtype=float) * seconds_per_frame)
			frame_times_parts.append(times)

			last_time = float(times[-1])

		def append_hover(seconds_per_frame: float, hover_seconds: float) -> None:
			nonlocal last_time
			n_frames = int(round(hover_seconds / seconds_per_frame))
			if n_frames <= 0 or last_time is None:
				return

			for idx in range(n_drones):
				last_x = combined_x[idx][-1][-1]
				last_y = combined_y[idx][-1][-1]
				last_z = combined_z[idx][-1][-1]
				combined_x[idx].append(np.full(n_frames, last_x))
				combined_y[idx].append(np.full(n_frames, last_y))
				combined_z[idx].append(np.full(n_frames, last_z))

			times = last_time + (np.arange(1, n_frames + 1, dtype=float) * seconds_per_frame)
			frame_times_parts.append(times)

			last_time = float(times[-1])

		append_phase(takeoff_xyz, phase_durations["takeoff"], drop_first=False)
		append_hover(phase_durations["show"], wait_after_takeoff)

		append_phase(show_forward_xyz, phase_durations["show"], drop_first=True)
		for _ in range(num_trials - 1):
			append_hover(phase_durations["show"], wait_between_passes)
			append_phase(show_backward_xyz, phase_durations["show"], drop_first=True)
			append_hover(phase_durations["show"], wait_between_passes)
			append_phase(show_forward_xyz, phase_durations["show"], drop_first=True)

		append_hover(phase_durations["show"], wait_before_landing)
		append_phase(landing_xyz, phase_durations["landing"], drop_first=True)

		all_xi = [np.concatenate(parts) if parts else np.array([]) for parts in combined_x]
		all_yi = [np.concatenate(parts) if parts else np.array([]) for parts in combined_y]
		all_zi = [np.concatenate(parts) if parts else np.array([]) for parts in combined_z]
		frame_times = np.concatenate(frame_times_parts) if frame_times_parts else np.array([], dtype=float)

		return all_xi, all_yi, all_zi, frame_times

	def _build_scene(self) -> None:
		self._fig.clf()
		self._ax = self._fig.add_subplot(111, projection="3d")

		self._trails = []
		self._markers = []
		self._labels = []

		for n in range(self._n_drones):
			color = DRONE_COLORS[n % len(DRONE_COLORS)]
			self._ax.plot(
				self._all_xi[n],
				self._all_yi[n],
				self._all_zi[n],
				color=color,
				linewidth=0.8,
				alpha=0.25,
			)
			trail, = self._ax.plot([], [], [], color=color, linewidth=1.5)
			marker, = self._ax.plot(
				[],
				[],
				[],
				"o",
				color=color,
				markersize=8,
				label=f"Drone {n}",
			)
			label = self._ax.text(
				0, 0, 0,
				str(n),
				color=color,
				fontsize=8,
				visible=False,
			)
			self._trails.append(trail)
			self._markers.append(marker)
			self._labels.append(label)

		all_x = np.concatenate(self._all_xi)
		all_y = np.concatenate(self._all_yi)
		all_z = np.concatenate(self._all_zi)

		self._set_testbed_axes()
		self._ax.set_xlabel("X (m)")
		self._ax.set_ylabel("Y (m)")
		self._ax.set_zlabel("Z (m)")
		self._ax.set_title(f"Full show trajectories - {self._n_drones} drone(s)")
		self._ax.legend(ncols=2, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
		self._fig.tight_layout()

	def _set_testbed_axes(self) -> None:
		self._ax.set_xlim(TESTBED_X_MIN, TESTBED_X_MAX)
		self._ax.set_ylim(TESTBED_Y_MIN, TESTBED_Y_MAX)
		self._ax.set_zlim(TESTBED_Z_MIN, TESTBED_Z_MAX)

	def _update_frame(self) -> None:
		if self._n_frames == 0:
			self._timer.stop()
			return

		elapsed = time.perf_counter() - self._sim_start_time
		frame = int(np.searchsorted(self._frame_times, elapsed, side="right") - 1)
		if frame < 0:
			frame = 0
		frame = min(frame, self._n_frames - 1)

		if frame <= self._last_rendered_frame:
			if frame >= self._n_frames - 1:
				self._timer.stop()
			return

		for n in range(self._n_drones):
			start = max(0, frame - self._trail_len)
			self._trails[n].set_data_3d(
				self._all_xi[n][start:frame + 1],
				self._all_yi[n][start:frame + 1],
				self._all_zi[n][start:frame + 1],
			)
			x, y, z = self._all_xi[n][frame], self._all_yi[n][frame], self._all_zi[n][frame]
			self._markers[n].set_data_3d([x], [y], [z])
			self._labels[n].set_position((x, y))
			self._labels[n].set_3d_properties(z + 0.1, zdir="z")
			self._labels[n].set_visible(True)

		self._canvas.draw_idle()
		self._last_rendered_frame = frame
		self._frame_idx = frame
		if frame >= self._n_frames - 1:
			self._timer.stop()

	def _show_message(self, message: str) -> None:
		self._timer.stop()
		self._fig.clf()
		ax = self._fig.add_subplot(111)
		ax.axis("off")
		ax.text(0.5, 0.5, message, ha="center", va="center", color="gray", fontsize=12)
		self._canvas.draw_idle()
