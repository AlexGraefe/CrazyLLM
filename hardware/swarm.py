"""Swarm hardware controller with an async state machine.

The :class:`Swarm` class owns all Crazyflie connection and flight logic.
It reports progress back to whatever object implements :class:`SwarmGUI`
(normally :class:`gui.gui.MainWindow`).
"""

import asyncio
from enum import Enum, auto
from typing import Any, Protocol

import numpy as np

from cflib2 import Crazyflie, LinkContext
from cflib2.toc_cache import FileTocCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 2.0
LOG_INTERVAL = 100  # ms
STAGGER_STRIDE = 5   # launch/land every Nth drone per round (round 0: idx 0,4,8…; round 1: 1,5,9…)
STAGGER_DELAY  = TAKEOFF_DURATION + 0.5  # seconds between stagger groups
LED_COLORS = [0x00FF0000, 0x0000FF00, 0x000000FF]  # Red, Green, Blue (WRGB8888, cycles for >3 drones)

# -- Force-field navigation constants ----------------------------------------
FF_LOG_INTERVAL             = 50     # ms – position-logger rate during goto
FF_D_MIN                    = 0.5    # minimum effective distance for repulsion
FF_D_MAX                    = 0.4    # distance threshold for repulsive force onset
FF_K_REPULSIVE              = 1.5    # repulsive force gain
FF_K_ATTRACTIVE             = 2.0    # attractive force gain
FF_K_BOUNDARY               = 3.0    # boundary repulsive force gain
FF_WAYPOINT_INTERVAL        = 0.1    # seconds between go_to commands
FF_VIRTUAL_UPDATES_PER_GOTO = 10     # force-field steps per go_to command
FF_VIRTUAL_UPDATE_INTERVAL  = FF_WAYPOINT_INTERVAL / FF_VIRTUAL_UPDATES_PER_GOTO
FF_MAX_VELOCITY             = 0.5    # m/s cap on the virtual velocity
FF_POSITION_TOLERANCE       = 0.05   # m – considered "reached"
FF_BOUNDARY_MIN             = np.array([-1.5, -1.5, 0.1])
FF_BOUNDARY_MAX             = np.array([ 1.5,  1.5, 2.0])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SwarmState(Enum):
    UNCONNECTED = auto()
    CONNECTED   = auto()
    FLYING      = auto()
    LANDED      = auto()
    ERROR       = auto()


# ---------------------------------------------------------------------------
# GUI callback protocol
# ---------------------------------------------------------------------------

class SwarmGUI(Protocol):
    """Callbacks the :class:`Swarm` uses to feed information back to the GUI.

    The GUI object passed to :class:`Swarm.__init__` must implement all of
    these methods (duck-typed via :class:`typing.Protocol`).
    """

    def on_swarm_state_changed(self, state: SwarmState) -> None:
        """Called whenever the swarm transitions to a new state."""
        ...

    def on_live_positions(
        self,
        positions: list,
    ) -> None:
        """Called with the latest drone positions during flight.

        *positions* is a ``list[tuple[float, float, float] | None]``, one
        element per drone.  ``None`` means no data has arrived yet for that
        drone.
        """
        ...

    def on_live_mode_started(self, n_drones: int) -> None:
        """Called just before the live-flight visualisation begins."""
        ...

    def on_live_mode_stopped(self) -> None:
        """Called after the live-flight visualisation has ended."""
        ...

    def on_fly_enabled(self, enabled: bool) -> None:
        """Tell the GUI whether the *Fly* button should be enabled."""
        ...

    def on_error(self, title: str, message: str) -> None:
        """Display an error to the user.  May be called from any async context."""
        ...


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------

class Swarm:
    """Async state machine that controls a Crazyflie swarm.

    Usage::

        swarm = Swarm(gui)
        await swarm.connect("radio://0/80/2M/E7E7E7E7", 3)
        await swarm.fly(takeoff_csv, active_csv, landing_csv, dt_start, dt_show, trials)
        await swarm.emergency_land()
        await swarm.disconnect()
    """

    def __init__(self, gui: SwarmGUI) -> None:
        self._gui = gui
        self._state = SwarmState.UNCONNECTED
        self._connected_cfs: list[object] = []
        self._link_context: object | None = None
        self._live_log_streams: list[object] = []
        self._latest_positions: list[dict[str, Any]] = []
        self._connect_task: asyncio.Task | None = None
        self._fly_task: asyncio.Task | None = None
        self._goto_task: asyncio.Task | None = None
        self._pad_positions: list[tuple[float, float, float]] = []
        self._virtual_positions: list[np.ndarray] = []

    # -- State ---------------------------------------------------------------

    @property
    def state(self) -> SwarmState:
        return self._state

    def _set_state(self, state: SwarmState) -> None:
        print(f"[Swarm] {self._state.name} -> {state.name}")
        self._state = state
        self._gui.on_swarm_state_changed(state)

    # -- Public API ----------------------------------------------------------

    def connect(self, base_address: str, num_drones: int) -> None:
        """Start connecting to *num_drones* drones derived from *base_address*.

        Returns immediately; progress is reported via :class:`SwarmGUI` callbacks.
        Ignored if a connection attempt is already in progress.
        """
        if self._connect_task is not None and not self._connect_task.done():
            return
        self._connect_task = asyncio.create_task(
            self._connect_impl(base_address, num_drones)
        )
        self._connect_task.add_done_callback(self._on_connect_task_done)

    async def _connect_impl(self, base_address: str, num_drones: int) -> None:
        """Internal coroutine that performs the actual connection sequence."""
        if self._connected_cfs:
            await self._disconnect_all()

        self._link_context = LinkContext()
        uris = [f"{base_address}{index:02X}" for index in range(1, num_drones + 1)]

        try:
            self._connected_cfs = list(
                await asyncio.gather(
                    *[
                        Crazyflie.connect_from_uri(
                            self._link_context,
                            uri,
                            FileTocCache("cache"),
                        )
                        for uri in uris
                    ]
                )
            )
        except Exception as exc:
            await self._disconnect_all()
            self._set_state(SwarmState.ERROR)
            self._gui.on_fly_enabled(False)
            self._gui.on_error("Connection Failed", f"Could not connect to all drones: {exc}")
            return

        print(f"Connected to {len(self._connected_cfs)} drone(s): {', '.join(uris)}")
        self._gui.on_fly_enabled(True)
        self._set_state(SwarmState.CONNECTED)

    async def disconnect(self) -> None:
        """Disconnect all drones.

        Cancels any in-progress fly sequence first.
        Transitions: * → UNCONNECTED.
        """
        if self._fly_task is not None and not self._fly_task.done():
            self._fly_task.cancel()
            await asyncio.gather(self._fly_task, return_exceptions=True)
            self._fly_task = None
        await self._disconnect_all()
        self._set_state(SwarmState.UNCONNECTED)

    async def emergency_land(self) -> None:
        """Immediately land all connected drones regardless of current state.

        Cancels any in-progress fly sequence first.
        Transitions: * → CONNECTED (so the operator can reconnect and retry).
        """
        if self._fly_task is not None and not self._fly_task.done():
            self._fly_task.cancel()
            await asyncio.gather(self._fly_task, return_exceptions=True)
            self._fly_task = None
        if not self._connected_cfs:
            return

        self._set_state(SwarmState.LANDED)
        print("Emergency land: commanding all drones to land...")
        try:
            await asyncio.gather(
                *[
                    cf.high_level_commander().land(0.0, None, 2.0, None)
                    for cf in self._connected_cfs
                ],
                return_exceptions=True,
            )
            await asyncio.sleep(3.0)
            await asyncio.gather(
                *[cf.high_level_commander().stop(None) for cf in self._connected_cfs],
                return_exceptions=True,
            )
            await asyncio.gather(
                *[cf.platform().send_arming_request(False) for cf in self._connected_cfs],
                return_exceptions=True,
            )
        except Exception as exc:
            print(f"Emergency land error: {exc}")
        finally:
            await self._stop_live_position_logging()
            self._gui.on_live_mode_stopped()
            self._set_state(SwarmState.CONNECTED)

    def land(self) -> None:
        """Navigate drones to their pad positions + 1 m, then land.

        Returns immediately; ignored if not currently flying.
        Cancels any in-progress goto before starting.
        """
        if self._state != SwarmState.FLYING:
            self._gui.on_error("Not Flying", "Drones must be airborne before landing.")
            return
        if self._goto_task is not None and not self._goto_task.done():
            self._goto_task.cancel()
        if self._fly_task is not None and not self._fly_task.done():
            return
        self._fly_task = asyncio.create_task(self._land_impl())
        self._fly_task.add_done_callback(self._on_fly_task_done)

    async def _land_impl(self) -> None:
        """Go to pad position + 1 m, then land all drones staggered."""
        try:
            if self._pad_positions:
                above_pad = [
                    (x, y, z + 1.0) for x, y, z in self._pad_positions
                ]
                await self._goto_impl(above_pad)

            print("Landing drones...")
            await asyncio.gather(
                *[
                    cf.high_level_commander().land(0.0, None, 2.0, None)
                    for cf in self._connected_cfs
                ]
            )
            await self._sleep_with_live_updates(2.0 + 0.5)

            await asyncio.sleep(1.0)
            await asyncio.gather(
                *[cf.high_level_commander().stop(None) for cf in self._connected_cfs],
                return_exceptions=True,
            )
            await asyncio.gather(
                *[cf.platform().send_arming_request(False) for cf in self._connected_cfs],
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set_state(SwarmState.ERROR)
            self._gui.on_error("Landing Failed", f"Landing sequence failed: {exc}")
            return
        finally:
            await self._stop_live_position_logging()
            self._gui.on_live_mode_stopped()
        self._set_state(SwarmState.CONNECTED)

    def takeoff(self) -> None:
        """Arm all connected drones and take off to hover height.

        Returns immediately; ignored if already flying or no drones connected.
        """
        if not self._connected_cfs:
            self._gui.on_error("Not Connected", "Connect to drones before taking off.")
            return
        if self._fly_task is not None and not self._fly_task.done():
            return
        self._fly_task = asyncio.create_task(self._takeoff_impl())
        self._fly_task.add_done_callback(self._on_fly_task_done)

    async def _takeoff_impl(self) -> None:
        """Internal coroutine that arms drones and lifts them to hover height."""
        self._set_state(SwarmState.FLYING)
        self._gui.on_fly_enabled(False)
        self._gui.on_live_mode_started(len(self._connected_cfs))
        try:
            print("Applying controller parameters for takeoff...")
            for i, cf in enumerate(self._connected_cfs):
                param = cf.param()
                param.set("colorLedBot.wrgb8888", LED_COLORS[i % len(LED_COLORS)])
                param.set("stabilizer.controller", 1)

            print("Reading pad positions...")
            self._pad_positions = list(
                await asyncio.gather(
                    *[self._read_pad_position(cf) for cf in self._connected_cfs]
                )
            )

            print("Starting live position logger streams...")
            await self._start_live_position_logging()

            print("Arming drones...")
            await asyncio.gather(
                *[cf.platform().send_arming_request(True) for cf in self._connected_cfs]
            )
            await self._sleep_with_live_updates(1.0)

            print("Taking off...")
            await asyncio.gather(
                *[
                    cf.high_level_commander().take_off(
                        TAKEOFF_HEIGHT, None, TAKEOFF_DURATION, None
                    )
                    for cf in self._connected_cfs
                ]
            )
            await self._sleep_with_live_updates(TAKEOFF_DURATION + 0.5)

            # Initialise virtual setpoints from measured positions after takeoff.
            self._virtual_positions = []
            for pos_data in self._latest_positions:
                data = pos_data.get("data")
                if data:
                    self._virtual_positions.append(np.array([
                        float(data["stateEstimate.x"]),
                        float(data["stateEstimate.y"]),
                        float(data["stateEstimate.z"]),
                    ]))
                else:
                    self._virtual_positions.append(np.array([0.0, 0.0, TAKEOFF_HEIGHT]))

            print("Hovering...")
            self._set_state(SwarmState.FLYING)
        except Exception as exc:
            self._set_state(SwarmState.ERROR)
            self._gui.on_error("Takeoff Failed", f"Takeoff sequence failed: {exc}")
            await self._stop_live_position_logging()
            self._gui.on_live_mode_stopped()
            self._gui.on_fly_enabled(True)

    # -- Private helpers -----------------------------------------------------

    def _on_connect_task_done(self, task: asyncio.Task) -> None:
        self._connect_task = None
        if task.cancelled():
            return
        _ = task.exception()

    def _on_fly_task_done(self, task: asyncio.Task) -> None:
        self._fly_task = None
        if task.cancelled():
            return
        _ = task.exception()

    def goto_positions(self, positions: list) -> None:
        """Navigate drones toward *positions* using force-field collision avoidance.

        *positions* is a list of ``(x, y, z)`` tuples (or ``None`` to leave a
        drone stationary), one entry per connected drone.  Returns immediately;
        progress is reported via :meth:`SwarmGUI.on_live_positions` callbacks.
        If a previous goto is still running it is cancelled before the new one
        starts.
        """
        if self._state != SwarmState.FLYING:
            self._gui.on_error(
                "Not Flying", "Drones must be airborne before sending goto commands."
            )
            return
        if not self._virtual_positions:
            self._gui.on_error(
                "Not Ready", "Complete takeoff before sending goto commands."
            )
            return
        if self._goto_task is not None and not self._goto_task.done():
            self._goto_task.cancel()
        self._goto_task = asyncio.create_task(self._goto_impl(positions))
        self._goto_task.add_done_callback(self._on_goto_task_done)

    def _on_goto_task_done(self, task: asyncio.Task) -> None:
        self._goto_task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"[Swarm] goto error: {exc}")

    async def _goto_impl(self, positions: list) -> None:
        """Internal coroutine – force-field navigation to target positions."""
        n = len(self._connected_cfs)
        # Build target array aligned to drone count.
        targets: list = []
        for i in range(n):
            pos = positions[i] if i < len(positions) else None
            if pos is not None:
                targets.append(np.array(pos, dtype=float))
            else:
                targets.append(None)

        active_indices = [i for i, t in enumerate(targets) if t is not None]
        if not active_indices:
            return

        max_iterations = int(30.0 / FF_VIRTUAL_UPDATE_INTERVAL)
        steps_since_goto = 0

        for iteration in range(max_iterations):
            next_positions = list(self._virtual_positions)
            all_reached = True

            for i in active_indices:
                others = [self._virtual_positions[j] for j in range(n) if j != i]
                nxt = Swarm._ff_next_position(self._virtual_positions[i], targets[i], others)
                next_positions[i] = nxt
                if np.linalg.norm(targets[i] - self._virtual_positions[i]) > FF_POSITION_TOLERANCE:
                    all_reached = False

            for i in range(n):
                self._virtual_positions[i] = next_positions[i]

            steps_since_goto += 1
            if steps_since_goto >= FF_VIRTUAL_UPDATES_PER_GOTO:
                steps_since_goto = 0
                await asyncio.gather(
                    *[
                        self._connected_cfs[i].high_level_commander().go_to(
                            float(next_positions[i][0]),
                            float(next_positions[i][1]),
                            float(next_positions[i][2]),
                            0.0,
                            FF_WAYPOINT_INTERVAL,
                            relative=False,
                            linear=True,
                            group_mask=None,
                        )
                        for i in active_indices
                    ]
                )
                await self._sleep_with_live_updates(FF_WAYPOINT_INTERVAL)

            if all_reached and iteration > FF_VIRTUAL_UPDATES_PER_GOTO * 2:
                print("[Swarm] All drones reached their targets.")
                break

    @staticmethod
    def _stagger_groups(n: int, stride: int = STAGGER_STRIDE) -> list:
        """Return groups of drone indices for staggered launch/land.

        Round 0: [0, stride, 2*stride, ...]
        Round 1: [1, 1+stride, 1+2*stride, ...]
        ...
        """
        groups = []
        for offset in range(stride):
            group = list(range(offset, n, stride))
            if group:
                groups.append(group)
        return groups

    # -- Force-field helpers -------------------------------------------------

    @staticmethod
    def _ff_repulsive(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        d = p1 - p2
        dist = float(np.linalg.norm(d))
        if dist > FF_D_MAX or dist < 1e-3:
            return np.zeros(3)
        d_eff = max(FF_D_MIN, dist)
        return (d / (dist + 1e-7)) * (1.0 / (d_eff + 1e-6))

    @staticmethod
    def _ff_boundary_repulsive(p: np.ndarray, margin: float = 0.3) -> np.ndarray:
        force = np.zeros(3)
        for i in range(3):
            dist_min = float(p[i]) - float(FF_BOUNDARY_MIN[i])
            if dist_min < margin:
                force[i] += 1.0 / (dist_min + 1e-6) ** 2
            dist_max = float(FF_BOUNDARY_MAX[i]) - float(p[i])
            if dist_max < margin:
                force[i] -= 1.0 / (dist_max + 1e-6) ** 2
        return force

    @staticmethod
    def _ff_attractive(p: np.ndarray, target: np.ndarray, max_force: float = 1.0) -> np.ndarray:
        d = target - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-3:
            return np.zeros(3)
        return (d / dist) * min(dist, max_force)

    @staticmethod
    def _ff_next_position(
        current: np.ndarray,
        target: np.ndarray,
        others: list,
    ) -> np.ndarray:
        force = np.zeros(3)
        for other in others:
            if other is not None and np.linalg.norm(current - other) > 1e-3:
                force += FF_K_REPULSIVE * Swarm._ff_repulsive(current, other)
        force += FF_K_BOUNDARY * Swarm._ff_boundary_repulsive(current)
        force += FF_K_ATTRACTIVE * Swarm._ff_attractive(current, target)
        velocity = force * FF_VIRTUAL_UPDATE_INTERVAL
        mag = float(np.linalg.norm(velocity))
        if mag > FF_MAX_VELOCITY * FF_VIRTUAL_UPDATE_INTERVAL:
            velocity = velocity / mag * FF_MAX_VELOCITY * FF_VIRTUAL_UPDATE_INTERVAL
        new_pos = current + velocity
        new_pos = np.maximum(new_pos, FF_BOUNDARY_MIN)
        new_pos = np.minimum(new_pos, FF_BOUNDARY_MAX)
        return new_pos

    async def _disconnect_all(self) -> None:
        if not self._connected_cfs:
            return
        for cf in self._connected_cfs:
            param = cf.param()
            param.set("colorLedBot.wrgb8888", 0x00000000)
        await asyncio.gather(
            *[cf.disconnect() for cf in self._connected_cfs],
            return_exceptions=True,
        )
        self._connected_cfs = []
        self._link_context = None
        self._gui.on_fly_enabled(False)

    @staticmethod
    async def _read_pad_position(cf: object) -> tuple[float, float, float]:
        log = cf.log()
        block = await log.create_block()
        await block.add_variable("stateEstimate.x")
        await block.add_variable("stateEstimate.y")
        await block.add_variable("stateEstimate.z")
        log_stream = await block.start(LOG_INTERVAL)
        try:
            values = (await log_stream.next()).data
            return (
                float(values["stateEstimate.x"]),
                float(values["stateEstimate.y"]),
                float(values["stateEstimate.z"]),
            )
        finally:
            await log_stream.stop()

    async def _start_live_position_logging(self) -> None:
        self._latest_positions = [{"data": None} for _ in self._connected_cfs]
        self._live_log_streams = []

        for cf in self._connected_cfs:
            log = cf.log()
            block = await log.create_block()
            await block.add_variable("stateEstimate.x")
            await block.add_variable("stateEstimate.y")
            await block.add_variable("stateEstimate.z")
            log_stream = await block.start(LOG_INTERVAL)
            self._live_log_streams.append(log_stream)

    async def _stop_live_position_logging(self) -> None:
        for stream in self._live_log_streams:
            stop = getattr(stream, "stop", None)
            if callable(stop):
                result = stop()
                if asyncio.iscoroutine(result):
                    await result
        self._live_log_streams = []
        self._latest_positions = []

    async def _poll_live_positions_once(self, max_wait: float = 0.12) -> None:
        if not self._live_log_streams:
            return

        samples = await asyncio.gather(
            *[
                asyncio.wait_for(stream.next(), timeout=max_wait)
                for stream in self._live_log_streams
            ],
            return_exceptions=True,
        )

        for idx, sample in enumerate(samples):
            if isinstance(sample, Exception):
                continue
            self._latest_positions[idx]["data"] = sample.data

        positions: list[tuple[float, float, float] | None] = []
        for last in self._latest_positions:
            data = last.get("data")
            if not data:
                positions.append(None)
                continue
            positions.append(
                (
                    float(data["stateEstimate.x"]),
                    float(data["stateEstimate.y"]),
                    float(data["stateEstimate.z"]),
                )
            )
        self._gui.on_live_positions(positions)

    async def _sleep_with_live_updates(self, duration: float) -> None:
        if duration <= 0:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            step = min(0.12, remaining)
            if self._live_log_streams:
                await self._poll_live_positions_once(max_wait=step)
            else:
                await asyncio.sleep(step)
