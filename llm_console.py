#!/usr/bin/env python3
"""
LLM Console - Client side of the drone show LLM.
Writes prompts to temp/prompt.txt and waits for temp/answer.txt from llm_server.py.
Now controls 3 drones (Red, Green, Blue) based on LLM responses.
"""

import argparse
import ast
import asyncio
import time
from pathlib import Path
import numpy as np

from cflib2 import Crazyflie, LinkContext
from cflib2.toc_cache import FileTocCache
import shutil

from llm.dataset_generation import SYSTEM_PROMPT

RESET_COMMAND = "__RESET__"
POLL_INTERVAL = 0.2   # seconds between answer-file checks
TIMEOUT = 120         # seconds to wait for a response before giving up

# Drone configuration - Red, Green, Blue
DRONE_URIS = [
    "radio://0/90/2M/ABAD1DEA01",  # Red drone
    "radio://0/90/2M/ABAD1DEA02",  # Green drone
    "radio://0/90/2M/ABAD1DEA03",  # Blue drone
]
DRONE_NAMES = ["red", "green", "blue"]  # , "blue"]

TAKEOFF_HEIGHT = 0.3  # meters
TAKEOFF_DURATION = 2.0  # seconds
LOG_INTERVAL = 50  # ms

# Force field parameters
FORCE_FIELD_D_MIN = 0.5  # minimum distance for force calculation
FORCE_FIELD_D_MAX = 0.4  # maximum distance for repulsive force
K_REPULSIVE = 1.5  # repulsive force gain
K_ATTRACTIVE = 2.0  # attractive force gain for target
K_BOUNDARY = 3.0  # boundary repulsive force gain
WAYPOINT_DT = 0.1  # time step for force field integration
WAYPOINT_INTERVAL = 0.1  # seconds between waypoints
VIRTUAL_UPDATE_INTERVAL = WAYPOINT_INTERVAL / 10  # virtual position update rate (10x faster)
VIRTUAL_UPDATES_PER_GOTO = 10  # number of virtual updates per goto command
MAX_VELOCITY = 0.5  # maximum velocity (m/s)
POSITION_TOLERANCE = 0.02  # tolerance for reaching target (m)
BOUNDARY_MIN = np.array([-1.5, -1.5, 0.1])  # minimum boundary
BOUNDARY_MAX = np.array([1.5, 1.5, 2.0])  # maximum boundary


async def drain_log(log_stream, last_log: dict) -> None:
    """Continuously drain a log stream, keeping only the most recent reading."""
    while True:
        data = await log_stream.next()
        last_log["data"] = data.data


def calc_repulsive_force(p1, p2, d_min=FORCE_FIELD_D_MIN, d_max=FORCE_FIELD_D_MAX):
    """Calculate repulsive force from p2 acting on p1.
    
    Args:
        p1: Current position (numpy array)
        p2: Other drone position (numpy array)
        d_min: Minimum distance for force calculation
        d_max: Maximum distance for repulsive force
    
    Returns:
        Force vector (numpy array)
    """
    d = p1 - p2  # Direction away from p2
    dist = np.linalg.norm(d)
    
    if dist > d_max or dist < 1e-3:
        return np.zeros_like(p1)
    
    d_dist = max(d_min, dist)
    # Repulsive force: pushes away from other drone
    force_magnitude = 1.0 / (d_dist + 1e-6)
    return (d / (dist + 1e-7)) * force_magnitude


def calc_boundary_repulsive_force(p, boundary_min, boundary_max, margin=0.3):
    """Calculate repulsive force from boundaries.
    
    Args:
        p: Current position (numpy array)
        boundary_min: Minimum boundary coordinates
        boundary_max: Maximum boundary coordinates
        margin: Distance from boundary where force starts
    
    Returns:
        Force vector (numpy array)
    """
    force = np.zeros_like(p)
    
    for i in range(len(p)):
        # Distance to lower boundary
        dist_min = p[i] - boundary_min[i]
        if dist_min < margin:
            # Push away from lower boundary
            force[i] += 1.0 / (dist_min + 1e-6) ** 2
        
        # Distance to upper boundary
        dist_max = boundary_max[i] - p[i]
        if dist_max < margin:
            # Push away from upper boundary
            force[i] -= 1.0 / (dist_max + 1e-6) ** 2
    
    return force


def calc_attractive_force(p, target, max_force=1.0):
    """Calculate attractive force toward target.
    
    Args:
        p: Current position
        target: Target position
        max_force: Maximum force magnitude
    
    Returns:
        Force vector (numpy array)
    """
    d = target - p
    dist = np.linalg.norm(d)
    
    if dist < 1e-3:
        return np.zeros_like(p)
    
    # Attractive force proportional to distance, but capped
    force_magnitude = min(dist, max_force)
    return (d / dist) * force_magnitude


def compute_next_position(current_pos, target_pos, other_positions, dt=WAYPOINT_DT):
    """Compute next position using force fields.
    
    Args:
        current_pos: Current position (numpy array)
        target_pos: Target position (numpy array)
        other_positions: List of other drone positions
        dt: Time step
    
    Returns:
        Next position (numpy array)
    """
    force = np.zeros_like(current_pos)
    
    # Repulsive forces from other drones
    for other_pos in other_positions:
        if other_pos is not None and np.linalg.norm(current_pos - other_pos) > 1e-3:
            force += K_REPULSIVE * calc_repulsive_force(current_pos, other_pos)
    
    # Repulsive forces from boundaries
    force += K_BOUNDARY * calc_boundary_repulsive_force(current_pos, BOUNDARY_MIN, BOUNDARY_MAX)
    
    # Attractive force toward target
    force += K_ATTRACTIVE * calc_attractive_force(current_pos, target_pos)
    
    # Calculate velocity and new position
    velocity = force * dt
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude > MAX_VELOCITY * dt:
        velocity = velocity / velocity_magnitude * MAX_VELOCITY * dt
    
    new_pos = current_pos + velocity
    
    # Clamp to boundaries
    new_pos = np.maximum(new_pos, BOUNDARY_MIN)
    new_pos = np.minimum(new_pos, BOUNDARY_MAX)
    
    return new_pos


class DroneController:
    """Controller for managing 3 drones (Red, Green, Blue)."""
    
    def __init__(self):
        self.sessions = []
        self.ctx = None
        self.cfs = []
        
    async def connect(self):
        """Connect to all drones and initialize."""
        print(f"\nConnecting to {len(DRONE_URIS)} drones...")
        self.ctx = LinkContext()
        self.cfs = await asyncio.gather(
            *[Crazyflie.connect_from_uri(self.ctx, uri, FileTocCache('cache')) 
              for uri in DRONE_URIS]
        )
        print("All connected!")
        
        # Setup each drone
        for cf, uri, name in zip(self.cfs, DRONE_URIS, DRONE_NAMES):
            hlc = cf.high_level_commander()
            
            param = cf.param()
            # Landing control parameters
            param.set("landingCrtl.hOffset", 0.02)
            param.set("landingCrtl.hDuration", 1.0)
            param.set("ctrlMel.ki_z", 1.5)
            
            # Position PID for landing
            param.set("landingCrtl.m_pos_kp", 1.0)
            param.set("landingCrtl.m_pos_ki", 0.7)
            param.set("landingCrtl.m_pos_kd", 0.1)
            
            # Attitude PID for landing
            param.set("landingCrtl.m_att_kp", 100000.0)
            param.set("landingCrtl.m_att_ki", 0.0)
            param.set("landingCrtl.m_att_kd", 10000.0)
            
            param.set("stabilizer.controller", 2)

            
            log = cf.log()
            block = await log.create_block()
            await block.add_variable("stateEstimate.x")
            await block.add_variable("stateEstimate.y")
            await block.add_variable("stateEstimate.z")
            
            log_stream = await block.start(LOG_INTERVAL)
            last_log = {"data": (await log_stream.next()).data}
            log_task = asyncio.create_task(drain_log(log_stream, last_log))
            
            values = last_log["data"]
            pad_x = values["stateEstimate.x"]
            pad_y = values["stateEstimate.y"]
            pad_z = values["stateEstimate.z"]
            print(f"[{name}] Pad position: x={pad_x:.3f} y={pad_y:.3f} z={pad_z:.3f}")

            param = cf.param()
            param.set("colorLedBot.wrgb8888", 0x0FF0000 if name == "red" else (0x000FF00 if name == "green" else 0x0000FF))
            
            self.sessions.append({
                'cf': cf,
                'hlc': hlc,
                'name': name,
                'uri': uri,
                'last_log': last_log,
                'log_task': log_task,
                'pad_x': pad_x,
                'pad_y': pad_y,
                'pad_z': pad_z,
            })
    
    async def takeoff(self):
        """Arm and takeoff all drones."""
        print("\nArming all drones...")
        await asyncio.gather(
            *[s['cf'].platform().send_arming_request(True) for s in self.sessions]
        )
        await asyncio.sleep(1.0)
        print("All armed!")
        
        print("\nTaking off all drones...")
        await asyncio.gather(
            *[s['hlc'].take_off(TAKEOFF_HEIGHT, None, TAKEOFF_DURATION, None) 
              for s in self.sessions]
        )
        await asyncio.sleep(TAKEOFF_DURATION + 1.0)
        print("All drones in the air!")
        
        # Initialize virtual setpoint positions for each drone
        print("\nInitializing virtual setpoint positions...")
        for session in self.sessions:
            values = session['last_log']['data']
            virtual_pos = np.array([
                values['stateEstimate.x'],
                values['stateEstimate.y'],
                values['stateEstimate.z']
            ], dtype=float)
            session['virtual_position'] = virtual_pos
            print(f"[{session['name']}] Virtual position: [{virtual_pos[0]:.2f}, {virtual_pos[1]:.2f}, {virtual_pos[2]:.2f}]")
    
    async def safe_goto(self, targets, max_time=30.0, tolerance=POSITION_TOLERANCE):
        """Navigate drones to target positions using force field navigation.
        
        Args:
            targets: List of np.array target positions, or None to skip a drone.
            max_time: Maximum navigation time in seconds.
        """
        active_indices = [i for i, t in enumerate(targets) if t is not None]
        if not active_indices:
            return

        max_iterations = int(max_time / VIRTUAL_UPDATE_INTERVAL)
        iteration = 0

        while iteration < max_iterations:
            virtual_positions = [s['virtual_position'] for s in self.sessions]
            all_reached = True

            next_positions = []
            for i in range(len(self.sessions)):
                if i in active_indices:
                    other_positions = [virtual_positions[j] for j in range(len(self.sessions)) if j != i]
                    next_pos = compute_next_position(
                        virtual_positions[i],
                        targets[i],
                        other_positions,
                        VIRTUAL_UPDATE_INTERVAL
                    )
                    next_positions.append(next_pos)
                    if np.linalg.norm(targets[i] - virtual_positions[i]) > POSITION_TOLERANCE:
                        all_reached = False
                else:
                    next_positions.append(virtual_positions[i])

            if all_reached and iteration > VIRTUAL_UPDATES_PER_GOTO * 2:
                print("All drones reached their targets!")
                break

            for i in range(len(self.sessions)):
                virtual_positions[i] = next_positions[i]
                self.sessions[i]['virtual_position'] = next_positions[i]

            if iteration % VIRTUAL_UPDATES_PER_GOTO == 0:
                tasks = [
                    self.sessions[i]['hlc'].go_to(
                        *next_positions[i], 0.0, WAYPOINT_INTERVAL,
                        relative=False, linear=True, group_mask=None
                    )
                    for i in active_indices
                ]
                if tasks:
                    await asyncio.gather(*tasks)
                await asyncio.sleep(VIRTUAL_UPDATE_INTERVAL * VIRTUAL_UPDATES_PER_GOTO)

            iteration += 1

        if iteration >= max_iterations:
            print(f"Warning: Reached maximum iterations ({max_iterations})")

        print("\nFinal positions:")
        for i in active_indices:
            session = self.sessions[i]
            virtual_pos = virtual_positions[i]
            dist = np.linalg.norm(targets[i] - virtual_pos)
            values = session['last_log']['data']
            actual_pos = np.array([
                values['stateEstimate.x'],
                values['stateEstimate.y'],
                values['stateEstimate.z']
            ], dtype=float)
            print(f"[{session['name']}] Virtual: [{virtual_pos[0]:.2f}, {virtual_pos[1]:.2f}, {virtual_pos[2]:.2f}], Actual: [{actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}], Target: {targets[i][0]:.2f}, {targets[i][1]:.2f}, {targets[i][2]:.2f}  Distance to target: {dist:.3f}m")

    async def goto_positions(self, positions):
        """Send drones to target positions using force field navigation.
        
        Args:
            positions: List of (x, y, z) tuples or None, one per drone (Red, Green, Blue).
        """
        targets = [np.array(pos, dtype=float) if pos is not None else None for pos in positions]
        print("\nStarting force field navigation...")
        await self.safe_goto(targets, max_time=30.0)

    
    async def land(self):
        """Land all drones - first return to pad positions using force fields, then land."""
        print("\nReturning to landing pads...")
        targets = [np.array([s['pad_x'], s['pad_y'], s['pad_z'] + 0.4], dtype=float) for s in self.sessions]
        await self.safe_goto(targets, max_time=10.0, tolerance=0.005)

        await asyncio.sleep(1.0)
        
        print("\nLanding all drones..." + str(self.sessions[0]["pad_z"]))
        await asyncio.gather(
            *[s['hlc'].land(s['pad_z'], None, 5.0, None) for s in self.sessions]
        )
        await asyncio.sleep(5.0)
        
        print("Disarming all drones...")
        await asyncio.gather(
            *[s['cf'].platform().send_arming_request(False) for s in self.sessions]
        )
        await asyncio.sleep(1.0)
    
    async def disconnect(self):
        """Stop and disconnect from all drones."""
        if self.sessions:
            await asyncio.gather(*[s['hlc'].stop(None) for s in self.sessions])
            for s in self.sessions:
                s['log_task'].cancel()
        
        if self.cfs:
            print("Disconnecting...")
            await asyncio.gather(*[cf.disconnect() for cf in self.cfs])
            print("Disconnected!")


def send_and_receive(prompt_file: Path, answer_file: Path, text: str) -> str:
    """Write text to prompt_file and block until answer_file appears."""

    success = False
    # Write to local file first, then copy to remote temp directory
    while not success:
        try:
            local_file = Path("prompt.txt")
            local_file.write_text(text)
            shutil.copy(local_file, "temp/")
            success = True
        except Exception as e:
            print(f"Error writing prompt file: {e}. Retrying...")
            time.sleep(0.1)

    if answer_file.exists():
        answer_file.unlink()

    deadline = time.monotonic() + TIMEOUT
    while not answer_file.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"No response from server after {TIMEOUT}s.")
        time.sleep(POLL_INTERVAL)

    answer = answer_file.read_text()
    return answer


def parse_positions(llm_response: str):
    """Parse target positions from LLM response for Red, Green, Blue drones.
    
    Expected format: Python dict string like:
    {'Red': {'x': 1.0, 'y': 2.0, 'z': 1.5}, 'Green': {...}, 'Blue': {...}}
    or
    {'Red': [1.0, 2.0, 1.5], 'Green': [...], 'Blue': [...]}
    
    Returns:
        List of (x, y, z) tuples or None for each drone [Red, Green, Blue]
    """
    positions = [None, None, None]
    
    try:
        # Try to parse as Python dict
        data = ast.literal_eval(llm_response.strip())
        print(f"Parsed LLM response as dict: {data}, {llm_response}")
        if not isinstance(data, dict):
            print(f"Warning: LLM response is not a dict: {type(data)}")
            return positions
        
        for i, name in enumerate(DRONE_NAMES):
            if name in data:
                pos_data = data[name]   
                
                # Handle dict format: {'x': 1.0, 'y': 2.0, 'z': 1.5}
                if isinstance(pos_data, dict):
                    if 'x' in pos_data and 'y' in pos_data and 'z' in pos_data:
                        positions[i] = (
                            float(pos_data['x']),
                            float(pos_data['y']),
                            float(pos_data['z'])
                        )
                # Handle list/tuple format: [1.0, 2.0, 1.5]
                elif isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                    positions[i] = (
                        float(pos_data[0]),
                        float(pos_data[1]),
                        float(pos_data[2])
                    )
    
    except (ValueError, SyntaxError, KeyError, TypeError, IndexError) as e:
        print(f"Warning: Could not parse LLM response as dict: {e}")
        print(f"Response was: {llm_response[:200]}")
    
    return positions


async def main_async():
    parser = argparse.ArgumentParser(description="LLM console client with drone control")
    parser.add_argument("--temp-dir", type=str, default="temp", help="Directory shared with the server")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    prompt_file = temp_dir / "prompt.txt"
    answer_file = temp_dir / "answer.txt"

    print("LLM Console with Drone Control (Red, Green, Blue)")
    print(f"Communicating via '{temp_dir}/' — make sure llm_server.py is running.")
    print("Commands: 'start' to connect/takeoff, 'land' to land, 'exit'/'quit' to stop.\n")

    drone_controller = DroneController()
    started = False

    try:
        while True:
            try:
                # Run input in executor to not block asyncio event loop
                loop = asyncio.get_event_loop()
                prompt = await loop.run_in_executor(None, lambda: input(">>> ").strip())
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not prompt:
                continue

            if prompt.lower() in ("exit", "quit"):
                print("Exiting.")
                break
            
            if prompt.lower() == "start":
                if started:
                    print("Drones already started!")
                    continue
                await drone_controller.connect()
                await drone_controller.takeoff()
                started = True
                print("\nDrones ready! Send prompts to the LLM to control them.\n")
                continue
            
            if prompt.lower() == "land":
                if not started:
                    print("Drones not started yet!")
                    continue
                await drone_controller.land()
                print("Drones landed!\n")
                continue

            if prompt.lower() == "reset":
                print("Resetting server chat history...")
                answer = await loop.run_in_executor(
                    None, send_and_receive, prompt_file, answer_file, RESET_COMMAND
                )
                print("Server chat history reset.\n" if answer == "__OK__" 
                      else f"Server replied: {answer}\n")
                continue

            if not started:
                print("Please type 'start' first to initialize the drones!")
                continue

            print("Waiting for response from LLM server...")
            try:
                answer = await loop.run_in_executor(
                    None, send_and_receive, prompt_file, answer_file, prompt
                )
            except TimeoutError as e:
                print(f"[client] {e}\n")
                continue

            print("\n--- LLM Response ---")
            print(answer)
            print("--------------------\n")
            
            # Parse positions from LLM response
            positions = parse_positions(answer)
            
            if any(pos is not None for pos in positions):
                print("Extracted positions:")
                for name, pos in zip(DRONE_NAMES, positions):
                    if pos:
                        print(f"  {name}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                    else:
                        print(f"  {name}: No position found")
                
                print("\nSending drones to target positions...")
                await drone_controller.goto_positions(positions)
                print("Drones reached target positions!\n")
            else:
                print("No drone positions found in LLM response.\n")

    finally:
        if started:
            await drone_controller.disconnect()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
