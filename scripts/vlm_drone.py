#!/usr/bin/env python3
"""
VLM Drone Controller

Combines VLM console with drone control. Reads the latest image from the
captures folder, sends it to vlm_server.py via file-based IPC, parses the
response, and commands the drone via relative goto moves.

Make sure vlm_server.py is running before starting this script.
"""

import asyncio
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.single_drone import SingleDrone

# ---------------------------------------------------------------------------
# Global settings — tune these without touching the logic below
# ---------------------------------------------------------------------------
DRONE_URI = "radio://0/84/2M/D91F700101"

TAKEOFF_HEIGHT = 0.5        # meters
TAKEOFF_DURATION = 3.0      # seconds

STEP_SIZE = 0.3             # meters per forward / backward / left / right step
YAW_STEP = 30.0             # degrees per turn-left / turn-right step
STEP_DURATION = 1.5         # seconds allocated for each goto movement

IMAGES_FOLDER = "/home/alex/Documents/bitcraze/Owl_Q/owl-streamer/captures"
TEMP_DIR = "temp"
VLM_TIMEOUT = 60.0          # seconds to wait for a VLM answer
LOOP_DELAY = 3.0            # seconds between processing iterations

USER_PROMPT = """
    "You goal is to search for a human. If you do not see a human in the image, turn left. If you see a human but it is not in the center of the image, turn towards it. If the human is in the center of the image, fly forward. Do not forget that the action you choose has to be written at the end of your answer separated by a |."""
# ---------------------------------------------------------------------------


def parse_image_filename(filename: str):
    """Parse frame_<num>_<YYYYMMDD>_<HHMMSS>_<ms>.jpg -> (num, datetime) or None."""
    pattern = r'frame_(\d+)_(\d{8})_(\d{6})_(\d{3})\.jpg'
    match = re.match(pattern, filename)
    if not match:
        return None
    image_number = int(match.group(1))
    try:
        dt = datetime.strptime(f"{match.group(2)}_{match.group(3)}", "%Y%m%d_%H%M%S")
        return (image_number, dt)
    except ValueError:
        return None


def get_latest_image(images_folder: str = IMAGES_FOLDER):
    """Return the path to the most recent image in the folder, or None."""
    images_path = Path(images_folder)
    if not images_path.exists():
        print(f"Error: Images folder '{images_folder}' does not exist.")
        return None

    parsed = []
    for img_file in images_path.glob("frame_*.jpg"):
        result = parse_image_filename(img_file.name)
        if result:
            num, dt = result
            parsed.append((img_file, num, dt))

    if not parsed:
        return None

    # Sort by datetime then image number, descending — newest first
    parsed.sort(key=lambda x: (x[2], x[1]), reverse=True)
    latest = parsed[0][0]
    print(f"Found latest image: {latest.name}")
    return str(latest)


def parse_command(answer: str):
    """Extract the command token from 'explanation | command' VLM answer."""
    if '|' not in answer:
        return None
    return answer.split('|')[-1].strip().lower()


async def wait_for_answer(answer_file: Path, timeout: float = VLM_TIMEOUT):
    """Async-poll until the answer file appears, then return its contents."""
    elapsed = 0.0
    interval = 0.1
    while elapsed < timeout:
        if answer_file.exists():
            await asyncio.sleep(0.1)  # brief settle delay
            try:
                return answer_file.read_text()
            except Exception as e:
                print(f"Error reading answer file: {e}")
                return None
        await asyncio.sleep(interval)
        elapsed += interval
    print("Timeout waiting for VLM response.")
    return None


async def execute_command(drone: SingleDrone, command: str) -> None:
    """Map a VLM command string to a relative drone goto call.

    Crazyflie coordinate frame (relative):
        x  — forward / backward
        y  — left / right
        yaw — counter-clockwise positive
    """
    if command is None or command == "nothing":
        print(f"[drone] Command '{command}': hovering in place.")
        return

    if command == "forward":
        dx, dy, dyaw = STEP_SIZE, 0.0, 0.0
    elif command == "backward":
        dx, dy, dyaw = -STEP_SIZE, 0.0, 0.0
    elif command == "left":
        dx, dy, dyaw = 0.0, STEP_SIZE, 0.0
    elif command == "right":
        dx, dy, dyaw = 0.0, -STEP_SIZE, 0.0
    elif command == "turn left":
        dx, dy, dyaw = 0.0, 0.0, -YAW_STEP
    elif command == "turn right":
        dx, dy, dyaw = 0.0, 0.0, YAW_STEP
    else:
        print(f"[drone] Unknown command '{command}', skipping.")
        return

    print(f"[drone] Executing '{command}': dx={dx} m, dy={dy} m, dyaw={dyaw} deg")
    await drone.goto(
        x=dx,
        y=dy,
        z=0.0,
        yaw=dyaw * (3.14159 / 180.0),  # convert to radians
        duration=STEP_DURATION,
        relative=True,
    )


async def main():
    print("VLM Drone Controller")
    print("=" * 60)
    print(f"Drone URI : {DRONE_URI}")
    print(f"Prompt    : {USER_PROMPT}")
    print(f"Step size : {STEP_SIZE} m  |  Yaw step: {YAW_STEP} deg  |  Duration: {STEP_DURATION} s")
    print("Make sure vlm_server.py is running!")
    print("=" * 60)

    temp_dir = Path(TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)
    prompt_file = temp_dir / "prompt.txt"
    image_file = temp_dir / "image.jpg"
    answer_file = temp_dir / "answer.txt"

    drone = SingleDrone()

    try:
        print(f"\nConnecting to drone at {DRONE_URI}...")
        await drone.connect(DRONE_URI)

        print(f"Taking off to {TAKEOFF_HEIGHT} m...")
        await drone.takeoff(height=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
        print("Drone airborne. Starting VLM control loop.\n")

        last_processed_image = None
        iteration = 0

        while True:
            iteration += 1
            print(f"\n[Iteration {iteration}] Looking for latest image...")

            latest_image_path = get_latest_image(IMAGES_FOLDER)

            if not latest_image_path:
                print("No image found. Waiting...")
                await asyncio.sleep(2.0)
                continue

            if latest_image_path == last_processed_image:
                print("Same image as last iteration. Waiting for a new one...")
                await asyncio.sleep(2.0)
                continue

            try:
                image = Image.open(latest_image_path)
                print(f"New image: {Path(latest_image_path).name} (size: {image.size})")
            except Exception as e:
                print(f"Error loading image: {e}")
                await asyncio.sleep(2.0)
                continue

            # Send prompt + image to vlm_server.py via file IPC
            print("Sending to VLM server...")
            success = False
            while not success:
                try:
                    prompt_file.write_text(USER_PROMPT)
                    shutil.copy(latest_image_path, str(image_file))
                    success = True
                except Exception as e:
                    print(f"Error writing IPC files: {e}. Retrying...")
                    await asyncio.sleep(0.1)
            # prompt_file.write_text(USER_PROMPT)
            # shutil.copy(latest_image_path, str(image_file))

            print("Waiting for VLM response...")
            answer = await wait_for_answer(answer_file, timeout=VLM_TIMEOUT)

            if answer:
                print(f"\nVLM Response: {answer.strip()}")
                answer_file.unlink()
                last_processed_image = latest_image_path

                command = parse_command(answer)
                await execute_command(drone, command)
            else:
                print("Failed to get a response from VLM server.")
                for f in [prompt_file, image_file]:
                    if f.exists():
                        f.unlink()

            await asyncio.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Landing drone...")
        try:
            await drone.land()
        except Exception as e:
            print(f"Error during landing: {e}")
        await drone.disconnect()
        # Clean up any leftover IPC files
        for f in [prompt_file, image_file, answer_file]:
            if f.exists():
                f.unlink()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
