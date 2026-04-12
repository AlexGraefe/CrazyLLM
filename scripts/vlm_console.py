#!/usr/bin/env python3
"""
VLM Console Script

This script continuously reads the latest image from the ./IMAGES folder and processes it with a VLM server.
Images follow the pattern: frame_<image_nmbr>_<date>_<time>.jpg

The script communicates with vlm_server.py via file-based IPC (temp/prompt.txt, temp/image.jpg, temp/answer.txt).
"""

import os
import re
import time
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
import sys


def parse_image_filename(filename: str) -> tuple:
    """
    Parse image filename to extract image number, date, and time.
    
    Expected format: frame_<image_nmbr>_<date>_<time>.jpg
    Example: frame_42_20260412_153045.jpg
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (image_number, datetime_object) or None if parsing fails
    """
    pattern = r'frame_(\d+)_(\d{8})_(\d{6})_(\d{3})\.jpg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    image_number = int(match.group(1))
    date_str = match.group(2)  # YYYYMMDD
    time_str = match.group(3)  # HHMMSS
    
    try:
        # Parse date and time
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return (image_number, dt)
    except ValueError:
        return None


def get_latest_image(images_folder: str = "./IMAGES") -> str:
    """
    Find the latest image in the folder based on date, time, and image number.
    
    Args:
        images_folder: Path to the images folder
        
    Returns:
        Full path to the latest image, or None if no valid images found
    """
    images_path = Path(images_folder)
    
    if not images_path.exists():
        print(f"Error: Images folder '{images_folder}' does not exist.")
        return None
    
    # Get all .jpg files
    image_files = list(images_path.glob("frame_*.jpg"))
    
    if not image_files:
        print(f"No images found in '{images_folder}'.")
        return None
    
    # Parse all filenames and filter valid ones
    parsed_images = []
    for img_file in image_files:
        parsed = parse_image_filename(img_file.name)
        if parsed:
            image_number, dt = parsed
            parsed_images.append((img_file, image_number, dt))
    
    if not parsed_images:
        print("No valid images found matching the pattern frame_<number>_<date>_<time>.jpg")
        return None
    
    # Sort by datetime (descending), then by image_number (descending)
    # The latest image has the highest datetime, and if tied, highest image number
    parsed_images.sort(key=lambda x: (x[2], x[1]), reverse=True)
    
    latest_image = parsed_images[0][0]
    print(f"Found latest image: {latest_image.name}")
    return str(latest_image)


def wait_for_answer(answer_file: Path, timeout: float = 30.0) -> str:
    """
    Wait for the answer file to appear and return its content.
    
    Args:
        answer_file: Path to the answer file
        timeout: Maximum time to wait in seconds
        
    Returns:
        Content of the answer file or None if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if answer_file.exists():
            time.sleep(0.1)  # Brief delay to ensure file is fully written
            try:
                content = answer_file.read_text()
                return content
            except Exception as e:
                print(f"Error reading answer: {e}")
                return None
        time.sleep(0.1)
    
    print("Timeout waiting for VLM response")
    return None


def main():
    """Main function to run the VLM console in continuous loop mode."""
    print("VLM Console - Continuous Image Processing")
    print("=" * 60)
    print("This console reads the latest image from ./IMAGES folder in a loop.")
    print("Make sure vlm_server.py is running!")
    print("=" * 60)
    
    # Setup temp directory for IPC
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    prompt_file = temp_dir / "prompt.txt"
    image_file = temp_dir / "image.jpg"
    answer_file = temp_dir / "answer.txt"
    
    # Get the prompt once at the start
    print("\nEnter your prompt (this will be used for all images):")
    user_prompt = "Fly towards a human. For this, first center it in the image. Then, if it is in the center fly forward. If you do not see a human, turn left."  # input("Prompt: ").strip()
    
    if not user_prompt:
        print("No prompt provided. Exiting.")
        return
    
    print(f"\nUsing prompt: {user_prompt}")
    print("\nStarting continuous processing loop...")
    print("Press Ctrl+C to stop\n")
    
    last_processed_image = None
    iteration = 0
    
    while True:
        try:
            iteration += 1
            print(f"\n[Iteration {iteration}] Looking for latest image...")
            
            # Get the latest image
            latest_image_path = get_latest_image("./IMAGES")
            
            if not latest_image_path:
                print("No image found. Waiting 2 seconds...")
                time.sleep(2)
                continue
            
            # Check if this is a new image
            if latest_image_path == last_processed_image:
                print(f"Same image as before: {Path(latest_image_path).name}")
                print("Waiting 2 seconds for new image...")
                time.sleep(2)
                continue
            
            # Load and verify the image
            try:
                image = Image.open(latest_image_path)
                print(f"Found new image: {Path(latest_image_path).name} (size: {image.size})")
            except Exception as e:
                print(f"Error loading image: {e}")
                time.sleep(2)
                continue
            
            # Write prompt and image to temp files for VLM server
            print("Sending to VLM server...")
            prompt_file.write_text(user_prompt)
            shutil.copy(latest_image_path, image_file)
            
            # Wait for answer
            print("Waiting for VLM response...")
            answer = wait_for_answer(answer_file, timeout=60.0)
            
            if answer:
                print("\n" + "=" * 60)
                print("VLM Response:")
                print(answer)
                print("=" * 60)
                
                # Clean up answer file
                answer_file.unlink()
                last_processed_image = latest_image_path
            else:
                print("Failed to get response from VLM server.")
                # Clean up in case of timeout
                if prompt_file.exists():
                    prompt_file.unlink()
                if image_file.exists():
                    image_file.unlink()
            
            # Wait before next iteration
            print("\nWaiting 3 seconds before next check...")
            time.sleep(3)
            
        except KeyboardInterrupt:
            print("\n\nStopping console. Goodbye!")
            # Clean up any remaining files
            for f in [prompt_file, image_file, answer_file]:
                if f.exists():
                    f.unlink()
            break
        except Exception as e:
            print(f"\nError in main loop: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
