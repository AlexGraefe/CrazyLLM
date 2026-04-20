#!/usr/bin/env python3
"""
VLM Server - Watches for prompt and image files and generates responses.
"""

import argparse
import time
from pathlib import Path
from PIL import Image

from llm.vlm import VLM

SYSTEM_PROMPT = """You are a drone. You receive instructions from the user and an image. The image is taken from your onboard camera. The camera faces forward and what you see in the middle is what you fly onto. Based on the instruction and the image control the drone. You have to answer with forward, backward, left, right, turn left, turn right, nothing, which the drone will execute. You answer should have a short explanation (max. 2 sentences), followed by a | and then the command. For example, if the user says "fly towards the red object" and the image contains a red object in the center, you can answer "The red object is in front of me, I will fly forward | forward". Do not think too long about your answer."""


def main():
    parser = argparse.ArgumentParser(description="VLM server for drone console (file-based IPC)")
    parser.add_argument("--model", type=str, default="qwen-vl", help="Model name (qwen-vl or full HF path)")
    parser.add_argument("--cluster", action="store_true", help="Use cluster model paths")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--temp-dir", type=str, default="temp", help="Directory for prompt/answer/image files")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    prompt_file = temp_dir / "prompt.txt"
    image_file = temp_dir / "image.jpg"
    answer_file = temp_dir / "answer.txt"

    print(f"Loading model '{args.model}'...")
    vlm = VLM(
        model_name=args.model,
        use_cluster=args.cluster,
        quantize=not args.no_quantize,
    )
    print(f"Watching {temp_dir} for prompt.txt and image.jpg ... (Ctrl-C to stop)\n")

    while True:
        try:
            # Wait for both prompt and image files
            if not (prompt_file.exists() and image_file.exists()):
                time.sleep(0.1)
                continue

            time.sleep(1.0)  # ensure files are fully written
            
            prompt = prompt_file.read_text().strip()
            image = Image.open(image_file)
            
            print(f"[server] Received prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            print(f"[server] Received image: {image.size}")

            print("[server] Generating response...")
            answer = vlm.predict(prompt=prompt, image=image, system_prompt=SYSTEM_PROMPT, max_new_tokens=args.max_tokens)

            answer_file.write_text(answer)
            print(f"[server] Answer written ({len(answer)} chars).")
            
            # Clean up input files
            prompt_file.unlink()
            image_file.unlink()

        except KeyboardInterrupt:
            print("\n[server] Shutting down.")
            break
        except Exception as e:
            print(f"[server] Error: {e}")
            answer_file.write_text(f"ERROR: {e}")
            if prompt_file.exists():
                prompt_file.unlink()
            if image_file.exists():
                image_file.unlink()


if __name__ == "__main__":
    main()
