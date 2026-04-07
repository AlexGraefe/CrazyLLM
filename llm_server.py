#!/usr/bin/env python3
"""
LLM Server - Watches for prompt files and generates responses, maintaining chat history.
"""

import argparse
import time
from pathlib import Path

from llm.llm import LLM
from llm.dataset_generation import SYSTEM_PROMPT


RESET_COMMAND = "__RESET__"


def extract_answer(response: str) -> str:
    assistant_idx = response.lower().rfind("assistant")
    if assistant_idx != -1:
        return response[assistant_idx + len("assistant"):].strip()
    raise ValueError("Could not find 'assistant' in the response.")


def fresh_chat() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def main():
    parser = argparse.ArgumentParser(description="LLM server for drone console (file-based IPC)")
    parser.add_argument("--model", type=str, default="devjalx", help="Model name (gemma, qwen, devjalx, or full HF path)")
    parser.add_argument("--cluster", action="store_true", help="Use cluster model paths")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (for supported models)")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--temp-dir", type=str, default="temp", help="Directory for prompt/answer files")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    prompt_file = temp_dir / "prompt.txt"
    answer_file = temp_dir / "answer.txt"

    print(f"Loading model '{args.model}'...")
    llm = LLM(
        model_name=args.model,
        thinking=args.thinking,
        use_cluster=args.cluster,
        quantize=not args.no_quantize,
    )
    print(f"Watching {temp_dir} for prompt.txt ... (Ctrl-C to stop)\n")

    current_chat = fresh_chat()

    while True:
        try:
            if (not prompt_file.exists()) and answer_file.exists():
                time.sleep(0.1)
                continue

            time.sleep(1.0)  # ensure file is fully written
            prompt = prompt_file.read_text().strip()
            prompt_file.unlink()
            print(f"[server] Received prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

            if prompt == RESET_COMMAND:
                current_chat = fresh_chat()
                print("[server] Chat history reset.")
                answer_file.write_text("__OK__")
                continue

            current_chat.append({"role": "user", "content": prompt})
            print("[server] Generating response...")
            response = llm.chat(current_chat)
            answer = extract_answer(response)
            current_chat.append({"role": "assistant", "content": answer})

            answer_file.write_text(answer)
            print(f"[server] Answer written ({len(answer)} chars).")

        except KeyboardInterrupt:
            print("\n[server] Shutting down.")
            break
        except Exception as e:
            print(f"[server] Error: {e}")
            answer_file.write_text(f"ERROR: {e}")
            if prompt_file.exists():
                prompt_file.unlink()


if __name__ == "__main__":
    main()
