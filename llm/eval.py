import argparse
from llm.llm import LLM
from llm.dataset_generation import SYSTEM_PROMPT

def extract_answer(response: str) -> str:
    print(response)
    assistant_idx = response.lower().rfind("assistant")
    if assistant_idx != -1:
        return response[assistant_idx + len("assistant"):].strip()
    raise ValueError("Could not find 'assistant' in the response.")


def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM in the console.")
    parser.add_argument("--model", type=str, default="devjalx", help="Model name: gemma, qwen, devjalx, or a full model path")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (Qwen3 only)")
    parser.add_argument("--cluster", action="store_true", help="Use HPC cluster cache directory")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    llm = LLM(
        model_name=args.model,
        thinking=args.thinking,
        use_cluster=args.cluster,
        quantize=not args.no_quantize,
    )

    print("\nChat session started. Type 'exit' or 'quit' to stop, 'reset' to clear history.\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        if user_input.lower() == "reset":
            history = []
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": user_input})

        result = llm.chat(history)

        if args.thinking:
            response, thinking = result
            print(f"\n[Thinking]\n{thinking}\n")
        else:
            response = extract_answer(result)

        print(f"\nAssistant: {response}\n")
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

