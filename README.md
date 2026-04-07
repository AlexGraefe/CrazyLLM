# CrazyLLM

## How to use it
Start [./scripts/llm_server.py](./scripts/llm_server.py) locally or on a remote system with a GPU:

```
uv run -m scripts.llm_server
```

If in a remote system, run [.mount_llm_service.sh](.mount_llm_service.sh) on your system to mount the temp folder to the one on the remote system (you might need to adjust paths). 

Then, start [./scripts/llm_console.py](./scripts/llm_console.py) on your local machine:

```
uv run -m scripts.llm_console
```