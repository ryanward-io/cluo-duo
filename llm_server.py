"""Infrastructure: open-source LLMs served by vLLM on Modal GPUs.

Two models, each its own class (so its own GPU type and containers):

- `LLM`: Qwen3-8B (8-bit) on an L4, about $0.80/hour while running.
- `Qwen27`: Qwen3.6-27B (8-bit, dense) on an L40S, about $2/hour while running.
- `Qwen36`: Qwen3.6-35B-A3B (8-bit) on an L40S, about $2/hour while running. A "mixture of
  experts": 35B parameters in total but only ~3B are used per token, so it knows far more
  than the 8B model yet generates about as fast. The weights (37.5 GB) need the 48 GB card.

This is a separate Modal app from the game API so each can be deployed on its own.
There is deliberately NO public URL: the model is a Modal class whose `chat` method can
only be called by code signed in to this Modal workspace (the game backend, or the
evaluation scripts on Ryan's machine). A public GPU endpoint would let anyone run up the bill.

    uv run modal deploy llm_server.py                 # deploy both (first boots download the weights)
    uv run modal run llm_server.py                    # quick smoke test from the terminal
    uv run modal run llm_server.py --model Qwen36     # the same on the bigger model

Both scale to zero when idle, so the bill is roughly (number of play sessions) x
(session length + 5 minutes) at that GPU's hourly rate. GPU time comes out of the plan's
monthly credit.
"""

import json
import subprocess
import time

import modal

PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "httpx==0.28.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster weight downloads from Hugging Face
)

# Volumes persist between containers, so the weights download once, not on every cold start.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("cluo-duo-llm")


class Server:
    """Shared by both models: start vLLM inside the container, then forward chat requests to it."""

    MODEL: str
    REVISION: str  # pinned so the model can't change under us
    VLLM_ARGS: list[str] = []
    SAMPLING: dict = {}  # the model card's recommended settings, sent with every request

    @modal.enter()
    def start(self) -> None:
        cmd = [
            "vllm", "serve", self.MODEL,
            "--revision", self.REVISION,
            "--served-model-name", "llm",
            "--host", "127.0.0.1", "--port", str(PORT),  # only reachable inside this container
            "--reasoning-parser", "qwen3",  # keeps any <think> text out of the JSON answer
            *self.VLLM_ARGS,
        ]
        self.process = subprocess.Popen(cmd)
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        import httpx

        deadline = time.time() + 20 * MINUTES
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("vLLM exited during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError("vLLM did not become ready")

    @modal.method()
    async def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 600,
                   thinking: bool = False, temperature: float = 0.7, seed: int | None = None) -> dict:
        """One chat completion. With `schema`, the reply is forced to be JSON matching it."""
        import httpx

        body = {
            "model": "llm",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking},  # Qwen's switch for step-by-step reasoning
            **self.SAMPLING,
        }
        if seed is not None:
            body["seed"] = seed
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        async with httpx.AsyncClient(timeout=280) as client:
            r = await client.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content"), "usage": data["usage"]}

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}


@app.cls(
    image=image,
    gpu="L4",
    volumes=VOLUMES,
    min_containers=0,  # scale to zero: no GPU bill while nobody is playing
    max_containers=1,  # never more than one GPU at a time, whatever the traffic
    scaledown_window=5 * MINUTES,  # stay warm for 5 minutes after the last request
    timeout=2 * MINUTES,  # per request
    startup_timeout=20 * MINUTES,  # first boot downloads the weights
)
@modal.concurrent(max_inputs=64)  # vLLM batches concurrent requests on the one GPU
class LLM(Server):
    MODEL = "Qwen/Qwen3-8B-FP8"  # 8-bit weights: ~9 GB, fits an L4 (24 GB) with room for the KV cache
    REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"
    VLLM_ARGS = [
        "--enforce-eager",  # skip graph compilation: faster cold starts, slightly slower generation
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
    ]


@app.cls(
    image=image,
    gpu="L40S",
    volumes=VOLUMES,
    min_containers=0,
    max_containers=1,
    scaledown_window=5 * MINUTES,
    timeout=5 * MINUTES,  # room for thinking-mode replies
    startup_timeout=30 * MINUTES,  # first boot downloads 37.5 GB
)
@modal.concurrent(max_inputs=64)
class Qwen36(Server):
    MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
    REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
    VLLM_ARGS = [
        "--language-model-only",  # skip the vision encoder: frees memory for the KV cache
        "--max-model-len", "16384",  # thinking replies can run to a few thousand tokens
        "--gpu-memory-utilization", "0.90",
        # No --enforce-eager: CUDA graphs speed up generation a lot for this kind of model, and the
        # compiled graphs are kept on the vllm-cache volume, so only the first boot pays for them.
        # But the weights fill most of the card, and recording graphs for vLLM's default batch sizes
        # ran out of memory at startup. We never send more than ~16 requests at once, so cap at 32.
        "--max-num-seqs", "32",
        "--max-cudagraph-capture-size", "32",
    ]
    # Model card, non-thinking mode (thinking mode wants temperature 1.0 and top_p 0.95).
    SAMPLING = {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


@app.cls(
    image=image,
    gpu="L40S",
    volumes=VOLUMES,
    min_containers=0,
    max_containers=1,
    scaledown_window=5 * MINUTES,
    timeout=5 * MINUTES,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=64)
class Qwen27(Server):
    """Qwen3.6-27B: dense, so all 27B parameters work on every word (the 35B model uses ~3B).
    Slower per word, but much more model per decision. 8-bit weights are 31 GB."""

    MODEL = "Qwen/Qwen3.6-27B-FP8"
    REVISION = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
    VLLM_ARGS = [
        "--language-model-only",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        # Same caps that stopped Qwen36 running out of memory while recording CUDA graphs.
        "--max-num-seqs", "32",
        "--max-cudagraph-capture-size", "32",
    ]
    SAMPLING = {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


@app.local_entrypoint()
def main(model: str = "LLM") -> None:
    schema = {
        "type": "object",
        "properties": {"clue": {"type": "string"}, "targets": {"type": "array", "items": {"type": "string"}}},
        "required": ["clue", "targets"],
    }
    start = time.time()
    out = {"LLM": LLM, "Qwen36": Qwen36, "Qwen27": Qwen27}[model]().chat.remote(
        [{"role": "user", "content": "Give a one-word clue linking PENGUIN, OSTRICH and KITE (the bird). Reply as JSON."}],
        schema=schema,
    )
    print(json.dumps(out, indent=2))
    print(f"took {time.time() - start:.0f}s (includes cold start)")
