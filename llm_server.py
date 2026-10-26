"""Infrastructure: an open-source LLM (Qwen3-8B) served by vLLM on a Modal GPU.

This is a separate Modal app from the game API so each can be deployed on its own.
There is deliberately NO public URL: the model is a Modal class whose `chat` method can
only be called by code signed in to this Modal workspace (the game backend, or the
evaluation scripts on Ryan's machine). A public GPU endpoint would let anyone run up the bill.

    uv run modal deploy llm_server.py     # deploy (first boot downloads ~9 GB of weights)
    uv run modal run llm_server.py        # quick smoke test from the terminal

Cost: an L4 GPU is about $0.80/hour while a container is running. It scales to zero
when idle, so the bill is roughly (number of play sessions) x (session length + 5 minutes).
"""

import json
import subprocess
import time

import modal

MODEL = "Qwen/Qwen3-8B-FP8"  # 8-bit weights: ~9 GB, fits an L4 (24 GB) with room for the KV cache
REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"  # pinned so the model can't change under us
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


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    min_containers=0,  # scale to zero: no GPU bill while nobody is playing
    max_containers=1,  # never more than one GPU at a time, whatever the traffic
    scaledown_window=5 * MINUTES,  # stay warm for 5 minutes after the last request
    timeout=2 * MINUTES,  # per request
    startup_timeout=20 * MINUTES,  # first boot downloads the weights
)
@modal.concurrent(max_inputs=64)  # vLLM batches concurrent requests on the one GPU
class LLM:
    @modal.enter()
    def start(self) -> None:
        cmd = [
            "vllm", "serve", MODEL,
            "--revision", REVISION,
            "--served-model-name", "llm",
            "--host", "127.0.0.1", "--port", str(PORT),  # only reachable inside this container
            "--enforce-eager",  # skip graph compilation: faster cold starts, slightly slower generation
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--reasoning-parser", "qwen3",  # keeps any <think> text out of the JSON answer
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
            "chat_template_kwargs": {"enable_thinking": thinking},  # Qwen3's switch for step-by-step reasoning
        }
        if seed is not None:
            body["seed"] = seed
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        async with httpx.AsyncClient(timeout=110) as client:
            r = await client.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content"), "usage": data["usage"]}

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


@app.local_entrypoint()
def main() -> None:
    schema = {
        "type": "object",
        "properties": {"clue": {"type": "string"}, "targets": {"type": "array", "items": {"type": "string"}}},
        "required": ["clue", "targets"],
    }
    start = time.time()
    out = LLM().chat.remote(
        [{"role": "user", "content": "Give a one-word clue linking PENGUIN, OSTRICH and KITE (the bird). Reply as JSON."}],
        schema=schema,
    )
    print(json.dumps(out, indent=2))
    print(f"took {time.time() - start:.0f}s (includes cold start)")
