"""Infrastructure for Cluo Duo on Modal.

This file *is* the infrastructure: Modal reads it and creates the container image,
storage and web endpoint it describes. Deploy with:

    uv run modal deploy modal_app.py      # creates or updates the live app
    uv run modal serve modal_app.py       # temporary dev URL that reloads on save

The game code itself lives in src/cluoduo and knows nothing about Modal.
"""

import modal

# One "app" groups everything below under a single name in the Modal dashboard.
app = modal.App("cluo-duo")

# The container image: what's installed in the machine that runs our code.
# Versions are pinned to match uv.lock, so the server runs what the tests ran.
# add_local_python_source copies our own package into the image at deploy time.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("fastapi[standard]==0.162.0", "pydantic==2.14.1", "numpy==2.5.4")
    .env({"CLUODUO_VECTORS": "/data/glove50k.npz"})
    # Local files go last: Modal mounts them at container start instead of rebuilding the image.
    # Word vectors for the GloVe partner (built by scripts/prepare_vectors.py; not in git).
    .add_local_file("data/glove50k.npz", "/data/glove50k.npz")
    .add_local_python_source("cluoduo")
)

# Games in progress. A Modal Dict is a key-value store shared by every container,
# so a player's next move can land on any container and still find their game.
# Entries expire after 7 days without reads or writes, which suits abandoned games.
games = modal.Dict.from_name("cluo-duo-games", create_if_missing=True)

# Usage stats outlive the Dict: once a day every game is copied to a file on this volume.
# Read them with scripts/usage.py.
stats = modal.Volume.from_name("cluo-duo-stats", create_if_missing=True)
SNAPSHOT_DAYS = 7  # games untouched for this long are on the volume already, so drop them from the Dict


@app.function(
    image=image,
    # Cost and safety limits:
    min_containers=0,  # scale to zero when nobody's playing, so idle time costs nothing
    max_containers=2,  # hard ceiling on parallel machines, which also caps the bill
    scaledown_window=120,  # keep a warm container for 2 minutes after the last request
    # Kill any single request after 10 minutes. GloVe moves take milliseconds; the Smart partner's
    # first move can wait several minutes for its GPU to boot.
    timeout=10 * 60,
)
# One container serves many requests at once; game moves are short and mostly waiting.
@modal.concurrent(max_inputs=20)
# Serve the FastAPI app on a public HTTPS URL (printed by `modal deploy`).
@modal.asgi_app(label="cluo-duo")
def web():
    from cluoduo.api import create_app
    from cluoduo.store import ModalDictStore

    from cluoduo.llm import warm_up

    return create_app(ModalDictStore(games), warm_up=warm_up, usage=usage_this_month)


def usage_this_month() -> float:
    """This month's workspace usage in dollars, as Modal's usage limit counts it: metered cost less
    the free storage and egress allowances, before credits. Read with the container's own token."""
    s = modal.Workspace.from_context().billing.summary()
    free = sum(s.adjustments.get(k, 0) for k in ("Free Storage", "Network Egress Allowance"))  # negative
    return float(s.metered_cost + free)


@app.function(image=image, volumes={"/stats": stats}, schedule=modal.Cron("0 18 * * *"), timeout=10 * 60)
def snapshot():
    """Daily (5am Melbourne): copy every game to /stats/games/<id>.json, then drop long-idle games from the Dict."""
    import json
    import os
    import time

    os.makedirs("/stats/games", exist_ok=True)
    copied = dropped = 0
    for key, state in games.items():
        if key.startswith("__") or not isinstance(state, dict):
            continue
        with open(f"/stats/games/{key}.json", "w") as f:
            json.dump(state, f)
        copied += 1
        if time.time() - state.get("updated", time.time()) > SNAPSHOT_DAYS * 24 * 3600:
            games.pop(key, None)
            dropped += 1
    stats.commit()
    print(f"Copied {copied} games, dropped {dropped} idle ones from the Dict.")
