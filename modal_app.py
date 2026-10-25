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


@app.function(
    image=image,
    # Cost and safety limits:
    min_containers=0,  # scale to zero when nobody's playing, so idle time costs nothing
    max_containers=2,  # hard ceiling on parallel machines, which also caps the bill
    scaledown_window=120,  # keep a warm container for 2 minutes after the last request
    timeout=60,  # kill any single request that runs longer than a minute
)
# One container serves many requests at once; game moves are short and mostly waiting.
@modal.concurrent(max_inputs=20)
# Serve the FastAPI app on a public HTTPS URL (printed by `modal deploy`).
@modal.asgi_app(label="cluo-duo")
def web():
    from cluoduo.api import create_app
    from cluoduo.store import ModalDictStore

    return create_app(ModalDictStore(games))
