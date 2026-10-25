"""Run the API locally without Modal: uv run python dev.py, then open http://localhost:8010/?api=http://localhost:8001 (serve the page with python3 -m http.server 8010)"""

import uvicorn

from cluewords.api import create_app
from cluewords.store import MemoryStore

if __name__ == "__main__":
    uvicorn.run(create_app(MemoryStore(), allow_local=True), host="127.0.0.1", port=8001)
