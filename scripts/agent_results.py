"""Summarise saved agent games (data/agent-games/): result, agents found, and how each assassin was hit.

    uv run python scripts/agent_results.py [PATTERN]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cluoduo import game

GAMES = Path(__file__).resolve().parents[1] / "data" / "agent-games"


def summarise(state: dict) -> str:
    hist = state["history"]
    turns = sum(h["type"] == "clue" for h in hist)
    line = f"{state['id']:<14} {state['result'] or 'in progress':<13} found {game.found_count(state):>2}/15  turns {turns}"
    for n, h in enumerate(hist):
        if h["type"] == "guess" and h["outcome"] == "A":
            clue = [c for c in hist[:n] if c["type"] == "clue"][-1]
            guesses_on_clue = sum(1 for g in hist[hist.index(clue):n + 1] if g["type"] == "guess")
            guesser_key = state["key"][h["side"]][h["index"]]
            line += (f"\n    {h['side']} hit {h['word']} on {clue['word']} {clue['number']} (guess #{guesses_on_clue}"
                     f"{', past the number' if guesses_on_clue > clue['number'] else ''}; on guesser's own key: {guesser_key})")
    return line


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    for path in sorted(GAMES.glob(f"{pattern}.json")):
        print(summarise(json.loads(path.read_text())))
