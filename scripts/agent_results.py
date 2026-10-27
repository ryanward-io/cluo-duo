"""Summarise saved agent games (data/agent-games/): result, agents found, and how each assassin was hit.

    uv run python scripts/agent_results.py [PATTERN]
    uv run python scripts/agent_results.py --rounds     # one summary line per guide version
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


# Which boards each round of the player guide (prompts/player.md) was tested on.
ROUNDS = {
    "no guide (Haiku)": range(10, 12),
    "guide v1 (Haiku)": range(12, 16),
    "guide v2 (Haiku)": range(16, 28),
    "guide v2 (Sonnet)": range(28, 32),
    "guide v3 (Haiku)": range(32, 44),
    "guide v3 (Sonnet)": range(44, 48),
    "guide v4 (Haiku)": range(48, 60),
    "guide v4 (Sonnet)": range(60, 64),
    "guide v5 (Haiku)": range(64, 76),
    "guide v5 (Sonnet)": range(76, 80),
    "guide v6 (Haiku)": range(80, 92),
    "guide v6 (Sonnet)": range(92, 96),
}


def round_summary(name: str, seeds: range) -> str:
    states = [json.loads(p.read_text()) for s in seeds if (p := GAMES / f"seed{s}-duo.json").exists()]
    done = [s for s in states if s["result"]]
    if not done:
        return f"{name:<20} no finished games"
    clues = [h for s in done for h in s["history"] if h["type"] == "clue"]
    count = lambda r: sum(s["result"] == r for s in done)
    return (f"{name:<20} games {len(done):>2}  won {count('won'):>2}  assassin {count('assassin'):>2}  "
            f"out of turns {count('out_of_turns'):>2}  avg found {sum(map(game.found_count, done)) / len(done):4.1f}  "
            f"1-clues {sum(c['number'] == 1 for c in clues) / len(clues):4.0%}  "
            f"agents/clue {sum(map(game.found_count, done)) / len(clues):4.2f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--rounds"]:
        for name, seeds in ROUNDS.items():
            print(round_summary(name, seeds))
        sys.exit()
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    for path in sorted(GAMES.glob(f"{pattern}.json")):
        print(summarise(json.loads(path.read_text())))
