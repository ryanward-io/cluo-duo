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
    "v7 + sudden death (Haiku)": range(96, 104),
    "v7 + sudden death (Sonnet)": range(104, 112),
    "v8 (Haiku)": range(112, 120),
    "v8 (Sonnet)": range(120, 128),
    "v9 (Haiku)": range(128, 136),
    "v9 (Sonnet)": range(136, 144),
    "v10 (Sonnet)": range(144, 152),
    "v10 (Haiku)": range(152, 160),
}
# Replays: an earlier guide on the same boards as a later one, to tell guide effects from hard boards.
REPLAYS = {"v9 replayed on v10's boards (Sonnet)": (range(144, 152), "v9")}
# Reference-partner test: Sonnet + guide v9 on side a, each AI tier on side b, same boards.
REFERENCE = {tier: (range(200, 204), tag) for tier, tag in
             [("ref + Random", "random"), ("ref + Primitive", "glove-sum"), ("ref + Basic", "glove-rank"),
              ("ref + Smart (Qwen)", "qwen36-sim"), ("ref + Basic, capped at 3, no leftovers", "glove-rank-safe"),
              ("ref + Smart, GloVe vets later guesses", "qwen36-sim-cautious"),
              ("ref + Smart: Qwen clues, Bayesian guesses", "smart")]}
REFERENCE["ref + Smart: Qwen clues, Bayesian guesses"] = (range(200, 208), "smart")
REFERENCE |= {"ref + Smart, 10 turns": (range(200, 208), "smart-t10"),
              "ref + Basic (capped), 10 turns": (range(200, 204), "glove-rank-t10"),
              "ref + Smart, full clue numbers": (range(200, 208), "smart-full"),
              "ref + Smart on Qwen3.6-27B (dense)": (range(200, 208), "smart27"),
              "ref + Smart 27B, model also guesses": (range(200, 208), "smart27-llmguess"),
              "ref + Smart on Gemma 4 31B": (range(200, 208), "smart-gemma"),
              "ref + Smart Gemma 31B, model also guesses": (range(200, 208), "smart-gemma-llmguess"),
              "ref + Smart Gemma 31B, no GloVe": (range(200, 208), "smart-gemma-noglove"),
              "ref + Smart on gpt-oss-120b": (range(200, 208), "smart-gptoss"),
              "ref + Smart gpt-oss, model also guesses": (range(200, 208), "smart-gptoss-llmguess"),
              "ref + Smart on Mistral Small 3.2 24B": (range(200, 208), "smart-mistral"),
              "ref + Smart Mistral, model also guesses": (range(200, 208), "smart-mistral-llmguess"),
              "ref + Sonnet (guide v9)": (range(200, 208), "duo-sonnet"),
              "ref + Smart Gemma, rehearsals capped at 10 s": (range(200, 208), "smart-gemma-cap10")}


def round_summary(name: str, seeds: range, tag: str = "") -> str:
    suffix = (f"-duo-{tag}" if tag else "-duo") if not name.startswith("ref") else f"-{tag}"
    states = [json.loads(p.read_text()) for s in seeds if (p := GAMES / f"seed{s}{suffix}.json").exists()]
    done = [s for s in states if s["result"]]
    if not done:
        return f"{name:<20} no finished games"
    clues = [h for s in done for h in s["history"] if h["type"] == "clue"]
    count = lambda r: sum(s["result"] == r for s in done)
    in_turns = sum(s["result"] == "won" and not any(h["type"] == "sudden_death" for h in s["history"]) for s in done)
    return (f"{name:<27} games {len(done):>2}  won {count('won'):>2} (in 9 turns {in_turns:>2})  assassin {count('assassin'):>2}  "
            f"out of turns/sudden death {count('out_of_turns') + count('sudden_death'):>2}  avg found {sum(map(game.found_count, done)) / len(done):4.1f}  "
            f"1-clues {sum(c['number'] == 1 for c in clues) / len(clues):4.0%}  "
            f"agents/clue {sum(map(game.found_count, done)) / len(clues):4.2f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--rounds"]:
        for name, seeds in ROUNDS.items():
            print(round_summary(name, seeds))
        for name, (seeds, tag) in (REPLAYS | REFERENCE).items():
            print(round_summary(name, seeds, tag))
        sys.exit()
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    for path in sorted(GAMES.glob(f"{pattern}.json")):
        print(summarise(json.loads(path.read_text())))
