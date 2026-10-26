"""Play one self-play game and print what the AI was thinking at each step.

    uv run python scripts/transcript.py qwen-hybrid --seed 3
    uv run python scripts/transcript.py qwen-hybrid+glove-rank --seed 3 > data/llm/hybrid-game.txt

For LLM players it shows each clue's intended targets and reason, the guesser's reading of
the clue, and (for the hybrid) every candidate clue with GloVe's verdict on it.
"""

from __future__ import annotations

import argparse

from cluoduo import game
from cluoduo.players import PLAYERS, self_play


def show_llm_calls(entries: list[dict]) -> None:
    for entry in entries:
        if "fallback" in entry:
            print("    (no usable clue from the model: GloVe v2 gave this one)")
            continue
        data = entry.get("parsed", {})
        if "vetted" in entry:
            for c in data.get("candidates", []):
                print(f"    proposed {str(c.get('clue', '')).upper():<14} for {', '.join(c.get('targets', []))}")
            for r in entry["vetted"]:
                print(f"      GloVe keeps {r['clue']} {r['number']} ({', '.join(r['kept'])}), margin {r['margin']:.2f}")
        elif "targets" in data:
            print(f"    targets {', '.join(data['targets'])}: {data.get('reason', '')}")
        elif "guesses" in data:
            print(f"    reads it as: {data.get('reason', '')} -> {', '.join(data['guesses'])}")
        if entry.get("problem"):
            print(f"    rejected: {entry['problem']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("players", help="one player for both sides, or x+y")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    a, b = args.players.split("+") if "+" in args.players else (args.players, args.players)
    players = {"a": PLAYERS[a](seed=1), "b": PLAYERS[b](seed=2)}
    state = self_play(players["a"], players["b"], seed=args.seed, first="a" if args.seed % 2 == 0 else "b")

    keys = {s: game.view(state, s)["my_key"] for s in "ab"}
    print(f"Board {args.seed}: side a = {a}, side b = {b}\n")
    for side in "ab":
        print(f"Side {side} key:")
        for colour, label in ((game.GREEN, "agents"), (game.ASSASSIN, "assassins")):
            print(f"  {label}: {', '.join(w for w, k in zip(state['words'], keys[side]) if k == colour)}")
    print()

    # Replay the history. Each player's log is in call order: per clue, any rejected attempts, then
    # the accepted one (or a fallback marker); per guessing turn, one plan.
    logs = {side: list(getattr(players[side], "log", [])) for side in "ab"}
    turn, first_guess = 0, False
    for h in state["history"]:
        side = h.get("side")
        if h["type"] == "clue":
            turn += 1
            first_guess = True
            print(f"Turn {turn}: {side} gives {h['word']} {h['number']}")
            show_llm_calls(take_clue(logs[side]))
        elif h["type"] == "guess":
            if first_guess and logs[side] and "guesses" in logs[side][0].get("parsed", {}):
                show_llm_calls([logs[side].pop(0)])
            first_guess = False
            print(f"  {side} guesses {h['word']}: {dict(G='agent', N='bystander', A='ASSASSIN')[h['outcome']]}")
        elif h["type"] == "stop":
            print(f"  {side} stops")
    print(f"\nResult: {state['result']}, {game.found_count(state)} of 15 agents found")


def take_clue(log: list[dict]) -> list[dict]:
    """Remove and return the entries for one clue from the front of a player's log."""
    taken = []
    while log and "guesses" not in log[0].get("parsed", {}):
        entry = log.pop(0)
        taken.append(entry)
        if "fallback" in entry or ("problem" not in entry and not (log and "fallback" in log[0])):
            break
    return taken


if __name__ == "__main__":
    main()
