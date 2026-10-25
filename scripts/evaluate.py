"""Self-play evaluation: the same player on both sides, over many seeded boards.

    uv run python scripts/evaluate.py glove-sum random --games 300
    uv run python scripts/evaluate.py glove-sum --set link_threshold=0.35,0.4,0.45

Prints win rate (with a 95% interval), how games ended, agents found and turns used.
The seeds are fixed, so every player is tested on exactly the same boards.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import Counter

from cluoduo import game
from cluoduo.players import PLAYERS, self_play


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (stays within 0-1, sensible at 0 wins)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def evaluate(name: str, games: int, first_seed: int = 0, **kwargs) -> dict:
    cls = PLAYERS[name]
    a, b = cls(seed=1, **kwargs), cls(seed=2, **kwargs)
    ends, found, turns, sizes, hits = Counter(), [], [], [], []
    start = time.perf_counter()
    for seed in range(first_seed, first_seed + games):
        s = self_play(a, b, seed=seed, first="a" if seed % 2 == 0 else "b")
        ends[s["result"]] += 1
        found.append(game.found_count(s))
        turns.append(game.TURNS - s["turns_left"])
        sizes += [h["number"] for h in s["history"] if h["type"] == "clue"]
        hits.append(
            sum(h["outcome"] == "G" for h in s["history"] if h["type"] == "guess")
        )
    lo, hi = wilson_interval(ends["won"], games)
    return {
        "player": name
        + (" " + " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""),
        "win": ends["won"] / games,
        "ci": (lo, hi),
        "assassin": ends["assassin"] / games,
        "out_of_turns": ends["out_of_turns"] / games,
        "found": sum(found) / games,
        "turns": sum(turns) / games,
        "clue": sum(sizes) / max(len(sizes), 1),
        "per_turn": sum(hits) / max(sum(turns), 1),
        "secs": time.perf_counter() - start,
    }


def parse_sets(items: list[str]) -> list[dict]:
    """--set a=1,2 --set b=3 -> every combination: [{a:1,b:3}, {a:2,b:3}]"""
    grid = {}
    for item in items:
        key, values = item.split("=", 1)
        grid[key] = [float(v) for v in values.split(",")]
    keys = list(grid)
    return [
        dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))
    ] or [{}]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("players", nargs="+", choices=sorted(PLAYERS))
    p.add_argument("--games", type=int, default=200)
    p.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="tune on one range of boards, report on another",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PARAM=V1,V2",
        help="player settings to sweep",
    )
    args = p.parse_args()

    print(
        f"{'player':<44} {'win':>6} {'95% CI':>13} {'assassin':>9} {'no turns':>9} {'found':>6} {'turns':>6} {'clue n':>6} {'agents/turn':>11} {'secs':>6}"
    )
    for name in args.players:
        for kwargs in parse_sets(args.set) if name != "random" else [{}]:
            r = evaluate(name, args.games, args.first_seed, **kwargs)
            print(
                f"{r['player']:<44} {r['win']:6.1%} {r['ci'][0]:6.1%}-{r['ci'][1]:<6.1%} {r['assassin']:9.1%} "
                f"{r['out_of_turns']:9.1%} {r['found']:6.1f} {r['turns']:6.1f} {r['clue']:6.2f} {r['per_turn']:11.2f} {r['secs']:6.1f}"
            )


if __name__ == "__main__":
    main()
