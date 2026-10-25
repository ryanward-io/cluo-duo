"""Paired comparison of two players on the same boards.

    uv run python scripts/compare.py glove-sum glove-bayes --games 1000 --first-seed 20000

Both players see identical boards, so only boards where exactly one of them won tell them
apart. The exact McNemar test asks how surprising that split would be if they were equally
good: a much more sensitive comparison than checking whether two win-rate intervals overlap.
"""

from __future__ import annotations

import argparse
from math import comb

from cluoduo.players import PLAYERS, self_play


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for b vs c discordant pairs (a sign test with p = 0.5)."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("first", choices=sorted(PLAYERS))
    p.add_argument("second", choices=sorted(PLAYERS))
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--first-seed", type=int, default=20000)
    args = p.parse_args()

    players = {name: (PLAYERS[name](seed=1), PLAYERS[name](seed=2)) for name in (args.first, args.second)}
    wins = {name: 0 for name in players}
    assassins = {name: 0 for name in players}
    only = {name: 0 for name in players}
    for seed in range(args.first_seed, args.first_seed + args.games):
        start = "a" if seed % 2 == 0 else "b"
        results = {name: self_play(*pair, seed=seed, first=start)["result"] for name, pair in players.items()}
        for name, result in results.items():
            wins[name] += result == "won"
            assassins[name] += result == "assassin"
        won = [name for name, result in results.items() if result == "won"]
        if len(won) == 1:
            only[won[0]] += 1

    for name in players:
        print(f"{name:<14} wins {wins[name] / args.games:6.1%}   assassin losses {assassins[name] / args.games:6.1%}"
              f"   boards only it won: {only[name]}")
    print(f"exact McNemar p = {mcnemar_exact(only[args.first], only[args.second]):.3f} over {args.games} shared boards")


if __name__ == "__main__":
    main()
