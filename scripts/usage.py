"""Who's playing on the live site: games per day, distinct players, and roughly what their GPU time cost.

    uv run python scripts/usage.py              # this month
    uv run python scripts/usage.py --days 30    # the last 30 days

Games come from the daily snapshots on the `cluo-duo-stats` volume plus the live Dict (newest copy wins).
Each game is one of:
  players   a browser with a random player id (real visitors)
  Ryan      a browser marked with ?me=1
  untracked no player id: games from before tracking began (12 Nov 2026) and API tests/scripts

GPU cost: Modal's hourly bill for the model servers, split by whose Advanced games were active in each
hour. Hours with no live Advanced game were reference tests or warm-ups. It's approximate: the bill
trails by up to an hour, and the GPU stays up 5 minutes after the last move.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import modal

MELB = ZoneInfo("Australia/Melbourne")
GPU_APP = "cluo-duo-llm"
IDLE = 5 * 60  # the GPU stays up this long after its last call


def load_games() -> dict[str, dict]:
    games: dict[str, dict] = {}
    vol = modal.Volume.from_name("cluo-duo-stats", create_if_missing=True)
    try:
        paths = [e.path for e in vol.listdir("games")]
    except FileNotFoundError:
        paths = []  # no snapshot yet
    with ThreadPoolExecutor(16) as pool:  # one round trip per file, so fetch them side by side
        for state in pool.map(lambda p: json.loads(b"".join(vol.read_file(p))), paths):
            games[state["id"]] = state
    for key, state in modal.Dict.from_name("cluo-duo-games").items():
        if key.startswith("__") or not isinstance(state, dict):
            continue
        if state.get("updated", 0) >= games.get(key, {}).get("updated", 0):
            games[key] = state
    return games


def who(state: dict) -> str:
    if state.get("me"):
        return "Ryan"
    return "players" if state.get("player") else "untracked"


def human_moves(state: dict) -> int:
    return sum(1 for h in state["history"] if h.get("side") == "a" and h["type"] in ("clue", "guess", "stop"))


def hourly_gpu_cost(start: datetime) -> dict[datetime, float]:
    """Model-server cost per UTC hour (the CLI allows at most 7 days per hourly report)."""
    cost: dict[datetime, float] = defaultdict(float)
    now = datetime.now(timezone.utc)
    while start < now:
        end = min(start + timedelta(days=7), now)
        out = subprocess.run(["modal", "billing", "report", "--start", start.strftime("%Y-%m-%dT%H:%M"),
                              "--end", end.strftime("%Y-%m-%dT%H:%M"), "-r", "h", "--json"],
                             capture_output=True, text=True, check=True).stdout
        for row in json.loads(out):
            if row["description"] == GPU_APP:
                cost[datetime.fromisoformat(row["interval_start"]).replace(tzinfo=timezone.utc)] += float(row["cost"])
        start = end
    return cost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, help="look back this many days (default: since the 1st of this month)")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days) if args.days else now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    games = [g for g in load_games().values() if "created" not in g or g["created"] >= start.timestamp()]
    tracked = [g for g in games if "created" in g]

    print(f"Since {start.astimezone(MELB):%d %b %Y} (Melbourne time)\n")
    print("Games by who played them (played = you made at least one move):")
    rows = defaultdict(Counter)
    for g in games:
        r = rows[who(g)]
        r["games"] += 1
        r["played"] += human_moves(g) > 0
        r["finished"] += g["phase"] == "over"
        r["won"] += g.get("result") == "won"
        r["advanced"] += g.get("partner") == "smart"
    print(f"  {'':10} {'games':>6} {'played':>7} {'finished':>9} {'won':>4} {'Advanced':>9}")
    for name in ("players", "Ryan", "untracked"):
        r = rows[name]
        print(f"  {name:10} {r['games']:6} {r['played']:7} {r['finished']:9} {r['won']:4} {r['advanced']:9}")
    print(f"  Distinct player browsers: {len({g['player'] for g in games if who(g) == 'players'})}")

    days = defaultdict(lambda: (Counter(), set()))
    for g in tracked:
        if who(g) != "players":
            continue
        c, ids = days[datetime.fromtimestamp(g["created"], MELB).date()]
        c["games"] += 1
        c["played"] += human_moves(g) > 0
        c["advanced"] += g.get("partner") == "smart"
        ids.add(g["player"])
    if days:
        print("\nPlayers by day:")
        print(f"  {'':10} {'browsers':>9} {'games':>6} {'played':>7} {'Advanced':>9}")
        for d in sorted(days):
            c, ids = days[d]
            print(f"  {d:%a %d %b} {len(ids):9} {c['games']:6} {c['played']:7} {c['advanced']:9}")

    cost = hourly_gpu_cost(start)
    split = Counter()
    for hour, dollars in cost.items():
        lo, hi = hour.timestamp(), hour.timestamp() + 3600
        active = {who(g) for g in tracked if g.get("gpu_counted")
                  and g["created"] < hi and g.get("updated", g["created"]) + IDLE > lo}
        active.discard("untracked")
        label = ("players" if active == {"players"} else "Ryan" if active == {"Ryan"}
                 else "players and Ryan at once" if active else "no live Advanced game (tests, warm-ups, before tracking)")
        split[label] += dollars
    print(f"\nGPU cost, ${sum(cost.values()):.2f} in all:")
    for label, dollars in split.most_common():
        print(f"  ${dollars:6.2f}  {label}")


if __name__ == "__main__":
    main()
