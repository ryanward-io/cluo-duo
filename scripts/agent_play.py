"""Play one side of a game from the command line, the way a human plays on the site.

Lets a chat agent (or a person in a terminal) play using only its own side's view. Used to try
frontier models as players without an API. By default side "a" plays with the Basic partner
(GloVe v2) on side "b". With `--partner none`, two players (say, two agents) take a side each
and `wait` blocks until it's their move.

    uv run python scripts/agent_play.py new --seed 3 [--partner none]
    uv run python scripts/agent_play.py show GAME [--side b]
    uv run python scripts/agent_play.py wait GAME [--side b]
    uv run python scripts/agent_play.py clue GAME WORD NUMBER [--side b]
    uv run python scripts/agent_play.py guess GAME WORD [--side b]
    uv run python scripts/agent_play.py stop GAME [--side b]

Games are saved under data/agent-games/. Those files hold both keys: players must use only
these commands, never read the files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cluoduo import game
from cluoduo.players import PLAYERS, play_partner

GAMES = Path(__file__).resolve().parents[1] / "data" / "agent-games"
MARK = {game.GREEN: "AGENT", game.BYSTANDER: "bystander", game.ASSASSIN: "ASSASSIN"}


def load(game_id: str) -> dict:
    return json.loads((GAMES / f"{game_id}.json").read_text())


def save(state: dict) -> None:
    GAMES.mkdir(parents=True, exist_ok=True)
    (GAMES / f"{state['id']}.json").write_text(json.dumps(state))


def partner_moves(state: dict) -> dict:
    """The built-in partner (if any) plays side b until it's side a's move."""
    if state.get("partner", "glove-rank") == "none":
        return state
    return play_partner(state, "b", PLAYERS[state.get("partner", "glove-rank")](seed=state["seed"] + len(state["history"])))


def my_move(state: dict, side: str) -> bool:
    if state["phase"] == "over":
        return True
    if state["phase"] == "clue":
        return state["giver"] == side
    if state["phase"] == "sudden_death":
        return state["guesser"] == side
    return state["clue"]["giver"] != side


def show(state: dict, ME: str) -> None:
    PARTNER = game.other(ME)
    v = game.view(state, ME)
    print(f"Game {state['id']}  |  agents found {game.found_count(state)}/15  |  turns left {v['turns_left']}\n")
    print("Board (your key in brackets):")
    for i, w in enumerate(v["words"]):
        notes = []
        if v["found"][i]:
            notes.append("FOUND")
        if i in v["bystanders"][ME]:
            notes.append("known bystander on your key")
        if i in v["bystanders"][PARTNER]:
            notes.append("known bystander on partner's key")
        print(f"  {w:<12} [{MARK[v['my_key'][i]]}]" + (f"  ({'; '.join(notes)})" if notes else ""))
    print("\nHistory:")
    for h in v["history"]:
        who = "You" if h.get("side") == ME else "Partner"
        if h["type"] == "clue":
            print(f"  {who} gave {h['word']} {h['number']}")
        elif h["type"] == "guess":
            print(f"    {who} guessed {h['word']}: {dict(G='agent', N='bystander', A='ASSASSIN')[h['outcome']]}")
        elif h["type"] == "stop":
            print(f"    {who} stopped")
        elif h["type"] == "sudden_death":
            print("  Out of turns: SUDDEN DEATH")
    if not v["history"]:
        print("  (none)")
    print()
    if state["phase"] == "over":
        print(f"GAME OVER: {state['result']}")
    elif not my_move(state, ME):
        print(f"WAITING: it's your partner's move. Run:  wait {state['id']} --side {ME}")
    elif state["phase"] == "sudden_death":
        print("SUDDEN DEATH: the turns have run out. No more clues. YOUR MOVE: guess a word you think is one of")
        print("  your PARTNER's agents (judged by their key). A right guess finds it; any wrong guess loses the game.")
        print(f"  guess {state['id']} WORD --side {ME}")
    elif state["phase"] == "clue":
        print(f"YOUR MOVE: give a clue for your AGENT words:  clue {state['id']} WORD NUMBER --side {ME}")
    else:
        c = state["clue"]
        print(f"YOUR MOVE: your partner's clue is {c['word']} {c['number']} (you have made {c['guesses']} guess(es)).")
        print(f"  guess {state['id']} WORD --side {ME}   or, after at least one guess,   stop {state['id']} --side {ME}")
        print("  Guesses are judged by your PARTNER's key, which you can't see.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--seed", type=int, required=True)
    n.add_argument("--partner", choices=sorted(PLAYERS) + ["none"], default="glove-rank",
                   help="the AI on side b (any player in players.PLAYERS), or none for two agents")
    n.add_argument("--sudden-death", action="store_true", help="play on in sudden death when the turns run out")
    n.add_argument("--tag", default="", help="suffix for the game id, to replay the same board again (seed12-duo-v9)")
    n.add_argument("--turns", type=int, default=game.TURNS, help="9 (standard), 10 or 11")
    commands = {name: sub.add_parser(name) for name in ("show", "wait", "stop", "clue", "guess")}
    for c in commands.values():
        c.add_argument("game")
    commands["clue"].add_argument("word")
    commands["clue"].add_argument("number", type=int)
    commands["guess"].add_argument("word")
    for c in commands.values():
        c.add_argument("--side", choices=["a", "b"], default="a")
    args = p.parse_args()
    ME = getattr(args, "side", "a")

    if args.cmd == "new":
        # Same boards and first player as scripts/evaluate.py, so results line up with self-play.
        state = game.new_game(seed=args.seed, first_clue="a" if args.seed % 2 == 0 else "b",
                              game_id=f"seed{args.seed}" + ("-duo" if args.partner == "none" else "")
                              + (f"-{args.tag}" if args.tag else ""),
                              sudden_death=args.sudden_death, turns=args.turns)
        state["partner"] = args.partner
        state = partner_moves(state)
    elif args.cmd in ("show", "wait"):
        state = load(args.game)
        deadline = time.time() + 100  # under a tool call's usual 2-minute limit; just wait again
        while args.cmd == "wait" and not my_move(state, ME) and time.time() < deadline:
            time.sleep(2)
            state = load(args.game)
        show(state, ME)
        return
    else:
        state = load(args.game)
        if not my_move(state, ME):
            print("NOT ALLOWED: it's not your move.\n")
            show(state, ME)
            return
        before = len(state["history"])
        try:
            if args.cmd == "clue":
                state = game.give_clue(state, ME, args.word, args.number)
            elif args.cmd == "guess":
                index = {w: i for i, w in enumerate(state["words"])}.get(args.word.strip().upper())
                if index is None:
                    raise game.RuleError(f"{args.word.upper()} is not on the board.")
                move = game.sudden_death_guess if state["phase"] == "sudden_death" else game.guess
                state = move(state, ME, index)
            elif args.cmd == "stop":
                state = game.stop(state, ME)
        except game.RuleError as e:
            print(f"NOT ALLOWED: {e}\n")
        for h in state["history"][before:]:
            h["t"] = time.time()  # when each move was made, to time the players
        if state["phase"] != "over":
            state = partner_moves(state)
            for h in state["history"]:
                h.setdefault("t", time.time())  # the built-in partner's moves: when its turn ended
    state.setdefault("created", time.time())
    save(state)
    show(state, ME)


if __name__ == "__main__":
    main()
