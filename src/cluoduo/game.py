"""Rules for a two-player co-operative word game in the Duet style.

Two sides, "a" and "b", share a 5x5 grid of words. Each side has its own secret key
marking every word as an agent (G), a bystander (N) or an assassin (A). On a turn,
one side gives a one-word clue and a number; the other side guesses words one at a
time. Guesses are judged by the clue giver's key:

- agent: the word is found and the guesser may keep going (or stop);
- bystander: the word is marked as a bystander for the clue giver and the turn ends;
- assassin: the game is lost.

The team wins by finding all 15 distinct agents within 9 turns. A side with no agents
left to find stops giving clues, so its partner gives the rest.

The state is a plain JSON-compatible dict so it can be stored anywhere and replayed.
"""

from __future__ import annotations

import random
import re
import secrets
from copy import deepcopy
from typing import Any, Literal

from .words import WORDS

Side = Literal["a", "b"]
SIDES: tuple[Side, Side] = ("a", "b")
GREEN, BYSTANDER, ASSASSIN = "G", "N", "A"
BOARD_SIZE = 25
TURNS = 9
MAX_CLUE_NUMBER = 9

# How the two keys overlap: (side a, side b) -> number of cards.
# Each side ends up with 9 agents, 3 assassins and 13 bystanders; 15 agents in total.
KEY_LAYOUT: dict[tuple[str, str], int] = {
    (GREEN, GREEN): 3,
    (GREEN, BYSTANDER): 5,
    (GREEN, ASSASSIN): 1,
    (BYSTANDER, GREEN): 5,
    (ASSASSIN, GREEN): 1,
    (ASSASSIN, ASSASSIN): 1,
    (ASSASSIN, BYSTANDER): 1,
    (BYSTANDER, ASSASSIN): 1,
    (BYSTANDER, BYSTANDER): 7,
}
assert sum(KEY_LAYOUT.values()) == BOARD_SIZE

State = dict[str, Any]


class RuleError(ValueError):
    """A move that the rules don't allow (shown to the player)."""


def other(side: Side) -> Side:
    return "b" if side == "a" else "a"


def new_game(seed: int | None = None, first_clue: Side = "a", game_id: str | None = None) -> State:
    seed = secrets.randbelow(2**31) if seed is None else seed
    rng = random.Random(seed)
    words = rng.sample(WORDS, BOARD_SIZE)
    pairs = [pair for pair, count in KEY_LAYOUT.items() for _ in range(count)]
    rng.shuffle(pairs)
    return {
        "id": game_id or secrets.token_urlsafe(9),
        "seed": seed,
        "words": words,
        "key": {"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]},
        "found": [False] * BOARD_SIZE,
        "bystanders": {"a": [], "b": []},  # indices marked as bystanders on that side's key
        "turns_left": TURNS,
        "giver": first_clue,
        "phase": "clue",  # clue -> guess -> clue ... -> over
        "clue": None,  # {"word", "number", "giver", "guesses"} while guessing
        "history": [],  # every clue and guess, for the UI, replays and evaluation
        "result": None,  # "won" | "assassin" | "out_of_turns"
    }


# ---------- derived facts ----------

def agents_left(state: State, side: Side) -> list[int]:
    key = state["key"][side]
    return [i for i in range(BOARD_SIZE) if key[i] == GREEN and not state["found"][i]]


def total_agents(state: State) -> int:
    return sum(1 for i in range(BOARD_SIZE) if GREEN in (state["key"]["a"][i], state["key"]["b"][i]))


def found_count(state: State) -> int:
    return sum(state["found"])


def can_guess(state: State, index: int) -> bool:
    """A word can be guessed if it isn't found and the clue giver hasn't already marked it a bystander."""
    giver = state["clue"]["giver"]
    return not state["found"][index] and index not in state["bystanders"][giver]


def guessable(state: State) -> list[int]:
    return [i for i in range(BOARD_SIZE) if can_guess(state, i)]


# ---------- clue validation ----------

def normalise_clue(word: str) -> str:
    return word.strip().upper()


def spelling_variant(a: str, b: str) -> bool:
    """One inserted or deleted letter apart, for words of 5+ letters: HARBOR/HARBOUR, COLOR/COLOUR.

    Swapped letters are allowed (HOUSE/HORSE are different words).
    """
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) != 1:
        return False
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def check_clue(state: State, word: str, number: int) -> str:
    """Return the cleaned clue word, or raise RuleError explaining what's wrong."""
    clue = normalise_clue(word)
    if not re.fullmatch(r"[A-Z]{2,30}", clue):
        raise RuleError("A clue must be a single word of letters only.")
    for board_word in state["words"]:
        if clue == board_word:
            raise RuleError(f"{clue} is on the board.")
        if clue in board_word or board_word in clue:
            raise RuleError(f"{clue} overlaps with {board_word} on the board.")
        if spelling_variant(clue, board_word):
            raise RuleError(f"{clue} is another spelling of {board_word} on the board.")
    if not isinstance(number, int) or not 0 <= number <= MAX_CLUE_NUMBER:
        raise RuleError(f"The number must be between 0 and {MAX_CLUE_NUMBER}.")
    return clue


# ---------- moves ----------

def give_clue(state: State, side: Side, word: str, number: int) -> State:
    if state["phase"] != "clue":
        raise RuleError("It isn't time to give a clue.")
    if side != state["giver"]:
        raise RuleError("It's your partner's turn to give a clue.")
    clue = check_clue(state, word, number)
    s = deepcopy(state)
    s["clue"] = {"word": clue, "number": number, "giver": side, "guesses": 0}
    s["phase"] = "guess"
    s["history"].append({"type": "clue", "side": side, "word": clue, "number": number, "turn": TURNS - s["turns_left"] + 1})
    return s


def guess(state: State, side: Side, index: int) -> State:
    if state["phase"] != "guess":
        raise RuleError("There's no clue to guess from.")
    giver = state["clue"]["giver"]
    if side != other(giver):
        raise RuleError("Only the clue giver's partner can guess.")
    if not 0 <= index < BOARD_SIZE or not can_guess(state, index):
        raise RuleError("That word can't be guessed now.")
    s = deepcopy(state)
    s["clue"]["guesses"] += 1
    outcome = s["key"][giver][index]
    s["history"].append({"type": "guess", "side": side, "index": index, "word": s["words"][index], "outcome": outcome})
    if outcome == ASSASSIN:
        return _finish(s, "assassin")
    if outcome == BYSTANDER:
        s["bystanders"][giver].append(index)
        return _end_turn(s)
    s["found"][index] = True
    if found_count(s) == total_agents(s):
        return _finish(s, "won")
    if not agents_left(s, giver):
        return _end_turn(s)  # nothing left for this clue to point at
    return s


def stop(state: State, side: Side) -> State:
    if state["phase"] != "guess" or side != other(state["clue"]["giver"]):
        raise RuleError("You can't stop guessing now.")
    if state["clue"]["guesses"] == 0:
        raise RuleError("Make at least one guess before stopping.")
    s = deepcopy(state)
    s["history"].append({"type": "stop", "side": side})
    return _end_turn(s)


def _end_turn(s: State) -> State:
    s["turns_left"] -= 1
    s["clue"] = None
    if s["turns_left"] <= 0:
        return _finish(s, "out_of_turns")
    nxt = other(s["giver"])
    s["giver"] = nxt if agents_left(s, nxt) else s["giver"]
    s["phase"] = "clue"
    return s


def _finish(s: State, result: str) -> State:
    s["phase"] = "over"
    s["result"] = result
    s["clue"] = None
    s["history"].append({"type": "end", "result": result})
    return s


# ---------- what each side is allowed to see ----------

def view(state: State, side: Side) -> dict[str, Any]:
    """Everything `side` may know. The partner's key is hidden until the game ends."""
    v = {k: deepcopy(state[k]) for k in
         ("id", "words", "found", "bystanders", "turns_left", "giver", "phase", "clue", "history", "result")}
    v["side"] = side
    v["my_key"] = list(state["key"][side])
    v["found_count"] = found_count(state)
    v["total_agents"] = total_agents(state)
    v["partner_key"] = list(state["key"][other(side)]) if state["phase"] == "over" else None
    return v
