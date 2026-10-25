"""AI players plug in here. Each player only ever sees `game.view(state, side)`.

A player needs two methods:

- `give_clue(view) -> (word, number)`: a clue for the agents on its own key;
- `next_guess(view) -> index | None`: the next word to guess for the current clue,
  or None to stop (only allowed after at least one guess).

`play_clue` and `play_guesses` drive a player through the rules, so the web API and
the self-play evaluation use exactly the same code path.
"""

from __future__ import annotations

import random
from typing import Any, Protocol

from . import game
from .words import WORDS

View = dict[str, Any]


class Player(Protocol):
    name: str

    def give_clue(self, view: View) -> tuple[str, int]: ...

    def next_guess(self, view: View) -> int | None: ...


class RandomPlayer:
    """Placeholder partner: clues are random words and guesses are random. Useful for testing the plumbing."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def give_clue(self, view: View) -> tuple[str, int]:
        board = set(view["words"])
        options = [w for w in WORDS if w not in board and not any(w in b or b in w for b in board)]
        return self.rng.choice(options), 1

    def next_guess(self, view: View) -> int | None:
        clue = view["clue"]
        if clue["guesses"] >= max(clue["number"], 1):
            return None
        giver = clue["giver"]
        options = [i for i in range(game.BOARD_SIZE)
                   if not view["found"][i] and i not in view["bystanders"][giver]]
        return self.rng.choice(options) if options else None


def _glove_sum(seed=None, **kwargs):
    from .glove import GloveSumPlayer  # imported lazily: loads the word vectors

    return GloveSumPlayer(seed, **kwargs)


def _glove_bayes(seed=None, **kwargs):
    from .glove import GloveBayesPlayer

    return GloveBayesPlayer(seed, **kwargs)


PLAYERS = {"random": RandomPlayer, "glove-sum": _glove_sum, "glove-bayes": _glove_bayes}

MAX_CLUE_ATTEMPTS = 3


def play_clue(state: game.State, side: game.Side, player: Player) -> game.State:
    """Ask `player` for a clue, retrying if it breaks the clue rules."""
    error = None
    for _ in range(MAX_CLUE_ATTEMPTS):
        word, number = player.give_clue(game.view(state, side))
        try:
            return game.give_clue(state, side, word, number)
        except game.RuleError as e:
            error = e
    raise RuntimeError(f"{player.name} gave {MAX_CLUE_ATTEMPTS} invalid clues; last error: {error}")


def play_guesses(state: game.State, side: game.Side, player: Player) -> game.State:
    """Let `player` guess until the turn ends, it stops, or it picks something illegal (treated as stopping)."""
    while state["phase"] == "guess" and state["clue"]["giver"] != side:
        choice = player.next_guess(game.view(state, side))
        if choice is None or not 0 <= choice < game.BOARD_SIZE or not game.can_guess(state, choice):
            if state["clue"]["guesses"] == 0:
                options = game.guessable(state)  # must guess at least once; fall back to a random legal word
                choice = random.Random(state["seed"] + len(state["history"])).choice(options)
            else:
                return game.stop(state, side)
        state = game.guess(state, side, choice)
    return state


def play_partner(state: game.State, side: game.Side, player: Player) -> game.State:
    """Run the AI side until it's the human's move (or the game ends)."""
    while state["phase"] != "over":
        if state["phase"] == "clue" and state["giver"] == side:
            state = play_clue(state, side, player)
        elif state["phase"] == "guess" and state["clue"]["giver"] != side:
            state = play_guesses(state, side, player)
        else:
            break
    return state


def self_play(clue_a: Player, clue_b: Player, seed: int, first: game.Side = "a") -> game.State:
    """Two AI players on one board, for evaluation. Side a uses clue_a for everything it does, and so on."""
    state = game.new_game(seed=seed, first_clue=first)
    players = {"a": clue_a, "b": clue_b}
    while state["phase"] != "over":
        if state["phase"] == "clue":
            state = play_clue(state, state["giver"], players[state["giver"]])
        else:
            guesser = game.other(state["clue"]["giver"])
            state = play_guesses(state, guesser, players[guesser])
    return state
