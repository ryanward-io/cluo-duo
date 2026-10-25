"""Word-vector partners built on GloVe.

v1 ("glove-sum") is Ryan's first design:

- Clue: for each candidate clue word, add up its similarity to your own agents (only links
  at or above `link_threshold` count, since weak links can't be followed), then subtract
  a small penalty for similarity to bystanders and a large one for similarity to
  assassins. The highest total wins. The number is how many agents passed the threshold.
- Guess: rank the words you're allowed to guess by similarity to the clue and take them
  in order, up to the clue's number.

Similarity is cosine similarity between GloVe vectors: about 0 for unrelated words,
about 0.8 for very close ones (see scripts/prepare_vectors.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import game

DEFAULT_VECTORS = Path(__file__).resolve().parents[2] / "data" / "glove50k.npz"
CLUE_VOCAB = 20_000  # candidate clues: the most frequent real words in the vectors file


@dataclass(frozen=True)
class Vectors:
    words: list[str]
    index: dict[str, int]
    matrix: np.ndarray  # unit-length rows, float32, so matrix @ v is cosine similarity
    clue_ids: np.ndarray  # rows allowed as clues

    def ids(self, words: list[str]) -> np.ndarray:
        return np.array([self.index[w] for w in words])


@lru_cache(maxsize=1)
def load_vectors(path: str | None = None) -> Vectors:
    path = path or os.environ.get("CLUODUO_VECTORS") or str(DEFAULT_VECTORS)
    data = np.load(path)
    words = [str(w) for w in data["words"]]
    clue_ids = np.array([i for i, w in enumerate(words[:CLUE_VOCAB]) if len(w) >= 3])
    return Vectors(words, {w: i for i, w in enumerate(words)}, data["vectors"].astype(np.float32), clue_ids)


class BoardCache:
    """Similarities between every candidate clue and every board word, worked out once per board."""

    def __init__(self, vectors: Vectors):
        self.v = vectors
        self._board: tuple[str, ...] | None = None

    def get(self, board_words: list[str]) -> tuple[np.ndarray, np.ndarray]:
        board = tuple(board_words)
        if board != self._board:
            v = self.v
            board_vecs = v.matrix[v.ids(list(board))]
            self.sims = v.matrix[v.clue_ids] @ board_vecs.T  # (candidates, 25)
            # Clues can't be, contain, or be contained in a board word.
            self.legal = np.array([
                not any(c in b or b in c for b in board) for c in (v.words[i] for i in v.clue_ids)
            ])
            self._board = board
        return self.sims, self.legal


def open_cards(view: dict, giver: str) -> list[int]:
    """Cards still in play for a clue from `giver`: not found, and not already a known bystander on their key."""
    return [i for i in range(game.BOARD_SIZE) if not view["found"][i] and i not in view["bystanders"][giver]]


class GloveSumPlayer:
    name = "glove-sum"

    # Defaults tuned by self-play on boards 0-99, then checked on boards 10000-10299 (22% wins). See README.
    def __init__(self, seed: int | None = None, *, link_threshold: float = 0.35, agent_weight: float = 1.0,
                 bystander_weight: float = 0.25, assassin_weight: float = 1.0, vectors: Vectors | None = None):
        self.v = vectors or load_vectors()
        self.cache = BoardCache(self.v)
        self.link_threshold = link_threshold
        self.agent_weight = agent_weight
        self.bystander_weight = bystander_weight
        self.assassin_weight = assassin_weight

    # ---------- giving clues ----------

    def give_clue(self, view: dict) -> tuple[str, int]:
        sims, legal = self.cache.get(view["words"])
        me = view["side"]
        cards = open_cards(view, me)
        key = view["my_key"]
        agents = [i for i in cards if key[i] == game.GREEN]
        bystanders = [i for i in cards if key[i] == game.BYSTANDER]
        assassins = [i for i in cards if key[i] == game.ASSASSIN]

        agent_sims = sims[:, agents]
        strong = agent_sims >= self.link_threshold
        score = (
            self.agent_weight * np.where(strong, agent_sims, 0).sum(axis=1)
            - self.bystander_weight * np.clip(sims[:, bystanders], 0, None).sum(axis=1)
            - self.assassin_weight * np.clip(sims[:, assassins], 0, None).sum(axis=1)
        )
        score[~legal] = -np.inf
        # A clue that links no agent strongly is useless; fall back to the single closest agent.
        if strong[legal].any():
            score[~strong.any(axis=1)] = -np.inf
        else:
            score = np.where(legal, agent_sims.max(axis=1), -np.inf)
        best = int(np.argmax(score))
        number = max(1, int(strong[best].sum()))
        return self.v.words[self.v.clue_ids[best]], number

    # ---------- guessing ----------

    def next_guess(self, view: dict) -> int | None:
        clue = view["clue"]
        if clue["guesses"] >= max(clue["number"], 1):
            return None
        options = open_cards(view, clue["giver"])
        if not options:
            return None
        if clue["word"] not in self.v.index:
            return options[0]  # a clue word GloVe doesn't know: nothing better to go on
        clue_vec = self.v.matrix[self.v.index[clue["word"]]]
        board_vecs = self.v.matrix[self.v.ids([view["words"][i] for i in options])]
        return options[int(np.argmax(board_vecs @ clue_vec))]


# ---------- v1 + Bayesian guessing ----------

def known_giver_key(view: dict, giver: str) -> dict[int, str]:
    """What earlier turns revealed about `giver`'s key: every guess made on their clues shows that card's colour on their key."""
    known, current_giver = {}, None
    for h in view["history"]:
        if h["type"] == "clue":
            current_giver = h["side"]
        elif h["type"] == "guess" and current_giver == giver:
            known[h["index"]] = h["outcome"]
    return known


def key_prior(view: dict) -> dict[int, dict[str, float]]:
    """Prior P(agent / bystander / assassin on the clue giver's key) for each card I could guess.

    The rules fix how the two keys overlap (game.KEY_LAYOUT). Of my 9 agents, 3 are also my
    partner's agents, 5 are their bystanders and 1 is their assassin; of my 3 assassins, one
    each is their agent, bystander and assassin; and so on. Cards whose colour on the giver's
    key has already been revealed are taken out of those counts. Cards that left play without
    revealing it (found on my clues) are simply ignored: by symmetry, not knowing which one
    left doesn't change the odds for the rest.
    """
    giver, mine = view["clue"]["giver"], view["my_key"]
    counts = {m: {g: 0 for g in "GNA"} for m in "GNA"}
    for (a, b), n in game.KEY_LAYOUT.items():  # the layout is symmetric, so side order doesn't matter
        counts[a][b] += n
    for i, colour in known_giver_key(view, giver).items():
        counts[mine[i]][colour] -= 1
    prior = {}
    for i in open_cards(view, giver):
        c = counts[mine[i]]
        total = sum(c.values()) or 1
        prior[i] = {g: c[g] / total for g in "GNA"}
    return prior


class GloveBayesPlayer(GloveSumPlayer):
    """Same clues as v1; guesses combine GloVe similarity with what my own key implies about my partner's.

    For each card: prior from key_prior(), then a likelihood ratio from similarity,
    exp((similarity - mu) / tau), which multiplies the odds that the card is an agent.
    A card at similarity `mu` (roughly "unrelated") leaves the prior unchanged; every
    `tau` above it multiplies the agent odds by e. Then pick the card with the best
    P(agent) - assassin_cost * P(assassin), and stop early once that turns negative.
    """

    name = "glove-bayes"

    # Tuned on boards 0-99. On boards 20000-20999 it beat v1 23.7% to 22.1% (paired McNemar p = 0.026; scripts/compare.py).
    def __init__(self, seed: int | None = None, *, mu: float = 0.2, tau: float = 0.03,
                 assassin_cost: float = 1.0, **kwargs):
        super().__init__(seed, **kwargs)
        self.mu, self.tau, self.assassin_cost = mu, tau, assassin_cost

    def posterior(self, view: dict) -> dict[int, dict[str, float]]:
        clue = view["clue"]
        prior = key_prior(view)
        if clue["word"] not in self.v.index:
            return prior
        clue_vec = self.v.matrix[self.v.index[clue["word"]]]
        post = {}
        for i, p in prior.items():
            sim = float(self.v.matrix[self.v.index[view["words"][i]]] @ clue_vec)
            ratio = np.exp((sim - self.mu) / self.tau)
            z = p["G"] * ratio + p["N"] + p["A"]
            post[i] = {"G": p["G"] * ratio / z, "N": p["N"] / z, "A": p["A"] / z}
        return post

    def next_guess(self, view: dict) -> int | None:
        clue = view["clue"]
        if clue["guesses"] >= max(clue["number"], 1):
            return None
        post = self.posterior(view)
        if not post:
            return None
        value = {i: p["G"] - self.assassin_cost * p["A"] for i, p in post.items()}
        best = max(value, key=value.get)
        if clue["guesses"] > 0 and value[best] < 0:
            return None  # the next guess is more likely to hurt than help
        return best
