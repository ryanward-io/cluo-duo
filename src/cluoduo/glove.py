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
