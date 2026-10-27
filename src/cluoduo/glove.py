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
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import game
from .blocked import is_blocked

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
    clue_ids = np.array([i for i, w in enumerate(words[:CLUE_VOCAB]) if len(w) >= 3 and not is_blocked(w)])
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
            # Clues can't be, contain, or be contained in a board word, or be a spelling variant of one
            # (game.spelling_variant, done here with set lookups because there are 20,000 candidates).
            words = set(board)
            shortened = {b[:i] + b[i + 1:] for b in board if len(b) >= 6 for i in range(len(b))}
            self.legal = np.array([
                not (any(c in b or b in c for b in board)
                     or (len(c) >= 5 and c in shortened)
                     or (len(c) >= 6 and any(c[:i] + c[i + 1:] in words for i in range(len(c)))))
                for c in (v.words[i] for i in v.clue_ids)
            ])
            self._board = board
            self.position = {v.words[i]: n for n, i in enumerate(v.clue_ids)}
        return self.sims, self.legal

    def used(self, view: dict) -> np.ndarray:
        """Mask of candidate clues already given this game."""
        mask = np.zeros(len(self.v.clue_ids), dtype=bool)
        for word in used_clues(view):
            if word in self.position:
                mask[self.position[word]] = True
        return mask


def unfinished_clues(view: dict) -> list[str]:
    """Earlier clues from the current clue's giver that found fewer agents than their number."""
    giver, current = view["clue"]["giver"], len([h for h in view["history"] if h["type"] == "clue"])
    out, clue, found, n = [], None, 0, 0
    for h in view["history"]:
        if h["type"] == "clue":
            if clue and clue["side"] == giver and found < clue["number"]:
                out.append(clue["word"])
            clue, found, n = h, 0, n + 1
        elif h["type"] == "guess" and h["outcome"] == game.GREEN:
            found += 1
    return out  # the current clue (the last one) is never included


def used_clues(view: dict) -> set[str]:
    """Every clue word given so far this game, by either player."""
    return {h["word"] for h in view["history"] if h["type"] == "clue"}


def open_cards(view: dict, giver: str) -> list[int]:
    """Cards still in play for a clue from `giver`: not found, and not already a known bystander on their key."""
    return [i for i in range(game.BOARD_SIZE) if not view["found"][i] and i not in view["bystanders"][giver]]


class GloveSumPlayer:
    name = "glove-sum"

    # Defaults tuned by self-play on boards 0-99, then checked on boards 10000-10299 (22% wins). See README.
    def __init__(self, seed: int | None = None, *, link_threshold: float = 0.35, agent_weight: float = 1.0,
                 bystander_weight: float = 0.25, assassin_weight: float = 1.0, guess_noise: float = 0.0,
                 vectors: Vectors | None = None):
        self.v = vectors or load_vectors()
        self.cache = BoardCache(self.v)
        self.link_threshold = link_threshold
        self.agent_weight = agent_weight
        self.bystander_weight = bystander_weight
        self.assassin_weight = assassin_weight
        # Evaluation only: blur the guesser's similarities so it no longer shares the clue-giver's
        # exact associations (a stand-in for a human or a different model). 0 in the live game.
        self.guess_noise = guess_noise
        self.seed = seed or 0

    # ---------- giving clues ----------

    def give_clue(self, view: dict) -> tuple[str, int]:
        sims, legal = self.cache.get(view["words"])
        legal = legal & ~self.cache.used(view)  # never repeat a clue already given this game
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

    def clue_similarities(self, view: dict, cards: list[int]) -> np.ndarray:
        """Similarity of the current clue to each card, plus this guesser's fixed noise (if any)."""
        clue_vec = self.v.matrix[self.v.index[view["clue"]["word"]]]
        sims = self.v.matrix[self.v.ids([view["words"][i] for i in cards])] @ clue_vec
        if self.guess_noise:
            # Same clue on the same board always gets the same noise, so a turn's guesses are consistent.
            tag = f"{self.seed}|{view['clue']['word']}|{','.join(view['words'])}".encode()
            rng = np.random.default_rng(zlib.crc32(tag))
            sims = sims + self.guess_noise * rng.standard_normal(game.BOARD_SIZE)[cards]
        return sims

    def sudden_death_guess(self, view: dict) -> int:
        return sudden_death_guess(self, view)

    def next_guess(self, view: dict) -> int | None:
        clue = view["clue"]
        if clue["guesses"] >= max(clue["number"], 1):
            return None
        options = open_cards(view, clue["giver"])
        if not options:
            return None
        if clue["word"] not in self.v.index:
            return options[0]  # a clue word GloVe doesn't know: nothing better to go on
        return options[int(np.argmax(self.clue_similarities(view, options)))]


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
                 assassin_cost: float = 1.0, leftover_min: float = 1.0, **kwargs):
        super().__init__(seed, **kwargs)
        self.mu, self.tau, self.assassin_cost = mu, tau, assassin_cost
        self.leftover_min = leftover_min  # 1.0 = never use earlier clues (see leftover_guess)

    def posterior(self, view: dict) -> dict[int, dict[str, float]]:
        clue = view["clue"]
        prior = key_prior(view)
        if clue["word"] not in self.v.index:
            return prior
        cards = list(prior)
        sims = dict(zip(cards, self.clue_similarities(view, cards)))
        post = {}
        for i, p in prior.items():
            ratio = np.exp((sims[i] - self.mu) / self.tau)
            z = p["G"] * ratio + p["N"] + p["A"]
            post[i] = {"G": p["G"] * ratio / z, "N": p["N"] / z, "A": p["A"] / z}
        return post

    def next_guess(self, view: dict) -> int | None:
        clue = view["clue"]
        if clue["guesses"] >= max(clue["number"], 1):
            return self.leftover_guess(view) if clue["guesses"] == max(clue["number"], 1) else None
        post = self.posterior(view)
        if not post:
            return None
        value = {i: p["G"] - self.assassin_cost * p["A"] for i, p in post.items()}
        best = max(value, key=value.get)
        if clue["guesses"] > 0 and value[best] < 0:
            return None  # the next guess is more likely to hurt than help
        return best

    def leftover_guess(self, view: dict) -> int | None:
        """One extra guess, after using up the current clue, for a word an earlier clue left behind.

        An earlier clue from the same partner that found fewer agents than its number (it hit a
        bystander, or the guesser stopped) still points at words. Score every card under each
        such clue with the same posterior as a normal guess, and take the best only if it is
        clearly an agent (P(agent) at least `leftover_min`).
        """
        if self.leftover_min >= 1:
            return None
        best, best_p = None, self.leftover_min
        for word in unfinished_clues(view):
            if word not in self.v.index:
                continue
            post = self.posterior(dict(view, clue=dict(view["clue"], word=word)))
            for i, p in post.items():
                if p["G"] - self.assassin_cost * p["A"] > 0 and p["G"] >= best_p:
                    best, best_p = i, p["G"]
        return best


# ---------- sudden death ----------

def sudden_death_guess(player, view: dict) -> int:
    """No clue to go on: pick the word most likely to be one of my partner's agents.

    For each card, the Bayesian posterior (see GloveBayesPlayer) under each of my partner's
    earlier clues, keeping the best; with no usable clue, the prior from my own key alone.
    Any wrong guess loses now, so only P(agent) matters.
    """
    partner = game.other(view["side"])
    mu, tau = getattr(player, "mu", 0.2), getattr(player, "tau", 0.03)
    words = [h["word"] for h in view["history"] if h["type"] == "clue" and h["side"] == partner]
    best: dict[int, float] = {}
    for word in words or [None]:
        v = dict(view, clue={"word": word, "number": 1, "giver": partner, "guesses": 0})
        prior = key_prior(v)
        sims = dict(zip(prior, player.clue_similarities(v, list(prior)))) if word in player.v.index else {}
        for i, p in prior.items():
            ratio = np.exp((sims[i] - mu) / tau) if i in sims else 1.0
            g = p["G"] * ratio / (p["G"] * ratio + p["N"] + p["A"])
            best[i] = max(best.get(i, 0.0), g)
    return max(best, key=best.get)


# ---------- v2 clues: rankings instead of sums ----------

class GloveRankPlayer(GloveBayesPlayer):
    """v2 clues plus the Bayesian guesser.

    The partner guesses words in order of similarity to the clue, so what matters is the
    *ranking*: a clue for k agents works if those k agents are all more similar to the clue
    than every bystander and assassin still in play, by a safety margin. For each candidate
    clue, take my agents from most to least similar; the clue can cover the top k if the
    k-th agent is at least `link_threshold` similar, beats the closest bystander by
    `bystander_margin` and the closest assassin by `assassin_margin`. Choose the clue that
    covers the most agents; break ties by the widest margin.
    """

    name = "glove-rank"

    # Tuned on boards 0-99 against a *noisy* guesser (guess_noise 0.05 and 0.1), because with
    # identical vectors on both sides almost any setting wins (the "private language" problem).
    # Defaults are tuned for a partner who is NOT GloVe (a person), measured with a Claude
    # reference partner on shared boards, not for self-play:
    # - max_number 3: big clues (DEER 5) only work when both sides share the vectors. Capping
    #   costs self-play (77% -> 73%) but did better with a real partner.
    # - leftover_min 1.0 (off): leftover guesses won in self-play (79.2% vs 76.3%, p = 0.001) but
    #   cost a game with a real partner. Set 0.7 to switch them on.
    def __init__(self, seed: int | None = None, *, bystander_margin: float = 0.05,
                 assassin_margin: float = 0.1, link_threshold: float = 0.3,
                 stretch_slack: float = 0.0, stretch_assassin_margin: float = 0.2,
                 leftover_min: float = 1.0, max_number: int = 3, **kwargs):
        super().__init__(seed, link_threshold=link_threshold, leftover_min=leftover_min, **kwargs)
        # Self-play rewards big numbers (DEER 5) because both sides share the vectors; a human
        # (or another model) can't follow them. max_number caps the clue size.
        self.max_number = int(max_number)
        self.bystander_margin, self.assassin_margin = bystander_margin, assassin_margin
        # Stretch clues (Ryan's idea): when every assassin is far from the clue (by
        # stretch_assassin_margin), the weakest target may trail the closest bystander by up to
        # stretch_slack. The worst case is then a bystander, which only ends the turn. Off by
        # default: in self-play it cost wins (a GloVe guesser takes the closer bystander first).
        self.stretch_slack, self.stretch_assassin_margin = stretch_slack, stretch_assassin_margin

    def give_clue(self, view: dict) -> tuple[str, int]:
        sims, legal = self.cache.get(view["words"])
        legal = legal & ~self.cache.used(view)  # never repeat a clue already given this game
        key = view["my_key"]
        cards = open_cards(view, view["side"])
        agents = [i for i in cards if key[i] == game.GREEN]
        worst_bystander = sims[:, [i for i in cards if key[i] == game.BYSTANDER] or [0]].max(axis=1)
        if not any(key[i] == game.BYSTANDER for i in cards):
            worst_bystander = np.full(len(sims), -1.0)
        assassin_ids = [i for i in cards if key[i] == game.ASSASSIN]
        worst_assassin = sims[:, assassin_ids].max(axis=1) if assassin_ids else np.full(len(sims), -1.0)

        ranked = -np.sort(-sims[:, agents], axis=1)  # each clue's agent similarities, highest first
        best, best_k, best_margin = None, 0, -np.inf
        for k in range(min(len(agents), self.max_number), 0, -1):
            kth = ranked[:, k - 1]
            margin = np.minimum(kth - worst_bystander - self.bystander_margin,
                                kth - worst_assassin - self.assassin_margin)
            if self.stretch_slack:
                far_from_assassins = kth - worst_assassin >= self.stretch_assassin_margin
                stretch = np.minimum(kth - worst_bystander - self.bystander_margin + self.stretch_slack,
                                     kth - worst_assassin - self.assassin_margin)
                margin = np.where(far_from_assassins, np.maximum(margin, stretch), margin)
            ok = legal & (kth >= self.link_threshold) & (margin >= 0)
            if ok.any():
                margins = np.where(ok, margin, -np.inf)
                best, best_k, best_margin = int(np.argmax(margins)), k, float(margins.max())
                break
        if best is None:
            # Nothing passes the rules: give the safest single-agent clue available.
            margin = np.minimum(ranked[:, 0] - worst_bystander, ranked[:, 0] - worst_assassin)
            best, best_k = int(np.argmax(np.where(legal, margin, -np.inf))), 1
        return self.v.words[self.v.clue_ids[best]], best_k
