"""LLM partner: an open-source model (Qwen3-8B on Modal, see llm_server.py) plays the game from a text prompt.

Design:
- One model call per AI turn. A clue call returns the target words, a short reason and the
  clue; the number is how many targets it named. A guess call returns the words to guess in
  order (as many as it wants to risk).
- Replies are forced into a JSON schema built for the current board, so the model can only
  name its own unfound agents as targets (1-3 of them) and only guess words still in play.
  The first version let it say anything and it gave clues like "SHAPE 9".
- The guesser is not shown its own key: with it, the model guessed its *own* agents instead of
  following the clue. (The key carries a little information about the partner's, which the
  GloVe Bayesian guesser uses, but the model couldn't use it well.)
- The prompt includes the whole game so far (every clue and guess), so the model "remembers"
  earlier clues, e.g. a repeated clue that hasn't been fully used yet.
- Everything the model says is checked against the rules here. An invalid clue goes back to
  the model with the reason (up to MAX_ATTEMPTS); after that, the GloVe partner covers the turn.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

from . import game

MAX_ATTEMPTS = 3
MAX_TARGETS = 3  # bigger clues were nearly always reckless
CLUE_WORD = {"type": "string", "pattern": "^[A-Za-z]{2,30}$"}  # one word, no number or spaces
THINKING_TOKENS = 6000  # reply budget with thinking on: the reasoning counts towards it
COLOUR = {game.GREEN: "AGENT", game.BYSTANDER: "bystander", game.ASSASSIN: "ASSASSIN"}

Backend = Callable[..., dict]  # chat(messages, schema=..., seed=..., thinking=...) -> {"content": str, ...}

RULES = """You are playing a co-operative two-player word game (like Codenames Duet) with a human partner.
- There are 25 words on the board. You and your partner each have a secret key marking every word as an
  AGENT, a bystander or an ASSASSIN. The keys overlap but are different. You can see only your own key.
- On a clue turn, the clue giver says ONE word plus a number: how many of their own AGENT words the clue
  points to. The partner then guesses words one at a time.
- A guess is judged by the clue giver's key: an AGENT is found and the guesser may continue; a bystander
  ends the turn; an ASSASSIN loses the game for both players.
- A clue must be a single real English word. It must not be a word on the board, contain one, or be
  contained in one (so no plurals or parts of board words).
- The team wins by finding all 15 agents within 9 turns, so clues that safely cover 2 or 3 words matter.
"""

def clue_schema(agents: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "targets": {"type": "array", "items": {"enum": agents}, "minItems": 1, "maxItems": min(MAX_TARGETS, len(agents)),
                        "description": "1 to 3 of your AGENT words that share one strong, specific link"},
            "reason": {"type": "string", "description": "one short sentence: the link, and why it doesn't point at an ASSASSIN or bystander"},
            "clue": CLUE_WORD,
        },
        "required": ["targets", "reason", "clue"],
        "additionalProperties": False,
    }


def guess_schema(open_words: list[str], number: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "one short sentence on what the clue most likely means"},
            "guesses": {"type": "array", "items": {"enum": open_words}, "minItems": 1,
                        "maxItems": min(len(open_words), max(number, 1) + 1),
                        "description": "board words to guess, most confident first; only as many as are worth the risk"},
        },
        "required": ["reason", "guesses"],
        "additionalProperties": False,
    }


def describe_history(view: dict) -> str:
    me = view["side"]
    who = lambda side: "You" if side == me else "Your partner"
    lines, turn = [], 0
    for h in view["history"]:
        if h["type"] == "clue":
            turn += 1
            lines.append(f"Turn {turn}: {who(h['side'])} gave the clue {h['word']} {h['number']}.")
        elif h["type"] == "guess":
            outcome = {"G": "an agent", "N": "a bystander", "A": "an ASSASSIN"}[h["outcome"]]
            lines.append(f"  {who(h['side'])} guessed {h['word']}: {outcome} on the clue giver's key.")
        elif h["type"] == "stop":
            lines.append(f"  {who(h['side'])} stopped guessing.")
    return "\n".join(lines) or "(no turns yet)"


def describe_board(view: dict, show_key: bool) -> str:
    me, partner = view["side"], game.other(view["side"])
    rows = []
    for i, word in enumerate(view["words"]):
        notes = []
        if view["found"][i]:
            notes.append("already found (out of play)")
        if i in view["bystanders"][me]:
            notes.append("known bystander on YOUR key")
        if i in view["bystanders"][partner]:
            notes.append("known bystander on your PARTNER's key")
        key = f" [your key: {COLOUR[view['my_key'][i]]}]" if show_key else ""
        rows.append(f"- {word}{key}" + (f" ({'; '.join(notes)})" if notes else ""))
    return "\n".join(rows)


class LLMPlayer:
    name = "qwen"

    def __init__(self, seed: int | None = None, *, backend: Backend | None = None, server: str = "LLM",
                 thinking: bool = False, fallback: Any = None, **_ignored):
        self.backend = backend or modal_backend(server)
        self.thinking = thinking
        self.seed = seed or 0
        self._fallback = fallback
        self._plan: tuple[tuple, list[int]] | None = None  # (turn key, remaining guesses) for the current clue
        self.log: list[dict] = []  # every call: prompt, reply, problems (for evaluation and debugging)

    @property
    def fallback(self):
        if self._fallback is None:
            from .glove import GloveRankPlayer

            self._fallback = GloveRankPlayer(self.seed)
        return self._fallback

    def _ask(self, messages: list[dict], schema: dict) -> dict:
        reply = self.backend(messages, schema=schema, thinking=self.thinking, seed=self.seed + len(self.log),
                             max_tokens=THINKING_TOKENS if self.thinking else 600)
        try:
            data = json.loads(reply["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            data = {}
        self.log.append({"messages": messages, "reply": reply, "parsed": data})
        return data

    # ---------- giving clues ----------

    def give_clue(self, view: dict) -> tuple[str, int]:
        agents = self._agents(view)
        messages = [
            {"role": "system", "content": RULES},
            {"role": "user", "content": (
                f"Board (with your key):\n{describe_board(view, show_key=True)}\n\n"
                f"Game so far:\n{describe_history(view)}\n\n"
                f"Your AGENT words still to find: {', '.join(agents)}.\n"
                f"Turns left: {view['turns_left']}. It's your turn to give a clue. Choose 1 to 3 of your AGENT words "
                "that share one strong, specific link, then one clue word for that link. Your partner will guess the "
                "words closest to the clue, so check that no ASSASSIN or bystander is as close to it as your targets. "
                "Two or three safe targets is good; one obvious target beats a vague clue."
            )},
        ]
        for _ in range(MAX_ATTEMPTS):
            data = self._ask(messages, clue_schema(agents))
            problem = self._clue_problem(view, data, agents)
            if problem is None:
                targets = {t.upper() for t in data["targets"]} & set(agents)
                return game.normalise_clue(data["clue"]), min(len(targets), MAX_TARGETS)
            self.log[-1]["problem"] = problem
            messages = messages + [
                {"role": "assistant", "content": json.dumps(data)},
                {"role": "user", "content": f"That clue isn't allowed: {problem} Try again."},
            ]
        self.log.append({"fallback": "clue"})
        return self.fallback.give_clue(view)

    @staticmethod
    def _agents(view: dict) -> list[str]:
        return [w for i, w in enumerate(view["words"]) if view["my_key"][i] == game.GREEN and not view["found"][i]]

    def _clue_problem(self, view: dict, data: dict, agents: list[str]) -> str | None:
        if not data.get("clue"):
            return "The reply had no clue."
        try:
            game.check_clue(view, data["clue"], 1)  # only reads view["words"]
        except game.RuleError as e:
            return str(e)
        if not is_real_word(data["clue"]):
            return f"{data['clue'].upper()} isn't a common English word."
        if not any(str(t).upper() in agents for t in data.get("targets", [])):
            return f"None of the targets are your unfound AGENT words ({', '.join(agents)})."
        return None

    # ---------- guessing ----------

    def next_guess(self, view: dict) -> int | None:
        clue = view["clue"]
        turn = (clue["word"], clue["number"], len([h for h in view["history"] if h["type"] == "clue"]))
        if self._plan is None or self._plan[0] != turn:
            self._plan = (turn, self._plan_guesses(view))
        remaining = [i for i in self._plan[1] if game.can_guess(view, i)]
        self._plan = (turn, remaining[1:])
        if remaining:
            return remaining[0]
        return None if clue["guesses"] > 0 else self.fallback.next_guess(view)

    def _plan_guesses(self, view: dict) -> list[int]:
        clue = view["clue"]
        open_words = [w for i, w in enumerate(view["words"]) if game.can_guess(view, i)]
        messages = [
            {"role": "system", "content": RULES},
            {"role": "user", "content": (
                f"Board:\n{describe_board(view, show_key=False)}\n\n"
                f"Game so far:\n{describe_history(view)}\n\n"
                f"Your partner's clue is {clue['word']} {clue['number']}: they are pointing at {clue['number']} word(s) "
                "that are AGENTS on THEIR key. Words you can guess: " + ", ".join(open_words) + ".\n"
                "Pick the words most strongly linked to the clue, most confident first. Usually guess exactly the "
                "number given; add one more only if an earlier clue from your partner clearly still points at a word. "
                "A wrong guess can hit an ASSASSIN, so leave out words you are unsure of."
            )},
        ]
        data = self._ask(messages, guess_schema(open_words, clue["number"]))
        index = {w: i for i, w in enumerate(view["words"])}
        plan = []
        for w in data.get("guesses", []):
            i = index.get(str(w).strip().upper())
            if i is not None and i not in plan:
                plan.append(i)
        return plan[:max(clue["number"], 1) + 1]


class HybridPlayer(LLMPlayer):
    """The model proposes, GloVe vets.

    The LLM knows things word vectors don't (a penguin is a bird), but it can't measure how
    close a clue sits to the words it must avoid. GloVe can. So for a clue:

    1. The model proposes up to CANDIDATES clues, each with the agents it is meant to link.
    2. Each legal candidate is checked with GloVe against the cards still in play. Its targets
       are sorted by similarity to the clue; it covers the top k of them if the k-th is closer
       to the clue than every assassin by `assassin_margin` and every bystander by
       `bystander_margin`. Targets that don't clear that bar are dropped, not the whole clue.
    3. The candidate covering the most targets wins (ties: the widest safety margin). If none
       covers any, the GloVe v2 partner gives the clue instead.

    There is no minimum similarity for the targets themselves: deciding what links is the
    model's job. GloVe only says "this clue is no closer to your agents than to a danger word".
    Guessing is the same as the plain LLM partner.
    """

    name = "qwen-hybrid"
    CANDIDATES = 8

    def __init__(self, seed: int | None = None, *, assassin_margin: float = 0.1,
                 bystander_margin: float = 0.0, **kwargs):
        super().__init__(seed, **kwargs)
        self.assassin_margin, self.bystander_margin = assassin_margin, bystander_margin

    def give_clue(self, view: dict) -> tuple[str, int]:
        agents = self._agents(view)
        messages = [
            {"role": "system", "content": RULES},
            {"role": "user", "content": (
                f"Board (with your key):\n{describe_board(view, show_key=True)}\n\n"
                f"Game so far:\n{describe_history(view)}\n\n"
                f"Your AGENT words still to find: {', '.join(agents)}.\n"
                f"Turns left: {view['turns_left']}. It's your turn to give a clue. Brainstorm up to {self.CANDIDATES} "
                "different candidate clues. For each, name 1 to 3 of your AGENT words that share one strong, specific "
                "link, and the clue word for it. Vary them: different groups of agents, different kinds of link. "
                "Avoid clues that also suggest an ASSASSIN or bystander. A separate check will pick the safest."
            )},
        ]
        data = self._ask(messages, candidates_schema(agents, self.CANDIDATES))
        scored = [self.vet(view, c) for c in data.get("candidates", [])]
        scored = [r for r in scored if r["number"] > 0]
        self.log[-1]["vetted"] = scored
        if not scored:
            self.log.append({"fallback": "clue"})
            return self.fallback.give_clue(view)
        best = max(scored, key=lambda r: (r["number"], r["margin"]))
        return best["clue"], best["number"]

    def vet(self, view: dict, candidate: dict) -> dict:
        """GloVe safety check for one proposed clue: how many of its targets it can safely cover."""
        clue = game.normalise_clue(str(candidate.get("clue", "")))
        result = {"clue": clue, "targets": candidate.get("targets", []), "number": 0, "margin": float("-inf")}
        try:
            game.check_clue(view, clue, 1)
        except game.RuleError as e:
            result["problem"] = str(e)
            return result
        v = self.fallback.v  # the GloVe partner's vectors
        if clue not in v.index:
            result["problem"] = "not in the GloVe vocabulary"
            return result
        from .glove import open_cards

        cards = open_cards(view, view["side"])
        sims = v.matrix[v.ids([view["words"][i] for i in cards])] @ v.matrix[v.index[clue]]
        sim = dict(zip(cards, sims.tolist()))
        key = view["my_key"]
        danger = lambda colour: max((sim[i] for i in cards if key[i] == colour), default=-1.0)
        worst_assassin, worst_bystander = danger(game.ASSASSIN), danger(game.BYSTANDER)
        index = {w: i for i, w in enumerate(view["words"])}
        targets = sorted({index[t] for t in map(str.upper, map(str, result["targets"])) if t in index and index[t] in sim
                          and key[index[t]] == game.GREEN}, key=sim.get, reverse=True)
        for k in range(len(targets), 0, -1):
            kth = sim[targets[k - 1]]
            margin = min(kth - worst_assassin - self.assassin_margin, kth - worst_bystander - self.bystander_margin)
            if margin >= 0:
                result.update(number=k, margin=margin, kept=[view["words"][i] for i in targets[:k]])
                break
        return result


def candidates_schema(agents: list[str], n: int) -> dict:
    one = {
        "type": "object",
        "properties": {
            "targets": {"type": "array", "items": {"enum": agents}, "minItems": 1, "maxItems": min(MAX_TARGETS, len(agents))},
            "clue": CLUE_WORD,
        },
        "required": ["targets", "clue"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"candidates": {"type": "array", "items": one, "minItems": 1, "maxItems": n}},
        "required": ["candidates"],
        "additionalProperties": False,
    }


_REAL_WORDS: set[str] | None = None


def is_real_word(word: str) -> bool:
    global _REAL_WORDS
    if _REAL_WORDS is None:
        from .glove import load_vectors

        _REAL_WORDS = set(load_vectors().words)
    return word.strip().upper() in _REAL_WORDS


@lru_cache(maxsize=None)
def modal_backend(server: str = "LLM") -> Backend:
    """Calls a private model server (a class in llm_server.py) through Modal; needs Modal credentials.

    One shared handle per server for the whole process: separate handles per thread were
    observed to send requests one at a time, which made parallel evaluation as slow as serial.
    """
    import modal

    llm = modal.Cls.from_name("cluo-duo-llm", server)()

    def chat(messages, **kwargs):
        return llm.chat.remote(messages, **kwargs)

    return chat
