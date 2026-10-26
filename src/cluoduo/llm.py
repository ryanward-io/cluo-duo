"""LLM partner: an open-source model (Qwen3-8B on Modal, see llm_server.py) plays the game from a text prompt.

Design:
- One model call per AI turn. A clue call returns the target words, a short reason, the clue
  and the number; a guess call returns the words to guess in order (as many as it wants to
  risk). Replies are forced into a JSON schema by the server, so they always parse.
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

CLUE_SCHEMA = {
    "type": "object",
    "properties": {
        "targets": {"type": "array", "items": {"type": "string"}, "description": "your AGENT words the clue points to"},
        "reason": {"type": "string", "description": "one short sentence: why these words, and why the clue avoids your assassins"},
        "clue": {"type": "string"},
        "number": {"type": "integer"},
    },
    "required": ["targets", "reason", "clue", "number"],
    "additionalProperties": False,
}

GUESS_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "one short sentence on what the clue most likely means"},
        "guesses": {"type": "array", "items": {"type": "string"},
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

    def __init__(self, seed: int | None = None, *, backend: Backend | None = None, thinking: bool = False,
                 fallback: Any = None, **_ignored):
        self.backend = backend or modal_backend()
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
        reply = self.backend(messages, schema=schema, thinking=self.thinking, seed=self.seed + len(self.log))
        try:
            data = json.loads(reply["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            data = {}
        self.log.append({"messages": messages, "reply": reply, "parsed": data})
        return data

    # ---------- giving clues ----------

    def give_clue(self, view: dict) -> tuple[str, int]:
        agents = [w for i, w in enumerate(view["words"]) if view["my_key"][i] == game.GREEN and not view["found"][i]]
        messages = [
            {"role": "system", "content": RULES},
            {"role": "user", "content": (
                f"Board (with your key):\n{describe_board(view, show_key=True)}\n\n"
                f"Game so far:\n{describe_history(view)}\n\n"
                f"Your AGENT words still to find: {', '.join(agents)}.\n"
                f"Turns left: {view['turns_left']}. It's your turn to give a clue. Pick the AGENT words to target, "
                "then one clue word that links them strongly and does not suggest any ASSASSIN or bystander word."
            )},
        ]
        for _ in range(MAX_ATTEMPTS):
            data = self._ask(messages, CLUE_SCHEMA)
            problem = self._clue_problem(view, data, agents)
            if problem is None:
                targets = [t.upper() for t in data["targets"] if t.upper() in agents]
                number = max(1, min(int(data["number"]), len(agents), len(targets) or 1))
                return game.normalise_clue(data["clue"]), number
            self.log[-1]["problem"] = problem
            messages = messages + [
                {"role": "assistant", "content": json.dumps(data)},
                {"role": "user", "content": f"That clue isn't allowed: {problem} Try again."},
            ]
        self.log.append({"fallback": "clue"})
        return self.fallback.give_clue(view)

    def _clue_problem(self, view: dict, data: dict, agents: list[str]) -> str | None:
        if not data.get("clue"):
            return "The reply had no clue."
        try:
            game.check_clue(view, data["clue"], int(data.get("number", 1)))  # only reads view["words"]
        except (game.RuleError, ValueError) as e:
            return str(e)
        if not is_real_word(data["clue"]):
            return f"{data['clue'].upper()} isn't a common English word."
        if not any(t.upper() in agents for t in data.get("targets", [])):
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
                f"Board:\n{describe_board(view, show_key=True)}\n\n"
                f"Game so far:\n{describe_history(view)}\n\n"
                f"Your partner's clue is {clue['word']} {clue['number']}. Words you can guess: {', '.join(open_words)}.\n"
                "Guesses are judged by YOUR PARTNER's key, which you can't see. Your own key still helps: a word that is "
                "an ASSASSIN on your key is more likely to be one on theirs. Earlier clues from your partner that haven't "
                "been fully used may still point at words. List the words to guess, most confident first."
            )},
        ]
        data = self._ask(messages, GUESS_SCHEMA)
        index = {w: i for i, w in enumerate(view["words"])}
        plan = []
        for w in data.get("guesses", []):
            i = index.get(str(w).strip().upper())
            if i is not None and i not in plan:
                plan.append(i)
        return plan


_REAL_WORDS: set[str] | None = None


def is_real_word(word: str) -> bool:
    global _REAL_WORDS
    if _REAL_WORDS is None:
        from .glove import load_vectors

        _REAL_WORDS = set(load_vectors().words)
    return word.strip().upper() in _REAL_WORDS


@lru_cache(maxsize=1)
def modal_backend() -> Backend:
    """Calls the private model server (llm_server.py) through Modal; needs Modal credentials.

    One shared handle for the whole process: separate handles per thread were observed to
    send requests one at a time, which made parallel evaluation as slow as serial.
    """
    import modal

    llm = modal.Cls.from_name("cluo-duo-llm", "LLM")()

    def chat(messages, **kwargs):
        return llm.chat.remote(messages, **kwargs)

    return chat
