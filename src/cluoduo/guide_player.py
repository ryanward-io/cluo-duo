"""LLM partner driven by the player guide (prompts/player.md), built from the subagent rounds.

The guide was refined over ten rounds of Claude subagents playing each other (Sonnet with
guide v9 won 6 of 16). This player gives an open model (Qwen on Modal) the same guide and
the same text view of the board those subagents saw, and then enforces in code the checks
the models most often skipped:

- Clues: the reply must include the model's own rehearsal, the 4 words it thinks the partner
  would guess for the clue. If any of its own assassins is in that list the clue is rejected
  and the model is told why (the guide's FINAL CHECK, made mandatory). The clue's number
  counts only targets that appear in that list.
- Guesses: a word that is an ASSASSIN on the guesser's own key may only be the first guess
  (the "clear pointer beats your own key" rule never failed as a first guess; loose later
  guesses on own assassins lost several games). Guesses stop at the clue's number plus one
  leftover.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import game
from .llm import CLUE_WORD, THINKING_TOKENS, LLMPlayer

GUIDE = Path(__file__).resolve().parents[2] / "prompts" / "player.md"
MAX_GUIDE_TARGETS = 4  # the guide allows a 4 for a safe stretch clue
GUIDE_ATTEMPTS = 5  # Qwen often needs several tries to find a clue that passes its own assassin check
MARK = {game.GREEN: "AGENT", game.BYSTANDER: "bystander", game.ASSASSIN: "ASSASSIN"}


def board_text(view: dict, guessing: bool = False) -> str:
    """The board as the subagents saw it (scripts/agent_play.py), from this player's side.

    `guessing` hides my own AGENT marks: Qwen (like Haiku) kept guessing its own agents
    despite the guide saying they are not hints, so the interface removes the temptation.
    My ASSASSIN marks stay, because the guide's own-assassin rule needs them.
    """
    me, partner = view["side"], game.other(view["side"])
    lines = [f"Agents found {view['found_count']}/15  |  turns left {view['turns_left']}", "", "Board (your key in brackets):"]
    for i, w in enumerate(view["words"]):
        notes = []
        if view["found"][i]:
            notes.append("FOUND")
        if i in view["bystanders"][me]:
            notes.append("known bystander on your key")
        if i in view["bystanders"][partner]:
            notes.append("known bystander on partner's key")
        mark = MARK[view["my_key"][i]]
        if guessing:
            mark = "ASSASSIN on your key" if mark == "ASSASSIN" else "-"
        lines.append(f"  {w:<12} [{mark}]" + (f"  ({'; '.join(notes)})" if notes else ""))
    lines += ["", "History:"]
    for h in view["history"]:
        who = "You" if h.get("side") == me else "Partner"
        if h["type"] == "clue":
            lines.append(f"  {who} gave {h['word']} {h['number']}")
        elif h["type"] == "guess":
            lines.append(f"    {who} guessed {h['word']}: {dict(G='agent', N='bystander', A='ASSASSIN')[h['outcome']]}")
        elif h["type"] == "stop":
            lines.append(f"    {who} stopped")
    if not view["history"]:
        lines.append("  (none)")
    return "\n".join(lines)


class GuidePlayer(LLMPlayer):
    name = "qwen36-guide"

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(seed, **kwargs)
        self.guide = GUIDE.read_text()

    def _messages(self, view: dict, task: str, guessing: bool = False) -> list[dict]:
        return [{"role": "system", "content": self.guide},
                {"role": "user", "content": f"{board_text(view, guessing)}\n\n{task}"}]

    def _ask(self, messages, schema, kind):
        # Thinking replies need room; structured replies are longer than the old ones.
        reply = self.backend(messages, schema=schema, thinking=self.thinking, seed=self.seed + len(self.log),
                             max_tokens=THINKING_TOKENS if self.thinking else 1000)
        try:
            data = json.loads(reply["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            data = {}
        self.log.append({"kind": kind, "messages": messages, "reply": reply, "parsed": data})
        return data

    # ---------- clues ----------

    def give_clue(self, view: dict) -> tuple[str, int]:
        agents = self._agents(view)
        open_words = [w for i, w in enumerate(view["words"]) if not view["found"][i]
                      and i not in view["bystanders"][view["side"]]]
        assassins = [w for i, w in enumerate(view["words"]) if view["my_key"][i] == game.ASSASSIN and not view["found"][i]]
        schema = {
            "type": "object",
            "properties": {
                "targets": {"type": "array", "items": {"enum": agents}, "minItems": 1,
                            "maxItems": min(MAX_GUIDE_TARGETS, len(agents))},
                "clue": CLUE_WORD,
                "partner_would_guess": {"type": "array", "items": {"enum": open_words}, "minItems": min(4, len(open_words)),
                                        "maxItems": min(4, len(open_words)),
                                        "description": "FINAL CHECK: the first 4 words you would guess for this clue if you were your partner, in order"},
            },
            "required": ["targets", "clue", "partner_would_guess"],
            "additionalProperties": False,
        }
        messages = self._messages(view, (
            "YOUR MOVE: give a clue. Follow the guide: choose your targets, then the clue word, then do the FINAL "
            f"CHECK by listing the 4 words your partner would guess first. Your assassins are {', '.join(assassins) or 'all found'}."))
        for attempt in range(GUIDE_ATTEMPTS):
            data = self._ask(messages, schema, "clue")
            problem = self._clue_problem(view, data, agents)
            rehearsal = [str(w).upper() for w in data.get("partner_would_guess", [])]
            if problem is None:
                hit = [a for a in assassins if a in rehearsal]
                if hit:
                    problem = (f"Your own FINAL CHECK lists {', '.join(hit)}, which is your ASSASSIN. "
                               "The guide says to reject this clue. Choose a different clue.")
            targets = [t for t in map(str.upper, data.get("targets", [])) if t in agents]
            number = sum(t in rehearsal for t in targets)
            if problem is None and number == 0:
                problem = "None of your targets is in your own list of what your partner would guess. Choose a clearer clue."
            if problem is None:
                self.log[-1]["number"] = number
                return game.normalise_clue(data["clue"]), number
            self.log[-1]["problem"] = problem
            hint = (" Your pairs keep landing near an assassin: give a 1-clue for a single agent instead, "
                    "with a specific word far from all of your assassins." if attempt >= 1 else "")
            messages = messages + [{"role": "assistant", "content": json.dumps(data)},
                                   {"role": "user", "content": f"That clue isn't allowed: {problem}{hint} Try again."}]
        self.log.append({"fallback": "clue"})
        return self.fallback.give_clue(view)

    # ---------- guesses ----------

    def _plan_guesses(self, view: dict) -> list[int]:
        clue = view["clue"]
        open_words = [w for i, w in enumerate(view["words"]) if game.can_guess(view, i)]
        number = max(clue["number"], 1)
        schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "maxLength": 400,
                           "description": "what the clue most likely means, and any clear leftover from your partner's earlier clues"},
                "guesses": {"type": "array", "items": {"enum": open_words}, "minItems": 1,
                            "maxItems": min(len(open_words), number + 1)},
            },
            "required": ["reason", "guesses"],
            "additionalProperties": False,
        }
        data = self._ask(self._messages(view, (
            f"YOUR MOVE: your partner's clue is {clue['word']} {clue['number']}. Follow the guide's guessing rules. "
            f"List your guesses in order: up to {number} for this clue, plus at most one CLEAR leftover from your "
            "partner's earlier clues. Leave out anything loose. (Your own agents are hidden here on purpose: "
            "they are not hints about your partner's words.)"), guessing=True), schema, "guess")
        index = {w: i for i, w in enumerate(view["words"])}
        plan: list[int] = []
        for w in data.get("guesses", []):
            i = index.get(str(w).upper())
            if i is None or i in plan:
                continue
            if view["my_key"][i] == game.ASSASSIN and plan:
                break  # own assassin only ever as the first guess
            plan.append(i)
        return plan[:number + 1]


class GuideHybridPlayer(GuidePlayer):
    """The guide player, with GloVe vetoing clues that sit too close to one of its assassins."""

    name = "qwen36-guide-hybrid"

    def __init__(self, seed: int | None = None, *, assassin_margin: float = 0.05, **kwargs):
        super().__init__(seed, **kwargs)
        self.assassin_margin = assassin_margin

    def _clue_problem(self, view: dict, data: dict, agents: list[str]) -> str | None:
        problem = super()._clue_problem(view, data, agents)
        if problem or not data.get("clue"):
            return problem
        import numpy as np

        v = self.fallback.v
        clue = game.normalise_clue(data["clue"])
        if clue not in v.index:
            return None  # GloVe can't judge it; trust the model's own check
        targets = [t for t in map(str.upper, data.get("targets", [])) if t in agents and t in v.index]
        assassins = [w for i, w in enumerate(view["words"]) if view["my_key"][i] == game.ASSASSIN
                     and not view["found"][i] and w in v.index]
        if not targets or not assassins:
            return None
        sim = lambda w: float(np.dot(v.matrix[v.index[clue]], v.matrix[v.index[w]]))
        weakest = min(map(sim, targets))
        close = [a for a in assassins if sim(a) > weakest - self.assassin_margin]
        if close:
            return (f"{clue} is too close to your ASSASSIN {', '.join(close)} (by word statistics), "
                    "about as close as your weakest target. Choose a different clue.")
        return None
