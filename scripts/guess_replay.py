"""Replay past guessing decisions under different guess prompts (a paired, offline test).

Every time side a (a Sonnet reference player, or a human on the live site) gave a clue to a
Gemma partner on side b, the board at that moment is rebuilt and the guess call is asked
again, once per prompt variant and sample. The planned guesses are scored with side a's key,
the way the game would: an agent counts and play goes on, a bystander ends the turn, an
assassin loses. Only the guess turn is replayed (the rest of the game stays as it was), so
every variant sees exactly the same decisions.

    uv run python scripts/guess_replay.py --samples 3 --out data/llm/guess-replay.json
    uv run python scripts/guess_replay.py --report data/llm/guess-replay.json

Live games come from data/live/games.json (exported from the Modal Dict; not committed).
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from cluoduo import game
from cluoduo.llm import RULES, describe_history, guess_schema, modal_backend

ROOT = Path(__file__).resolve().parents[1]
GEMMA = {"smart", "smart-oct28", "smart-gemma-noglove", "smart-gemma-llmguess", "smart-gemma-cap10"}

# ---------- the prompt variants ----------

OLD_ASSASSIN_NOTE = "an ASSASSIN on YOUR key: 1 in 3 chance it is one on your partner's key too"
TRUST_ASSASSIN_NOTE = ("an ASSASSIN on YOUR key, which only matters if the clue is unclear: your partner checks "
                       "every clue against THEIR assassins, so a word their clue clearly points at is most likely "
                       "their agent")
OLD_BYSTANDER_NOTE = "known bystander on YOUR key"
FIXED_BYSTANDER_NOTE = "a bystander on YOUR key, which says nothing about your partner's key: still a fair guess"

OLD_TAIL = ", and avoid words that are ASSASSINS on your own key unless the clue clearly means them."
TRUST_TAIL = (". Trust your partner: they can see their own key and avoided clues that point at their assassins. "
              "So if a word is clearly the best fit for the clue, guess it, even if it is an ASSASSIN on your own key. "
              "Your own key only breaks ties: when two words fit about equally well, prefer the one that is not an "
              "ASSASSIN on your key.")

VARIANTS = {
    # name: (assassin note or None, bystander note, instruction tail, allow own assassin after the first pick, reason max)
    "live": (OLD_ASSASSIN_NOTE, OLD_BYSTANDER_NOTE, OLD_TAIL, False, 300),
    "bystander-fix": (OLD_ASSASSIN_NOTE, FIXED_BYSTANDER_NOTE, OLD_TAIL, False, 300),
    "trust": (TRUST_ASSASSIN_NOTE, FIXED_BYSTANDER_NOTE, TRUST_TAIL, True, 300),
    "trust-long": (TRUST_ASSASSIN_NOTE, FIXED_BYSTANDER_NOTE, TRUST_TAIL, True, 600),
    "no-marks": (None, FIXED_BYSTANDER_NOTE, ".", True, 300),
}
# trust-long, plus a list of the partner's unfinished earlier clues and room for one extra guess on one
LEFTOVER = {"trust-long-leftover"}
VARIANTS["trust-long-leftover"] = VARIANTS["trust-long"]

LEFTOVER_TAIL = (" UNFINISHED CLUES: when a turn ends on a bystander or a stop before all of a clue's targets "
                 "were found, those targets are probably still on the board. After your guesses for this clue, you "
                 "may add ONE extra guess: a word that clearly fits one of your partner's unfinished clues listed "
                 "above (the way a human player looks back at earlier clues). Only add it if the fit is clear; a loose "
                 "extra guess risks an ASSASSIN.")


def unfinished(view: dict) -> str:
    """The partner's earlier clues that found fewer agents than their number."""
    me, lines, turn, cur = view["side"], [], 0, None
    for h in view["history"]:
        if h["type"] == "clue":
            turn += 1
            cur = {"turn": turn, "word": h["word"], "number": h["number"], "found": [], "missed": None} \
                if h["side"] != me else None
            if cur is not None:
                lines.append(cur)
        elif h["type"] == "guess" and cur is not None:
            if h["outcome"] == "G":
                cur["found"].append(h["word"])
            else:
                cur["missed"] = h["word"]
    lines = [c for c in lines[:-1] if len(c["found"]) < c["number"]]  # the last one is the current clue
    if not lines:
        return "(none)"
    out = []
    for c in lines:
        got = f"you found {', '.join(c['found'])}" if c["found"] else "you found nothing"
        end = f", then hit the bystander {c['missed']}" if c["missed"] else ", then stopped"
        out.append(f"- Turn {c['turn']}: {c['word']} {c['number']}: {got}{end}. "
                   f"Up to {c['number'] - len(c['found'])} target(s) may still be hidden (or found since by other clues).")
    return "\n".join(out)


def board(view: dict, assassin_note: str | None, bystander_note: str) -> str:
    me, partner = view["side"], game.other(view["side"])
    rows = []
    for i, word in enumerate(view["words"]):
        notes = []
        if assassin_note and view["my_key"][i] == game.ASSASSIN and not view["found"][i]:
            notes.append(assassin_note)
        if view["found"][i]:
            notes.append("already found (out of play)")
        if i in view["bystanders"][me]:
            notes.append(bystander_note)
        if i in view["bystanders"][partner]:
            notes.append("known bystander on your PARTNER's key")
        rows.append(f"- {word}" + (f" ({'; '.join(notes)})" if notes else ""))
    return "\n".join(rows)


def messages(view: dict, variant: str) -> list[dict]:
    note, bnote, tail, _, _ = VARIANTS[variant]
    clue = view["clue"]
    extra = f"Your partner's unfinished earlier clues:\n{unfinished(view)}\n\n" if variant in LEFTOVER else ""
    if variant in LEFTOVER:
        tail += LEFTOVER_TAIL
    open_words = [w for i, w in enumerate(view["words"]) if game.can_guess(view, i)]
    return [
        {"role": "system", "content": RULES},
        {"role": "user", "content": (
            f"Board:\n{board(view, note, bnote)}\n\n"
            f"Game so far:\n{describe_history(view)}\n\n" + extra +
            f"Your partner's clue is {clue['word']} {clue['number']}: they are pointing at {clue['number']} word(s) "
            "that are AGENTS on THEIR key. Words you can guess: " + ", ".join(open_words) + ".\n"
            f"Pick at most {max(clue['number'], 1)} word(s) most strongly linked to the clue, most confident first. "
            "A wrong guess can hit an ASSASSIN, so leave out words you are unsure of" + tail
        )},
    ]


def schema(view: dict, variant: str) -> dict:
    open_words = [w for i, w in enumerate(view["words"]) if game.can_guess(view, i)]
    s = guess_schema(open_words, view["clue"]["number"])
    s["properties"]["reason"]["maxLength"] = VARIANTS[variant][4]
    if variant in LEFTOVER:
        s["properties"]["guesses"]["maxItems"] = min(len(open_words), max(view["clue"]["number"], 1) + 1)
    return s


def plan(view: dict, data: dict, variant: str) -> list[int]:
    """Same post-processing as LLMPlayer._plan_guesses, with the own-assassin cut-off optional."""
    index = {w: i for i, w in enumerate(view["words"])}
    out = []
    for w in data.get("guesses", []):
        i = index.get(str(w).strip().upper())
        if i is None or i in out or not game.can_guess(view, i):
            continue
        if view["my_key"][i] == game.ASSASSIN and out and not VARIANTS[variant][3]:
            break
        out.append(i)
    return out[:max(view["clue"]["number"], 1) + (variant in LEFTOVER)]


# ---------- the decisions ----------

def decisions() -> list[dict]:
    """Every point where side a has just given side b a clue, in Gemma partner games."""
    sources = []
    for f in sorted((ROOT / "data" / "agent-games").glob("*.json")):
        s = json.loads(f.read_text())
        if s.get("partner") in GEMMA:
            sources.append(("ref", s))
    live = ROOT / "data" / "live" / "games.json"
    if live.exists():
        sources += [("live", s) for s in json.loads(live.read_text()) if s.get("partner") == "smart"]
    out = []
    for origin, final in sources:
        s = {**deepcopy(final), "found": [False] * game.BOARD_SIZE, "bystanders": {"a": [], "b": []},
             "turns_left": final.get("turns", game.TURNS), "giver": final["history"][0]["side"] if final["history"] else "a",
             "phase": "clue", "clue": None, "guesser": None, "history": [], "result": None}
        for h in final["history"]:
            try:
                if h["type"] == "clue":
                    s = game.give_clue(s, h["side"], h["word"], h["number"])
                    if h["side"] == "a":
                        out.append({"origin": origin, "game": final["id"], "state": deepcopy(s)})
                elif h["type"] == "guess":
                    s = game.guess(s, h["side"], h["index"])
                elif h["type"] == "stop":
                    s = game.stop(s, h["side"])
                else:
                    break  # sudden death or end: not replayed
            except game.RuleError:
                break
    return out


def score(state: dict, guesses: list[int]) -> dict:
    key = state["key"]["a"]
    found, end = 0, "stop"
    for i in guesses:
        if key[i] == game.GREEN:
            found += 1
            continue
        end = "assassin" if key[i] == game.ASSASSIN else "bystander"
        break
    return {"found": found, "end": end}


# ---------- running and reporting ----------

def run(samples: int, variants: list[str], out: Path, server: str, workers: int) -> None:
    ds = decisions()
    print(f"{len(ds)} decisions ({sum(d['origin'] == 'live' for d in ds)} from live games)")
    chat = modal_backend(server)
    jobs = [(n, v, k) for n in range(len(ds)) for v in variants for k in range(samples)]

    def one(job):
        n, v, k = job
        view = game.view(ds[n]["state"], "b")
        reply = chat(messages(view, v), schema=schema(view, v), seed=1000 * k + n, max_tokens=250 if VARIANTS[v][4] <= 300 else 400)
        try:
            data = json.loads(reply["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            data = {}
        g = plan(view, data, v)
        return {"decision": n, "variant": v, "sample": k, "reason": data.get("reason", ""),
                "guesses": [view["words"][i] for i in g], **score(ds[n]["state"], g)}

    results = []
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, jobs):
            results.append(r)
            if len(results) % 50 == 0:
                print(f"  {len(results)}/{len(jobs)}", flush=True)
    meta = [{"origin": d["origin"], "game": d["game"], "clue": d["state"]["clue"]["word"],
             "number": d["state"]["clue"]["number"], "words": d["state"]["words"], "key_a": d["state"]["key"]["a"],
             "key_b": d["state"]["key"]["b"], "bystanders_b": d["state"]["bystanders"]["b"]} for d in ds]
    out.write_text(json.dumps({"decisions": meta, "results": results}))
    report(out)


def report(path: Path) -> None:
    data = json.loads(path.read_text())
    meta, results = data["decisions"], data["results"]

    def flagged(n):  # an agent of a that b's prompt marks: an own assassin or a known bystander on b's key
        m = meta[n]
        return any(m["key_a"][i] == "G" and (m["key_b"][i] == "A" or i in m["bystanders_b"]) for i in range(25))

    variants = list(dict.fromkeys(r["variant"] for r in results))
    print(f"{'variant':<15}{'found/clue':>11}{'assassin %':>11}{'bystander %':>12}"
          f"{'flagged: found':>16}{'flagged: assn %':>16}{'live: found':>12}")
    for v in variants:
        rs = [r for r in results if r["variant"] == v]
        fl = [r for r in rs if flagged(r["decision"])]
        lv = [r for r in rs if meta[r["decision"]]["origin"] == "live"]
        pct = lambda xs, end: 100 * sum(r["end"] == end for r in xs) / max(len(xs), 1)
        print(f"{v:<15}{statistics.mean(r['found'] for r in rs):>11.2f}{pct(rs, 'assassin'):>11.1f}{pct(rs, 'bystander'):>12.1f}"
              f"{statistics.mean(r['found'] for r in fl):>16.2f}{pct(fl, 'assassin'):>16.1f}"
              f"{statistics.mean(r['found'] for r in lv) if lv else float('nan'):>12.2f}")
    print(f"({len(meta)} decisions, {sum(flagged(n) for n in range(len(meta)))} flagged, "
          f"{sum(m['origin'] == 'live' for m in meta)} live; samples per decision: "
          f"{len(results) // max(len(variants), 1) // max(len(meta), 1)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--server", default="Gemma31")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "llm" / "guess-replay.json")
    p.add_argument("--report", type=Path)
    p.add_argument("--count", action="store_true", help="only count the decisions")
    a = p.parse_args()
    if a.count:
        ds = decisions()
        print(len(ds), sum(d["origin"] == "live" for d in ds))
    elif a.report:
        report(a.report)
    else:
        run(a.samples, a.variants.split(","), a.out, a.server, a.workers)
