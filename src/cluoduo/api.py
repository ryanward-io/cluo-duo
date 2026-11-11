"""HTTP API for the web game. The human always plays side "a"; the AI partner is side "b"."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import game
from .players import PLAYERS, play_partner
from .store import Store

HUMAN: game.Side = "a"
AI: game.Side = "b"

# CORS: which web pages browsers will let read our responses. This is not access control;
# non-browser clients ignore it. Abuse limits (rate limits, spend caps) must live on the server.
ALLOWED_ORIGINS = ["https://ryanward.dev"]
LOCAL_ORIGINS = r"http://(localhost|127\.0\.0\.1):\d+"  # any port; only used by dev.py


class NewGame(BaseModel):
    first: Literal["human", "partner"] = "partner"
    partner: str = "random"
    seed: int | None = Field(default=None, ge=0, lt=2**31)  # local dev only: the layout follows from the seed
    sudden_death: bool = False
    turns: int = Field(default=game.TURNS, ge=game.TURNS, le=game.TURNS + game.EXTRA_TURNS)
    # False: return the board at once and let the client ask for the partner's first move (POST .../partner),
    # so the player can study the board while a GPU partner wakes up.
    partner_moves: bool = True
    # Usage stats: a random id the browser keeps (no personal data), and whether this browser is Ryan's own.
    player: str | None = Field(default=None, pattern=r"^[0-9a-f-]{8,36}$")
    me: bool = False


# GPU-backed partners: each new game may wake a GPU, so cap them per day (UTC) to protect the credit.
GPU_PARTNERS = {"smart": "Gemma31"}
BUDGET_FALLBACK = "glove-rank"  # Basic, which plays a GPU partner's moves once the monthly budget is spent
PROGRESS = "__progress__"  # store key prefix for a GPU partner's current step
GPU_SEEN = "__gpu_seen__"  # when the GPU partner last answered anyone
GPU_AWAKE_SECONDS = 4 * 60  # the GPU sleeps 5 minutes after its last call; allow a minute's margin
GPU_GAMES_PER_DAY = 40
# Each warm-up that wakes the GPU costs its boot plus 5 idle minutes, so cap them too, and skip any
# warm-up while the GPU is already awake or was asked to wake recently.
GPU_WARMS_PER_DAY = 40
LAST_WARM = "__last_warm__"
# Modal's $30 usage limit stops the whole workspace, Basic included. So the server turns Advanced off
# once this month's usage reaches $27, leaving the rest for the CPU app. The check runs before every GPU
# move, not just new games, so games already under way switch to Basic too. Modal's figures trail by up
# to an hour and the GPU costs about $2 an hour; within $3 of the limit the figure is re-read every
# 10 minutes, so usage can overshoot $27 a little but not reach $30.
GPU_MONTHLY_USAGE_LIMIT = 27.0
USAGE = "__usage__"  # this month's usage in dollars, when it was read, and when to check again
USAGE_MAX_AGE = 60 * 60  # re-read Modal's figure at most once an hour...
USAGE_NEAR = 3.0  # ...or every USAGE_NEAR_MAX_AGE once within this many dollars of the limit
USAGE_NEAR_MAX_AGE = 10 * 60
USAGE_RETRY = 10 * 60  # after a failed lookup, wait this long before trying again
USAGE_STALE = 6 * 60 * 60  # a figure this old (lookups keep failing) turns Advanced off, to be safe
DAILY_CAP = "The Advanced partner has had its games for today. Please try Basic, or come back tomorrow."
MONTHLY_CAP = "The Advanced partner has used its budget for this month. Please try Basic, or come back next month."


class Warm(BaseModel):
    partner: str


class Clue(BaseModel):
    word: str = Field(min_length=1, max_length=40)
    number: int = Field(ge=0, le=game.MAX_CLUE_NUMBER)
    anyway: bool = False  # the player confirmed a clue their partner doesn't know


# Partners that read clues only through GloVe, so a word outside its vocabulary is a blind guess.
GLOVE_GUESSERS = {"glove-sum", "glove-bayes", "glove-rank", "glove-rank-safe"}


class Guess(BaseModel):
    index: int = Field(ge=0, lt=game.BOARD_SIZE)


def owes_move(state: game.State) -> bool:
    """Whether the AI side has a move to make (the same test play_partner loops on)."""
    phase = state["phase"]
    return ((phase == "clue" and state["giver"] == AI) or (phase == "guess" and state["clue"]["giver"] != AI)
            or (phase == "sudden_death" and state["guesser"] == AI))


def create_app(store: Store, allow_local: bool = False, warm_up=None, usage=None) -> FastAPI:
    """usage: a function returning this month's workspace usage in dollars (counted as Modal's usage
    limit counts it), or None to skip the monthly budget check (tests and local dev)."""
    app = FastAPI(title="Cluo Duo", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=LOCAL_ORIGINS if allow_local else None,
                       allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    def load(game_id: str) -> game.State:
        # Keys starting "__" hold the server's own records (counters, progress), never games.
        state = None if game_id.startswith("__") else store.get(game_id)
        if state is None:
            raise HTTPException(404, "Game not found (games expire after a while).")
        return state

    def partner_for(state: game.State):
        # The model server takes a small seed; the low bits say nothing useful about the board.
        return PLAYERS[state.get("partner", "random")](seed=state["seed"] % 2**31 + len(state["history"]))

    def public(state: game.State) -> dict:
        """The human's view. The partner's reasons for its moves would give its key away, so they wait for the end."""
        over = state["phase"] == "over"
        return game.view(state, HUMAN) | {"partner": state.get("partner", "random"),
                                          "thoughts": state.get("thoughts", []) if over else None}

    def respond(state: game.State) -> dict:
        state["updated"] = time.time()
        store.put(state)
        return public(state)

    def partner_plays(state: game.State) -> game.State:
        player = partner_for(state)
        gpu = state.get("partner") in GPU_PARTNERS and owes_move(state)
        if gpu and state.get("gpu_counted") and over_budget():
            # The month's GPU budget ran out mid-game: Basic plays the rest of this game's moves.
            player = PLAYERS[BUDGET_FALLBACK](seed=state["seed"] % 2**31 + len(state["history"]))
            gpu = False
        if gpu and not state.get("gpu_counted"):
            # Count a GPU game when it first needs the GPU, so games that never get that far can't use up the day.
            gpu_allowed(state["partner"])
            gpu_games_today(state["partner"], add=True)
            state["gpu_counted"] = True
        if gpu:
            # A GPU partner can take a while: it reports each step here, for GET .../progress.
            key = PROGRESS + state["id"]
            def progress(text: str, answered: bool) -> None:
                store.put({"id": key, "text": text, "answered": answered})
                if answered:
                    store.put({"id": GPU_SEEN, "t": time.time()})

            player.progress = progress
        try:
            state = play_partner(state, AI, player)
        finally:
            if gpu:
                store.put({"id": key, "text": None, "answered": True})
                if getattr(player, "answered", False):
                    store.put({"id": GPU_SEEN, "t": time.time()})
        # A clue retried after breaking a rule is noted twice: keep the last note for each move.
        thoughts = {t["at"]: t for t in state.get("thoughts", []) + getattr(player, "thoughts", [])}
        if thoughts:
            state["thoughts"] = sorted(thoughts.values(), key=lambda t: t["at"])
        return state

    def move(game_id: str, fn) -> dict:
        state = load(game_id)
        try:
            state = fn(state)
        except game.RuleError as e:
            raise HTTPException(400, str(e)) from e
        state = partner_plays(state)
        return respond(state)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    def gpu_games_today(partner: str, add: bool = False, kind: str = "count") -> int:
        from datetime import datetime, timezone

        key = f"__{kind}__{partner}__{datetime.now(timezone.utc):%Y-%m-%d}"
        record = store.get(key) or {"id": key, "count": 0}
        if add:
            record["count"] += 1
            store.put(record)
        return record["count"]

    def over_budget() -> bool:
        if usage is None:
            return False
        now = time.time()
        record = store.get(USAGE)
        near = record is not None and record.get("dollars") is not None \
            and record["dollars"] >= GPU_MONTHLY_USAGE_LIMIT - USAGE_NEAR
        max_age = USAGE_NEAR_MAX_AGE if near else USAGE_MAX_AGE
        if record is None or now - record["t"] > max_age:
            try:
                record = {"id": USAGE, "t": now, "read_at": now, "dollars": float(usage())}
            except Exception as e:
                # Keep the last figure and try again in a few minutes, rather than on every request.
                print(f"Monthly usage lookup failed: {e!r}")
                record = (record or {"id": USAGE, "dollars": None}) | {"t": now - max_age + USAGE_RETRY}
            store.put(record)
        if record.get("dollars") is None:
            return False  # never read yet: allow play; the daily caps and Modal's own $30 limit still apply
        if now - record.get("read_at", record["t"]) > USAGE_STALE:
            return True  # lookups have been failing for hours: switch Advanced off rather than guess
        return record["dollars"] >= GPU_MONTHLY_USAGE_LIMIT

    def gpu_allowed(partner: str) -> None:
        """Raise 429 if the GPU partner can't take another game today or this month."""
        if gpu_games_today(partner) >= GPU_GAMES_PER_DAY:
            raise HTTPException(429, DAILY_CAP)
        if over_budget():
            raise HTTPException(429, MONTHLY_CAP)

    @app.post("/api/warm")
    def warm(body: Warm) -> dict:
        """Called when a player selects a GPU partner, so the GPU boots while they read the menu."""
        if body.partner not in GPU_PARTNERS:
            return {"ok": False}
        if gpu_games_today(body.partner) >= GPU_GAMES_PER_DAY:
            return {"ok": False, "reason": "daily"}  # the page says which cap, so it can say when to come back
        if over_budget():
            return {"ok": False, "reason": "monthly"}
        now = time.time()
        recent = max((store.get(GPU_SEEN) or {}).get("t", 0), (store.get(LAST_WARM) or {}).get("t", 0))
        if now - recent < GPU_AWAKE_SECONDS:
            return {"ok": True}  # awake, or already waking: nothing to do
        if gpu_games_today(body.partner, kind="warms") >= GPU_WARMS_PER_DAY:
            return {"ok": True}  # out of warm-ups for today; the game's first move still wakes it
        gpu_games_today(body.partner, add=True, kind="warms")
        store.put({"id": LAST_WARM, "t": now})
        if warm_up is not None:
            warm_up(GPU_PARTNERS[body.partner])
        return {"ok": True}

    @app.post("/api/games")
    def new(body: NewGame) -> dict:
        if body.partner not in PLAYERS:
            raise HTTPException(400, f"Unknown partner {body.partner!r}.")
        if body.partner in GPU_PARTNERS:
            gpu_allowed(body.partner)
        # A seed fixes the whole layout, partner's key included, and the code is public: local dev only.
        state = game.new_game(seed=body.seed if allow_local else None, first_clue=HUMAN if body.first == "human" else AI,
                              sudden_death=body.sudden_death, turns=body.turns)
        state["partner"] = body.partner
        state |= {"created": time.time(), "player": body.player, "me": body.me}
        if body.partner_moves:
            state = partner_plays(state)
        return respond(state)

    @app.get("/api/games/{game_id}/progress")
    def progress(game_id: str) -> dict:
        """What a GPU partner is doing right now (text is None when it isn't mid-move)."""
        record = store.get(PROGRESS + game_id) or {}
        seen = (store.get(GPU_SEEN) or {}).get("t", 0)
        return {"text": record.get("text"), "answered": record.get("answered", False),
                "awake": time.time() - seen < GPU_AWAKE_SECONDS}

    @app.post("/api/games/{game_id}/partner")
    def partner(game_id: str) -> dict:
        """Let the partner make any move it owes (a no-op when it's the human's move)."""
        return move(game_id, lambda s: s)

    @app.get("/api/games/{game_id}")
    def get(game_id: str) -> dict:
        return public(load(game_id))

    @app.post("/api/games/{game_id}/clue")
    def clue(game_id: str, body: Clue) -> dict:
        state = load(game_id)
        if not body.anyway and state.get("partner") in GLOVE_GUESSERS:
            from .glove import load_vectors

            word = game.normalise_clue(body.word)
            if word not in load_vectors().index:
                raise HTTPException(409, {"unknown_word": word})
        return move(game_id, lambda s: game.give_clue(s, HUMAN, body.word, body.number))

    @app.post("/api/games/{game_id}/guess")
    def guess(game_id: str, body: Guess) -> dict:
        return move(game_id, lambda s: (game.sudden_death_guess if s["phase"] == "sudden_death" else game.guess)(
            s, HUMAN, body.index))

    @app.post("/api/games/{game_id}/stop")
    def stop(game_id: str) -> dict:
        return move(game_id, lambda s: game.stop(s, HUMAN))

    return app
