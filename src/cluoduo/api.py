"""HTTP API for the web game. The human always plays side "a"; the AI partner is side "b"."""

from __future__ import annotations

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
    seed: int | None = Field(default=None, ge=0, lt=2**31)
    sudden_death: bool = False
    turns: int = Field(default=game.TURNS, ge=game.TURNS, le=game.TURNS + game.EXTRA_TURNS)
    # False: return the board at once and let the client ask for the partner's first move (POST .../partner),
    # so the player can study the board while a GPU partner wakes up.
    partner_moves: bool = True


# GPU-backed partners: each new game may wake a GPU, so cap them per day (UTC) to protect the credit.
GPU_PARTNERS = {"smart": "Gemma31"}
PROGRESS = "__progress__"  # store key prefix for a GPU partner's current step
GPU_GAMES_PER_DAY = 40


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


def create_app(store: Store, allow_local: bool = False, warm_up=None) -> FastAPI:
    app = FastAPI(title="Cluo Duo", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=LOCAL_ORIGINS if allow_local else None,
                       allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    def load(game_id: str) -> game.State:
        state = store.get(game_id)
        if state is None:
            raise HTTPException(404, "Game not found (games expire after a while).")
        return state

    def partner_for(state: game.State):
        return PLAYERS[state.get("partner", "random")](seed=state["seed"] + len(state["history"]))

    def respond(state: game.State) -> dict:
        store.put(state)
        return game.view(state, HUMAN) | {"partner": state.get("partner", "random")}

    def partner_plays(state: game.State) -> game.State:
        player = partner_for(state)
        if state.get("partner") not in GPU_PARTNERS:
            return play_partner(state, AI, player)
        # A GPU partner can take a while: it reports each step here, for GET .../progress.
        key = PROGRESS + state["id"]
        player.progress = lambda text, answered: store.put({"id": key, "text": text, "answered": answered})
        try:
            return play_partner(state, AI, player)
        finally:
            store.put({"id": key, "text": None, "answered": True})

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

    def gpu_games_today(partner: str, add: bool = False) -> int:
        from datetime import datetime, timezone

        key = f"__count__{partner}__{datetime.now(timezone.utc):%Y-%m-%d}"
        record = store.get(key) or {"id": key, "count": 0}
        if add:
            record["count"] += 1
            store.put(record)
        return record["count"]

    @app.post("/api/warm")
    def warm(body: Warm) -> dict:
        """Called when a player selects a GPU partner, so the GPU boots while they read the menu."""
        if body.partner not in GPU_PARTNERS or gpu_games_today(body.partner) >= GPU_GAMES_PER_DAY:
            return {"ok": False}
        if warm_up is not None:
            warm_up(GPU_PARTNERS[body.partner])
        return {"ok": True}

    @app.post("/api/games")
    def new(body: NewGame) -> dict:
        if body.partner not in PLAYERS:
            raise HTTPException(400, f"Unknown partner {body.partner!r}.")
        if body.partner in GPU_PARTNERS:
            if gpu_games_today(body.partner) >= GPU_GAMES_PER_DAY:
                raise HTTPException(429, "The Advanced partner has had its games for today. Please try Basic, or come back tomorrow.")
            gpu_games_today(body.partner, add=True)
        state = game.new_game(seed=body.seed, first_clue=HUMAN if body.first == "human" else AI,
                              sudden_death=body.sudden_death, turns=body.turns)
        state["partner"] = body.partner
        if body.partner_moves:
            state = partner_plays(state)
        return respond(state)

    @app.get("/api/games/{game_id}/progress")
    def progress(game_id: str) -> dict:
        """What a GPU partner is doing right now (text is None when it isn't mid-move)."""
        record = store.get(PROGRESS + game_id) or {}
        return {"text": record.get("text"), "answered": record.get("answered", False)}

    @app.post("/api/games/{game_id}/partner")
    def partner(game_id: str) -> dict:
        """Let the partner make any move it owes (a no-op when it's the human's move)."""
        return move(game_id, lambda s: s)

    @app.get("/api/games/{game_id}")
    def get(game_id: str) -> dict:
        state = load(game_id)
        return game.view(state, HUMAN) | {"partner": state.get("partner", "random")}

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
