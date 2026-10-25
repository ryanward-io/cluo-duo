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

ALLOWED_ORIGINS = ["https://ryanward.dev"]
LOCAL_ORIGINS = r"http://(localhost|127\.0\.0\.1):\d+"  # any port, for local development


class NewGame(BaseModel):
    first: Literal["human", "partner"] = "partner"
    partner: str = "random"
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class Clue(BaseModel):
    word: str = Field(min_length=1, max_length=40)
    number: int = Field(ge=0, le=game.MAX_CLUE_NUMBER)


class Guess(BaseModel):
    index: int = Field(ge=0, lt=game.BOARD_SIZE)


def create_app(store: Store) -> FastAPI:
    app = FastAPI(title="Clue Words Duet", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=LOCAL_ORIGINS,
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

    def move(game_id: str, fn) -> dict:
        state = load(game_id)
        try:
            state = fn(state)
        except game.RuleError as e:
            raise HTTPException(400, str(e)) from e
        state = play_partner(state, AI, partner_for(state))
        return respond(state)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/api/games")
    def new(body: NewGame) -> dict:
        if body.partner not in PLAYERS:
            raise HTTPException(400, f"Unknown partner {body.partner!r}.")
        state = game.new_game(seed=body.seed, first_clue=HUMAN if body.first == "human" else AI)
        state["partner"] = body.partner
        state = play_partner(state, AI, partner_for(state))
        return respond(state)

    @app.get("/api/games/{game_id}")
    def get(game_id: str) -> dict:
        state = load(game_id)
        return game.view(state, HUMAN) | {"partner": state.get("partner", "random")}

    @app.post("/api/games/{game_id}/clue")
    def clue(game_id: str, body: Clue) -> dict:
        return move(game_id, lambda s: game.give_clue(s, HUMAN, body.word, body.number))

    @app.post("/api/games/{game_id}/guess")
    def guess(game_id: str, body: Guess) -> dict:
        return move(game_id, lambda s: game.guess(s, HUMAN, body.index))

    @app.post("/api/games/{game_id}/stop")
    def stop(game_id: str) -> dict:
        return move(game_id, lambda s: game.stop(s, HUMAN))

    return app
