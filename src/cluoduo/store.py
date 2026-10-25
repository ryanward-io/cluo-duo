"""Where games in progress live between requests."""

from __future__ import annotations

from typing import Any, Protocol


class Store(Protocol):
    def get(self, game_id: str) -> dict[str, Any] | None: ...

    def put(self, state: dict[str, Any]) -> None: ...


class MemoryStore:
    """For local development and tests."""

    def __init__(self) -> None:
        self._games: dict[str, dict[str, Any]] = {}

    def get(self, game_id: str) -> dict[str, Any] | None:
        return self._games.get(game_id)

    def put(self, state: dict[str, Any]) -> None:
        self._games[state["id"]] = state


class ModalDictStore:
    """Backed by a modal.Dict, so every container sees the same games."""

    def __init__(self, modal_dict: Any) -> None:
        self._d = modal_dict

    def get(self, game_id: str) -> dict[str, Any] | None:
        return self._d.get(game_id)

    def put(self, state: dict[str, Any]) -> None:
        self._d[state["id"]] = state
