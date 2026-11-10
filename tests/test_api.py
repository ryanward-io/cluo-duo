"""API guards: seeds, warm-up limits, the daily GPU game count, the monthly GPU budget and reserved store keys."""

import pytest
from fastapi.testclient import TestClient

from cluoduo import api, game
from cluoduo.store import MemoryStore


@pytest.fixture
def gpu_random(monkeypatch):
    """Treat the cheap Random partner as a GPU partner, so the GPU guards can be tested without a GPU."""
    monkeypatch.setattr(api, "GPU_PARTNERS", {"random": "Fake"})


def client(allow_local=False, warm_up=None, store=None):
    return TestClient(api.create_app(store or MemoryStore(), allow_local=allow_local, warm_up=warm_up))


def test_seed_ignored_in_production():
    words = game.new_game(seed=12345)["words"]
    c = client()
    got = {tuple(c.post("/api/games", json={"partner": "random", "seed": 12345, "first": "human"}).json()["words"])
           for _ in range(3)}
    assert tuple(words) not in got


def test_seed_kept_in_local_dev():
    words = game.new_game(seed=12345)["words"]
    r = client(allow_local=True).post("/api/games", json={"partner": "random", "seed": 12345, "first": "human"})
    assert r.json()["words"] == words


def test_reserved_keys_are_not_games():
    store = MemoryStore()
    store.put({"id": "__gpu_seen__", "t": 1})
    assert client(store=store).get("/api/games/__gpu_seen__").status_code == 404


def test_warm_skips_while_recently_warmed(gpu_random):
    calls = []
    c = client(warm_up=calls.append)
    for _ in range(3):
        assert c.post("/api/warm", json={"partner": "random"}).json() == {"ok": True}
    assert calls == ["Fake"]


def test_warm_daily_cap(gpu_random, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "GPU_AWAKE_SECONDS", -1)  # never "recently warmed"
    monkeypatch.setattr(api, "GPU_WARMS_PER_DAY", 2)
    c = client(warm_up=calls.append)
    for _ in range(5):
        c.post("/api/warm", json={"partner": "random"})
    assert len(calls) == 2


def test_gpu_game_counted_at_first_partner_move(gpu_random, monkeypatch):
    monkeypatch.setattr(api, "GPU_GAMES_PER_DAY", 1)
    c = client()
    # Creating games that never ask the partner for a move doesn't use up the day.
    for _ in range(3):
        assert c.post("/api/games", json={"partner": "random", "first": "partner", "partner_moves": False}).status_code == 200
    g = c.post("/api/games", json={"partner": "random", "first": "partner", "partner_moves": False}).json()
    assert c.post(f"/api/games/{g['id']}/partner").status_code == 200
    # The same game carries on; a new one is refused.
    assert c.post(f"/api/games/{g['id']}/partner").status_code == 200
    assert c.post("/api/games", json={"partner": "random"}).status_code == 429


def test_monthly_budget_under_limit_allows_play(gpu_random):
    dollars = [26.0]
    c = TestClient(api.create_app(MemoryStore(), usage=lambda: dollars[0]))
    assert c.post("/api/games", json={"partner": "random", "first": "human"}).status_code == 200
    dollars[0] = 27.5
    assert c.post("/api/games", json={"partner": "random", "first": "human"}).status_code == 200  # figure cached for an hour


def test_monthly_budget_over_limit(gpu_random):
    calls = []
    c = TestClient(api.create_app(MemoryStore(), warm_up=calls.append, usage=lambda: 27.0))
    r = c.post("/api/games", json={"partner": "random", "first": "human"})
    assert r.status_code == 429 and "this month" in r.json()["detail"]
    assert c.post("/api/warm", json={"partner": "random"}).json() == {"ok": False}
    assert calls == []


def test_monthly_budget_refreshes_hourly(gpu_random, monkeypatch):
    dollars = [26.0]
    reads = []
    def usage():
        reads.append(1)
        return dollars[0]
    store = MemoryStore()
    c = TestClient(api.create_app(store, usage=usage))
    c.post("/api/games", json={"partner": "random", "first": "human"})
    c.post("/api/games", json={"partner": "random", "first": "human"})
    assert len(reads) == 1
    dollars[0] = 28.0
    store.put(store.get(api.USAGE) | {"t": 0})  # an hour has passed
    assert c.post("/api/games", json={"partner": "random", "first": "human"}).status_code == 429
    assert len(reads) == 2


def test_monthly_budget_lookup_failure_allows_play(gpu_random):
    def broken():
        raise RuntimeError("billing unavailable")
    c = TestClient(api.create_app(MemoryStore(), usage=broken))
    assert c.post("/api/games", json={"partner": "random", "first": "human"}).status_code == 200


def test_basic_unaffected_by_monthly_budget():
    c = TestClient(api.create_app(MemoryStore(), usage=lambda: 100.0))
    assert c.post("/api/games", json={"partner": "glove-rank", "first": "human"}).status_code == 200
