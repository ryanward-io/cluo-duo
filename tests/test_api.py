"""API guards: seeds, warm-up limits, the daily GPU game count, the monthly GPU budget and reserved store keys."""

import pytest
from fastapi.testclient import TestClient

from cluoduo import api, game, players
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
    assert c.post("/api/warm", json={"partner": "random"}).json() == {"ok": False, "reason": "monthly"}
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


def test_warm_says_daily_cap(gpu_random, monkeypatch):
    monkeypatch.setattr(api, "GPU_GAMES_PER_DAY", 0)
    assert client().post("/api/warm", json={"partner": "random"}).json() == {"ok": False, "reason": "daily"}


def test_budget_running_out_mid_game_switches_to_basic(gpu_random, monkeypatch):
    """A game already using the GPU can't keep it once the month's budget is spent: Basic plays on."""
    played_by = []

    class Tracker(players.RandomPlayer):
        def __init__(self, seed=None, label="gpu"):
            super().__init__(seed)
            self.label = label

        def give_clue(self, view):
            played_by.append(self.label)
            return super().give_clue(view)

        def guess(self, view):
            played_by.append(self.label)
            return super().guess(view)

    monkeypatch.setitem(api.PLAYERS, "random", lambda seed=None: Tracker(seed, "gpu"))
    monkeypatch.setitem(api.PLAYERS, "basic-fake", lambda seed=None: Tracker(seed, "basic"))
    monkeypatch.setattr(api, "BUDGET_FALLBACK", "basic-fake")
    dollars = [10.0]
    store = MemoryStore()
    c = TestClient(api.create_app(store, allow_local=True, usage=lambda: dollars[0]))  # local: the seed fixes the board

    def a_clue(view):
        return players.RandomPlayer(1).give_clue(view)[0]

    s = c.post("/api/games", json={"partner": "random", "first": "human", "seed": 11}).json()
    r = c.post(f"/api/games/{s['id']}/clue", json={"word": a_clue(s), "number": 1})
    assert r.status_code == 200, r.text
    s = r.json()
    assert "gpu" in played_by  # the partner's guesses used the "GPU", and the game is now counted
    dollars[0] = 28.0
    store.put(store.get(api.USAGE) | {"t": 0})  # time for a fresh read, which finds the budget spent
    played_by.clear()
    for _ in range(40):  # finish the partner's turn, then give a clue so the partner has to guess
        if s["phase"] == "over":
            break
        if s["phase"] == "guess" and s["clue"]["giver"] == "b":
            i = next(i for i, f in enumerate(s["found"]) if not f)
            s = c.post(f"/api/games/{s['id']}/guess", json={"index": i}).json()
        elif s["phase"] == "clue" and s["giver"] == "a":
            s = c.post(f"/api/games/{s['id']}/clue", json={"word": a_clue(s), "number": 1}).json()
            break
    assert played_by and set(played_by) == {"basic"}


def test_failed_lookup_retries_later_not_every_request(gpu_random):
    reads = []

    def broken():
        reads.append(1)
        raise RuntimeError("billing unavailable")

    c = TestClient(api.create_app(MemoryStore(), usage=broken))
    for _ in range(3):
        assert c.post("/api/games", json={"partner": "random", "first": "human"}).status_code == 200
    assert len(reads) == 1


def test_stale_usage_figure_turns_advanced_off(gpu_random):
    store = MemoryStore()
    store.put({"id": api.USAGE, "t": 0, "read_at": 0, "dollars": 5.0})  # read long ago, lookups failing since

    def broken():
        raise RuntimeError("billing unavailable")

    c = TestClient(api.create_app(store, usage=broken))
    assert c.post("/api/games", json={"partner": "random", "first": "human"}).status_code == 429


def test_near_the_limit_rereads_more_often(gpu_random):
    import time as _time
    reads = []
    store = MemoryStore()
    store.put({"id": api.USAGE, "t": _time.time() - api.USAGE_NEAR_MAX_AGE - 1, "read_at": _time.time() - 700, "dollars": 25.0})

    def usage():
        reads.append(1)
        return 25.5

    c = TestClient(api.create_app(store, usage=usage))
    c.post("/api/games", json={"partner": "random", "first": "human"})
    assert reads == [1]  # $25 is within $3 of $27, so a 10-minute-old figure is re-read


def test_usage_fields_recorded():
    store = MemoryStore()
    c = client(store=store)
    player = "0f8e2c1a-4b7d-4e3a-9c2b-1a2b3c4d5e6f"
    gid = c.post("/api/games", json={"partner": "random", "first": "human", "player": player, "me": True}).json()["id"]
    state = store.get(gid)
    assert state["player"] == player and state["me"] is True
    assert state["created"] <= state["updated"]
    assert "player" not in c.get(f"/api/games/{gid}").json()  # not echoed back


def test_player_id_must_look_random():
    r = client().post("/api/games", json={"partner": "random", "player": "ryan@example.com"})
    assert r.status_code == 422
