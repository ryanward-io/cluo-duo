from collections import Counter

import pytest
from fastapi.testclient import TestClient

from cluoduo import game
from cluoduo.api import create_app
from cluoduo.players import RandomPlayer, self_play
from cluoduo.store import MemoryStore


def test_key_layout_matches_rules():
    s = game.new_game(seed=1)
    for side in game.SIDES:
        counts = Counter(s["key"][side])
        assert counts == {"G": 9, "A": 3, "N": 13}
    assert game.total_agents(s) == 15
    assert len(set(s["words"])) == 25


def test_same_seed_same_board():
    assert game.new_game(seed=7)["words"] == game.new_game(seed=7)["words"]
    assert game.new_game(seed=7)["key"] == game.new_game(seed=7)["key"]


def find(s, side, kind, exclude=()):
    return next(i for i, k in enumerate(s["key"][side]) if k == kind and i not in exclude and not s["found"][i])


def test_clue_rules():
    s = game.new_game(seed=3)
    word = s["words"][0]
    with pytest.raises(game.RuleError):
        game.give_clue(s, "a", word, 1)
    with pytest.raises(game.RuleError):
        game.give_clue(s, "a", word + "S", 1)  # contains a board word
    with pytest.raises(game.RuleError):
        game.give_clue(s, "a", "two words", 1)
    with pytest.raises(game.RuleError):
        game.give_clue(s, "b", "ZZZ", 1)  # not b's turn
    assert game.give_clue(s, "a", " zzz ", 2)["clue"]["word"] == "ZZZ"


def test_agent_then_bystander_ends_turn_and_marks():
    s = game.give_clue(game.new_game(seed=4), "a", "ZZZ", 2)
    g = find(s, "a", "G")
    s = game.guess(s, "b", g)
    assert s["found"][g] and s["phase"] == "guess"
    n = find(s, "a", "N")
    s = game.guess(s, "b", n)
    assert n in s["bystanders"]["a"] and s["phase"] == "clue"
    assert s["giver"] == "b" and s["turns_left"] == game.TURNS - 1


def test_assassin_loses():
    s = game.give_clue(game.new_game(seed=5), "a", "ZZZ", 1)
    s = game.guess(s, "b", find(s, "a", "A"))
    assert s["phase"] == "over" and s["result"] == "assassin"


def test_must_guess_before_stopping():
    s = game.give_clue(game.new_game(seed=6), "a", "ZZZ", 1)
    with pytest.raises(game.RuleError):
        game.stop(s, "b")


def test_view_hides_partner_key_until_over():
    s = game.new_game(seed=8)
    v = game.view(s, "a")
    assert v["my_key"] == s["key"]["a"] and v["partner_key"] is None
    assert "key" not in v and "seed" not in v


def test_self_play_always_finishes():
    results = Counter(self_play(RandomPlayer(i), RandomPlayer(i + 1000), seed=i)["result"] for i in range(200))
    assert sum(results.values()) == 200
    assert set(results) <= {"won", "assassin", "out_of_turns"}


def test_api_flow_hides_ai_key():
    client = TestClient(create_app(MemoryStore()))
    r = client.post("/api/games", json={"first": "human", "seed": 11})
    assert r.status_code == 200
    v = r.json()
    assert v["phase"] == "clue" and v["giver"] == "a" and v["partner_key"] is None
    assert "key" not in v
    r = client.post(f"/api/games/{v['id']}/clue", json={"word": "zzz", "number": 1})
    assert r.status_code == 200
    v = r.json()
    # The random partner guessed, then (unless the game ended) gave its own clue.
    assert v["phase"] in ("guess", "over")
    bad = client.post(f"/api/games/{v['id']}/clue", json={"word": v["words"][0], "number": 1})
    assert bad.status_code == 400


def test_glove_partner_plays_legal_games():
    from pathlib import Path

    from cluoduo.glove import DEFAULT_VECTORS

    if not Path(DEFAULT_VECTORS).exists():
        pytest.skip("word vectors not built (scripts/prepare_vectors.py)")
    from cluoduo.players import PLAYERS

    for name in ("glove-sum", "glove-bayes", "glove-rank"):
        _plays_legal_games(PLAYERS[name](), PLAYERS[name](guess_noise=0.05))


def _plays_legal_games(a, b):
    for seed in range(5):
        s = self_play(a, b, seed=seed)
        assert s["phase"] == "over"
        for h in s["history"]:
            if h["type"] == "clue":
                assert h["word"].isalpha() and h["word"] not in s["words"]


def test_key_prior_matches_layout_at_start():
    from cluoduo.glove import key_prior

    s = game.give_clue(game.new_game(seed=1, first_clue="b"), "b", "ZZZ", 1)
    prior = key_prior(game.view(s, "a"))
    mine = s["key"]["a"]
    agent = next(i for i in prior if mine[i] == "G")
    assassin = next(i for i in prior if mine[i] == "A")
    assert prior[agent] == pytest.approx({"G": 3 / 9, "N": 5 / 9, "A": 1 / 9})
    assert prior[assassin] == pytest.approx({"G": 1 / 3, "N": 1 / 3, "A": 1 / 3})


def test_key_prior_updates_after_a_revealed_bystander():
    from cluoduo.glove import key_prior

    s = game.give_clue(game.new_game(seed=1, first_clue="b"), "b", "ZZZ", 1)
    mine, theirs = s["key"]["a"], s["key"]["b"]
    hit = next(i for i in range(25) if mine[i] == "G" and theirs[i] == "N")
    s = game.guess(s, "a", hit)  # bystander on b's key: turn passes to a, then back to b
    s = game.give_clue(s, "a", "ZZZ", 1)
    s = game.guess(s, "b", next(i for i in range(25) if mine[i] == "N" and theirs[i] != "A" and not s["found"][i] and i not in s["bystanders"]["a"]))
    if s["phase"] == "guess":
        s = game.stop(s, "b")
    s = game.give_clue(s, "b", "ZZZ", 1)
    prior = key_prior(game.view(s, "a"))
    agent = next(i for i in prior if mine[i] == "G")
    assert prior[agent] == pytest.approx({"G": 3 / 8, "N": 4 / 8, "A": 1 / 8})
