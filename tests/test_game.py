import json
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
    with_sudden_death = Counter(self_play(RandomPlayer(i), RandomPlayer(i + 1000), seed=i, sudden_death=True)["result"]
                                for i in range(50))
    assert set(with_sudden_death) <= {"won", "assassin", "sudden_death"}


def test_sudden_death():
    s = game.new_game(seed=3, first_clue="a", sudden_death=True)
    s["turns_left"] = 1
    s = game.give_clue(s, "a", "ZZZ", 1)
    bystander = next(i for i, k in enumerate(s["key"]["a"]) if k == "N")
    s = game.guess(s, "b", bystander)  # ends the last turn
    assert s["phase"] == "sudden_death" and s["guesser"] == "b"  # the side that was guessing goes on
    with pytest.raises(game.RuleError):
        game.give_clue(s, "a", "ZZZ", 1)
    agent = next(i for i, k in enumerate(s["key"]["a"]) if k == "G")  # judged by the guesser's partner's key
    s = game.sudden_death_guess(s, "b", agent)
    assert s["found"][agent] and s["guesser"] == "a"  # turns alternate
    wrong = next(i for i, k in enumerate(s["key"]["b"]) if k == "N" and not s["found"][i])
    s = game.sudden_death_guess(s, "a", wrong)
    assert s["phase"] == "over" and s["result"] == "sudden_death"


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


def _fake_backend(replies):
    calls = []

    def chat(messages, **kwargs):
        calls.append(messages)
        return {"content": json.dumps(replies[min(len(calls), len(replies)) - 1])}

    return chat, calls


def test_llm_clue_retries_with_the_reason_then_accepts():
    from cluoduo.llm import LLMPlayer

    s = game.new_game(seed=2, first_clue="b")
    v = game.view(s, "b")
    agent = next(w for w, k in zip(v["words"], v["my_key"]) if k == "G")
    chat, calls = _fake_backend([
        {"targets": [agent], "reason": "", "clue": v["words"][0], "number": 1},  # a board word: not allowed
        {"targets": [agent], "reason": "", "clue": "ocean", "number": 3},
    ])
    clue, number = LLMPlayer(backend=chat, fallback=RandomPlayer(0)).give_clue(v)
    assert clue == "OCEAN" and number == 1  # number capped at the valid targets it named
    assert len(calls) == 2 and "isn't allowed" in calls[1][-1]["content"]


def test_llm_falls_back_after_repeated_bad_clues():
    from cluoduo.llm import MAX_ATTEMPTS, LLMPlayer

    v = game.view(game.new_game(seed=2, first_clue="b"), "b")
    chat, calls = _fake_backend([{"targets": [], "reason": "", "clue": "", "number": 1}])
    clue, number = LLMPlayer(backend=chat, fallback=RandomPlayer(0)).give_clue(v)
    assert len(calls) == MAX_ATTEMPTS and clue.isalpha()


def test_llm_guesses_in_order_and_skips_nonsense():
    from cluoduo.llm import LLMPlayer

    s = game.give_clue(game.new_game(seed=2, first_clue="b"), "b", "ZZZ", 2)
    v = game.view(s, "a")
    w = s["words"]
    chat, calls = _fake_backend([{"reason": "", "guesses": ["not on board", w[3].lower(), w[3], w[7]]}])
    p = LLMPlayer(backend=chat, fallback=RandomPlayer(0))
    assert p.next_guess(v) == 3
    v["clue"] = dict(v["clue"], guesses=1)
    assert p.next_guess(v) == 7 and len(calls) == 1  # one model call per turn


def test_hybrid_vets_candidates_with_glove():
    from cluoduo.glove import GloveRankPlayer
    from cluoduo.llm import HybridPlayer

    v = game.view(game.new_game(seed=2, first_clue="b"), "b")
    agents = [w for w, k in zip(v["words"], v["my_key"]) if k == "G"]
    candidates = [
        {"targets": agents[:3], "clue": v["words"][0]},  # a board word
        {"targets": agents[:2], "clue": "zzzqx"},  # not in the vocabulary
        {"targets": agents[:3], "clue": "water"},
        {"targets": agents[3:6], "clue": "music"},
    ]
    chat, calls = _fake_backend([{"candidates": candidates}])
    p = HybridPlayer(backend=chat, fallback=GloveRankPlayer(0))
    clue, number = p.give_clue(v)
    results = [p.vet(v, c) for c in candidates]
    assert results[0]["number"] == 0 and "board" in results[0]["problem"]
    assert results[1]["number"] == 0 and "vocabulary" in results[1]["problem"]
    safe = [r for r in results if r["number"] > 0]
    for r in safe:
        assert set(r["kept"]) <= set(r["targets"]) and r["margin"] >= 0
    if safe:
        best = max(safe, key=lambda r: (r["number"], r["margin"]))
        assert (clue, number) == (best["clue"], best["number"])
    else:
        assert p.log[-1] == {"fallback": "clue"}  # nothing safe: GloVe v2 gives the clue
    assert len(calls) == 1


def test_simulating_player_trusts_the_rehearsal():
    from cluoduo.glove import GloveRankPlayer
    from cluoduo.llm import SimulatingPlayer

    v = game.view(game.new_game(seed=2, first_clue="b"), "b")
    words, key = v["words"], v["my_key"]
    agents = [w for w, k in zip(words, key) if k == "G"]
    assassin = next(w for w, k in zip(words, key) if k == "A")
    p = SimulatingPlayer(backend=lambda *a, **k: {"content": "{}"}, fallback=GloveRankPlayer(0))
    safe = [
        {"clue": "ALPHA", "targets": agents[:3], "number": 3, "margin": 0.5},  # rehearsal hits an assassin second
        {"clue": "BETA", "targets": agents[3:5], "number": 2, "margin": 0.1},  # rehearsal finds both
    ]
    bystander = next(w for w, k in zip(words, key) if k == "N")
    # ALPHA: one agent, then the assassin. BETA: two agents, then a bystander (ends the run safely).
    replies = {"ALPHA": [agents[0], assassin], "BETA": [agents[3], agents[4], bystander]}

    def chat(messages, **kwargs):
        clue = next(c for c in replies if f"clue is {c} " in messages[-1]["content"])
        return {"content": json.dumps({"reason": "", "guesses": replies[clue]})}

    p.backend = chat
    assert p.choose(v, safe) == ("BETA", 2)
    rehearsed = {r["clue"]: r for r in p.log[-1]["rehearsed"]}
    assert rehearsed["ALPHA"]["run"] == 0 and rehearsed["ALPHA"]["stopped_on"] == "assassin"
    assert rehearsed["BETA"]["stopped_on"] == "bystander"


def test_rehearsals_report_progress():
    from cluoduo.glove import GloveRankPlayer
    from cluoduo.llm import SimulatingPlayer

    v = game.view(game.new_game(seed=2, first_clue="b"), "b")
    agents = [w for w, k in zip(v["words"], v["my_key"]) if k == "G"]
    p = SimulatingPlayer(backend=lambda *a, **k: {"content": json.dumps({"reason": "", "guesses": [agents[0]]})},
                         fallback=GloveRankPlayer(0))
    seen = []
    p.progress = lambda text, answered: seen.append((text, answered))
    p.choose(v, [{"clue": c, "targets": agents[:1], "number": 1, "margin": 0} for c in ("ALPHA", "BETA")])
    assert seen[0] == ("Your partner is testing its 2 clue ideas by playing your side…", False)
    assert seen[-1] == ("Your partner is testing its 2 clue ideas by playing your side: 2 of 2 done…", True)


def test_spelling_variants_of_board_words_are_not_clues():
    s = game.new_game(seed=1)
    s["words"][0] = "HARBOUR"
    with pytest.raises(game.RuleError, match="spelling"):
        game.check_clue(s, "harbor", 1)
    assert game.spelling_variant("COLOR", "COLOUR") and not game.spelling_variant("HOUSE", "HORSE")


def test_api_board_first_then_partner_clue():
    client = TestClient(create_app(MemoryStore()))
    v = client.post("/api/games", json={"first": "partner", "seed": 11, "partner_moves": False}).json()
    assert v["phase"] == "clue" and v["giver"] == "b" and v["history"] == []
    v = client.post(f"/api/games/{v['id']}/partner").json()
    assert v["history"][0]["type"] == "clue" and v["phase"] == "guess"
    again = client.post(f"/api/games/{v['id']}/partner").json()  # the human's move now: nothing changes
    assert again["history"] == v["history"]
    progress = client.get(f"/api/games/{v['id']}/progress").json()
    assert progress["text"] is None and progress["awake"] is False  # no GPU partner has answered yet


def test_api_sudden_death_guess():
    store = MemoryStore()
    client = TestClient(create_app(store))
    v = client.post("/api/games", json={"first": "human", "seed": 5, "partner": "glove-rank", "sudden_death": True}).json()
    state = store.get(v["id"])
    state.update(phase="sudden_death", guesser="a", turns_left=0, clue=None, giver="a")
    store.put(state)
    agent = next(i for i, k in enumerate(state["key"]["b"]) if k == "G" and not state["found"][i])
    r = client.post(f"/api/games/{v['id']}/guess", json={"index": agent}).json()
    assert r["found"][agent] and r["phase"] in ("sudden_death", "over")


def test_ai_clue_vocabulary_has_no_blocked_words():
    from cluoduo.blocked import is_blocked
    from cluoduo.glove import load_vectors

    v = load_vectors()
    assert not [v.words[i] for i in v.clue_ids if is_blocked(v.words[i])]
    assert is_blocked("kinky") and is_blocked("WOKE") and not is_blocked("BUTTER") and not is_blocked("GRAPE")


def test_glove_partners_never_repeat_a_clue():
    from cluoduo.glove import GloveRankPlayer, GloveSumPlayer

    for player in (GloveRankPlayer(0), GloveSumPlayer(0)):
        s = game.new_game(seed=8, first_clue="b")
        clues = []
        for _ in range(4):  # the guesser never finds anything, so the same clue would stay "best"
            word, number = player.give_clue(game.view(s, "b"))
            clues.append(word)
            s = game.give_clue(s, "b", word, number)
            s["phase"], s["clue"] = "clue", None  # skip the guessing: side b gives the next clue too
        assert len(set(clues)) == len(clues), clues


def test_guide_player_rejects_clues_its_own_check_flags():
    from cluoduo.glove import GloveRankPlayer
    from cluoduo.guide_player import GuidePlayer

    v = game.view(game.new_game(seed=2, first_clue="b"), "b")
    words, key = v["words"], v["my_key"]
    agents = [w for w, k in zip(words, key) if k == "G"]
    assassin = next(w for w, k in zip(words, key) if k == "A")
    others = [w for w, k in zip(words, key) if k == "N"]
    chat, calls = _fake_backend([
        {"targets": agents[:2], "clue": "planet", "partner_would_guess": [agents[0], assassin, agents[1], others[0]]},
        {"targets": agents[:2], "clue": "music", "partner_would_guess": [agents[0], others[0], others[1], others[2]]},
    ])
    p = GuidePlayer(backend=chat, fallback=GloveRankPlayer(0))
    assert p.give_clue(v) == ("MUSIC", 1)  # only one target made its own rehearsal list
    assert "ASSASSIN" in calls[1][-1]["content"]


def test_guide_player_takes_own_assassin_only_first():
    from cluoduo.glove import GloveRankPlayer
    from cluoduo.guide_player import GuidePlayer

    s = game.give_clue(game.new_game(seed=2, first_clue="b"), "b", "ZZZ", 2)
    v = game.view(s, "a")
    words, key = v["words"], v["my_key"]
    mine_a = next(w for w, k in zip(words, key) if k == "A")
    other = next(w for w, k in zip(words, key) if k == "N")
    chat, _ = _fake_backend([{"reason": "", "guesses": [other, mine_a]}])
    p = GuidePlayer(backend=chat, fallback=GloveRankPlayer(0))
    assert p._plan_guesses(v) == [words.index(other)]  # own assassin dropped as a second guess


def test_extra_turns_setting():
    assert game.new_game(seed=1, turns=11)["turns_left"] == 11
    with pytest.raises(game.RuleError):
        game.new_game(seed=1, turns=12)
    client = TestClient(create_app(MemoryStore()))
    assert client.post("/api/games", json={"first": "human", "turns": 10}).json()["turns_left"] == 10
    assert client.post("/api/games", json={"first": "human", "turns": 17}).status_code == 422


def test_forms_of_compound_parts_are_not_clues():
    s = game.new_game(seed=1)
    s["words"][0], s["words"][1] = "MAILBOX", "STRAWBERRY"
    for clue in ("mails", "mailed", "boxes", "berries"):
        with pytest.raises(game.RuleError, match="form of"):
            game.check_clue(s, clue, 1)
    assert game.check_clue(s, "letters", 1) == "LETTERS"


def test_smart_partner_daily_cap_and_warm_up(monkeypatch):
    from cluoduo import api

    monkeypatch.setattr(api, "GPU_GAMES_PER_DAY", 1)
    monkeypatch.setitem(api.PLAYERS, "smart", lambda seed=None: RandomPlayer(seed))  # no GPU in tests
    warmed = []
    client = TestClient(create_app(MemoryStore(), warm_up=warmed.append))
    assert client.post("/api/warm", json={"partner": "smart"}).json() == {"ok": True} and warmed == ["Gemma31"]
    assert client.post("/api/games", json={"first": "human", "partner": "smart"}).status_code == 200
    assert client.post("/api/games", json={"first": "human", "partner": "smart"}).status_code == 429
    assert client.post("/api/warm", json={"partner": "glove-rank"}).json() == {"ok": False}


def test_glove_guesser_stops_after_one_blind_guess_on_unknown_clue():
    from cluoduo.glove import GloveRankPlayer

    s = game.give_clue(game.new_game(seed=4, first_clue="a"), "a", "RUDOLPH", 2)  # not in the vocabulary
    p = GloveRankPlayer(0)
    first = p.next_guess(game.view(s, "b"))
    assert first is not None  # the rules require one guess
    s2 = game.guess(s, "b", first)
    if s2["phase"] == "guess":
        assert p.next_guess(game.view(s2, "b")) is None


def test_api_warns_about_clues_a_glove_partner_cannot_read():
    client = TestClient(create_app(MemoryStore()))
    g = client.post("/api/games", json={"first": "human", "partner": "glove-rank", "seed": 3}).json()
    r = client.post(f"/api/games/{g['id']}/clue", json={"word": "rudolph", "number": 2})
    assert r.status_code == 409 and r.json()["detail"] == {"unknown_word": "RUDOLPH"}
    r = client.post(f"/api/games/{g['id']}/clue", json={"word": "rudolph", "number": 2, "anyway": True})
    assert r.status_code == 200
