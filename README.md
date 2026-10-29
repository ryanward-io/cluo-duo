# Cluo Duo

A free online co-operative word game in the Duet style, played with an AI partner. You and your partner each hold a secret key to the same 25 words; take turns giving one-word clues to find all 15 agents within 9 turns without hitting an assassin.

**Play:** https://ryanward.dev/cluo-duo/

The point of the project is the AI partner: comparing word-vector and LLM partners and measuring how well each one plays with someone it has never played with before.

## Partners

| Partner | How it plays |
|---|---|
| Random | Random clues and guesses, as a floor. |
| Basic | Clues and guesses from [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors, with Bayesian guessing: it weighs each guess by what the board and its own key reveal, and stops when the assassin risk gets too high. |
| Advanced | Gemma 4 31B, an open model on our own GPU, playing on its own. It brainstorms clues, rehearses each one by playing the guesser, and gives the one its rehearsal reads best; it does its own guessing too. |

## How partners are measured

Self-play (a partner playing a copy of itself) is cheap but flattering, because the two copies share a private language. The real test pairs each partner with a fixed reference player (Claude Sonnet, following the player guide in `prompts/player.md`) on the same boards (seeds 200 to 207).

Average agents found out of 15 with the reference player:

| Partner | Avg found | Assassin losses | Games |
|---|---|---|---|
| Random | 2.8 | 4 | 4 |
| Primitive (GloVe alone; no longer offered) | 6.0 | 4 | 4 |
| Basic | 8.2 | 2 | 4 (1 win) |
| Qwen3.6-35B-A3B (clues only) | 8.6 | 4 | 8 |
| Mistral Small 3.2 24B (clues only) | 8.5 | 6 | 8 |
| Qwen3.6-27B (clues only) | 9.4 | 4 | 8 |
| gpt-oss-120b (clues only) | 10.4 | 2 | 8 |
| Gemma 4 31B (clues only) | 10.4 | 4 | 8 |
| Gemma 4 31B, model also guesses | 11.0 | 1 | 8 |
| Claude Sonnet, same player guide (paid API; not offered) | 11.2 | 4 | 8 (2 wins) |
| **Advanced: Gemma 4 31B, no GloVe (live)** | **13.0** | **0** | 8 (1 win) |

"Clues only" means the model gives the clues and Basic's Bayesian guesser does the guessing. In every variant before the live one, GloVe also vetted the model's clues for safety. Most of the assassin losses came from GloVe misreading the reference player's clues, and dropping GloVe altogether scored higher on all 8 boards, with the first win. Claude Sonnet in the partner's seat won two games outright but lost four to assassins, so Gemma at least matches it here, with no cost per game.

## How it's built

- `src/cluoduo/game.py`: the rules, as pure functions over a JSON-friendly state. The same code runs the web game and the evaluations. Clue checks reject board words, substrings, spelling variants and word forms.
- `src/cluoduo/glove.py`: the GloVe partners (Basic, and Primitive for testing).
- `src/cluoduo/llm.py`: the LLM partners. A player implements `give_clue(view)` and `next_guess(view)`, and only ever sees what its side is allowed to know (`game.view`). If the model call fails, GloVe covers the move.
- `src/cluoduo/players.py`: the registry of every partner variant that has been tested, by name.
- `src/cluoduo/api.py`: a small FastAPI app. The human is side `a`; the server plays side `b` and keeps its key hidden until the game ends. Browsers only let pages on ryanward.dev read its responses (CORS), but that isn't access control: the API is public, so cost limits are enforced on the server, including a daily cap on GPU games.
- `modal_app.py`: the game API's infrastructure on [Modal](https://modal.com): container image, game storage (a Modal Dict), the web endpoint, and cost limits (scale to zero, at most two containers, 10-minute request timeout).
- `llm_server.py`: the model servers (vLLM on Modal GPUs). They have no public URL, scale to zero, and run at most one container each.
- `scripts/`: self-play evaluation (`evaluate.py`, `compare.py`), reference-partner games (`agent_play.py`, `agent_results.py`) and game transcripts (`transcript.py`).
- `index.html`, `assets/`: a static front end on GitHub Pages that calls the API.

## Running locally

```sh
uv sync
uv run pytest
uv run python dev.py                  # API on http://localhost:8001 (in-memory games)
python3 -m http.server 8010           # page on http://localhost:8010/?api=http://localhost:8001
```

Deploy the API with `uv run modal deploy modal_app.py` and the model servers with `uv run modal deploy llm_server.py`.

## Licence

MIT for the code. A game in the Duet style; not affiliated with Codenames or Czech Games Edition. The word list is our own.
