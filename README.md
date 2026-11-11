# Cluo Duo

A free online co-operative word game in the Duet style, played with an AI partner. You and your partner each hold a secret key to the same 25 words; take turns giving one-word clues to find all 15 agents within 9 turns without hitting an assassin.

**Play:** https://ryanward.dev/cluo-duo/

The point of the project is the AI partner: comparing word-vector and LLM partners and measuring how well each one plays with someone it has never played with before. With a fixed reference player, the live partner (the open model Gemma 4) finds 11.2 of 15 agents on average, compared with 9.0 for GloVe word vectors and 11.2 for Claude Sonnet in the same seat, and it lost one game to an assassin to Sonnet's four.

## Partners

| Partner | How it plays |
|---|---|
| Random | Random clues and guesses, as a floor. |
| Basic | Clues and guesses from [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors, with Bayesian guessing: it weighs each guess by what the board and its own key reveal, and stops when the assassin risk gets too high. |
| Advanced | Google's Gemma 4 31B-it (Apache 2.0), as Red Hat AI's FP8 quantisation, served with vLLM on a GPU of mine. It brainstorms clues, rehearses each one by playing the guesser, and gives the one its rehearsal reads best. It does its own guessing too, without GloVe. |

## How partners are measured

Self-play (a partner playing a copy of itself) is cheap but flattering, because the two copies share a private language. The real test pairs each partner with a fixed reference player (Claude Sonnet, following the player guide in `prompts/player.md`) on the same 8 boards (seeds 200 to 207).

Average agents found out of 15 with the reference player, measured 27 Oct to 9 Nov 2026:

| Partner | Avg found | Assassin losses | Games |
|---|---|---|---|
| Random | 2.8 | 7 | 8 |
| Primitive (GloVe alone; no longer offered) | 6.0 | 4 | 4 (boards 200 to 203 only) |
| Basic | 9.0 | 5 | 8 (1 win) |
| Qwen3.6-35B-A3B (clues only) | 8.6 | 4 | 8 |
| Mistral Small 3.2 24B (clues only) | 8.5 | 6 | 8 |
| Qwen3.6-27B (clues only) | 9.4 | 4 | 8 |
| gpt-oss-120b (clues only) | 10.4 | 2 | 8 |
| Gemma 4 31B (clues only) | 10.4 | 4 | 8 |
| Gemma 4 31B, model also guesses | 11.0 | 1 | 8 |
| Claude Sonnet, same player guide (paid API; not offered) | 11.2 | 4 | 8 (2 wins) |
| Gemma 4 31B, no GloVe (28 Oct version) | 13.0 | 0 | 8 (1 win) |
| Advanced 31 Oct to 10 Nov: Gemma 4 31B, no GloVe, compact replies | 11.2 | 1 | 8 |
| **Advanced (live since 11 Nov): the same, with trusting guesses** | **12.7** | **4** (1 on the partner's guess) | 16 (4 wins) |

"Clues only" means the model gives the clues and Basic's Bayesian guesser does the guessing. In those variants GloVe also vetted the model's clues for safety. Most of their assassin losses came from GloVe misreading the reference player's clues, and dropping GloVe altogether scored higher on all 8 boards (sign test p ≈ 0.008), with the first win.

The live partner differs from the 28 Oct version in two ways that stopped occasional 30-second stalls: replies are compact JSON, and guess and rehearsal replies are capped at 250 tokens. In the first comparison it scored lower on 7 of the 8 boards (p ≈ 0.07), but the two were played by different reference-player sessions. A rerun on 10 Nov 2026 put both versions on the same 8 boards three times, side by side in one session (24 games each): the 28 Oct version found 10.9 agents on average and the live one 10.6, a difference of +0.25 (95% bootstrap interval −1.4 to +1.8; better on 13 boards, worse on 8, p ≈ 0.38). So the speed fix shows no clear cost in strength. The same version's average moved by up to 1.9 agents between rounds, which is about the size of the original gap. These reference sessions hit more assassins than earlier ones, so their averages sit below the table's.

Trusting guesses (11 Nov 2026). Players noticed the partner skipping the obvious answer to their clue when that word was an assassin, or a known bystander, on its own key. But a clue is about the clue giver's key, and the giver has already checked it against their own assassins. The guess prompt now says so: the partner's own assassins only break ties between equally good fits. In a replay of 419 past guess turns (`scripts/guess_replay.py`), the partner found 1.60 agents per clue instead of 1.38, with no clear rise in assassin hits. When it did guess one of its own assassins, that word was the clue giver's agent 91% of the time. In 48 blind reference games (16 each for the old prompt, the new one, and the new one plus a list of unfinished earlier clues, which the model never actually used), the old prompt found 11.3 agents on average with 1 win. The two new versions found 12.7 and 12.6, with 4 wins each. Paired by board and round, the new prompt was ahead by +1.3 (95% bootstrap interval −0.2 to +3.3; better on 9, worse on 5, tied on 2). The partner guessed its own assassin 29 times and none was the clue giver's assassin.

Against Claude Sonnet in the partner's seat, the live partner scored the same on average (higher on 3 boards, lower on 4, p = 1). Sonnet won two games outright but lost four to assassins, so Gemma is comparable here. Gemma isn't free: the GPU is billed by the hour, about $0.60 for one player's session (Modal bill, 30 Oct 2026), less per game when games run back to back.

These are small samples: 8 games per row (4 for Primitive), so differences under about 1.5 agents don't mean much. The games behind the table are in `data/agent-games/`; `uv run python scripts/agent_results.py --rounds` recomputes it.

## How it's built

- `src/cluoduo/game.py`: the rules, as pure functions over a JSON-friendly state. The same code runs the web game and the evaluations. Clue checks reject board words, substrings, spelling variants and word forms.
- `src/cluoduo/glove.py`: the GloVe partners (Basic, and Primitive for testing).
- `src/cluoduo/llm.py`: the LLM partners. A player implements `give_clue(view)` and `next_guess(view)`, and only ever sees what its side is allowed to know (`game.view`). If the model call fails, GloVe covers the move.
- `src/cluoduo/players.py`: the registry of every partner variant that has been tested, by name.
- `src/cluoduo/api.py`: a small FastAPI app. The human is side `a`; the server plays side `b` and keeps its key hidden until the game ends. Browsers only let pages on ryanward.dev read its responses (CORS), but that isn't access control: the API is public, so cost limits are enforced on the server: a daily cap on GPU games, and a monthly budget. Modal's $30 workspace usage limit would stop Basic too, so once this month's usage (read from Modal's billing API, at most hourly) reaches $27, the server stops starting Advanced games.
- `modal_app.py`: the game API's infrastructure on [Modal](https://modal.com): container image, game storage (a Modal Dict), the web endpoint, and cost limits (scale to zero, at most two containers, 10-minute request timeout).
- `llm_server.py`: the model servers (vLLM on Modal GPUs). They have no public URL, scale to zero, and run at most one container each.
- `scripts/`: self-play evaluation (`evaluate.py`, `compare.py`), reference-partner games (`agent_play.py`, `agent_results.py`) and game transcripts (`transcript.py`).
- `index.html`, `assets/`: a static front end on GitHub Pages that calls the API.

## Running locally

The GloVe partners need word vectors, which aren't in the repo (the source file is 5 GB):

1. Download `glove.42B.300d.zip` from the [GloVe page](https://nlp.stanford.edu/projects/glove/) and unzip it.
2. Save the public-domain [ENABLE word list](https://github.com/dolph/dictionary/blob/master/enable1.txt) as `data/raw/enable1.txt`.
3. Run `uv run python scripts/prepare_vectors.py path/to/glove.42B.300d.txt`. This writes `data/glove50k.npz`: the 50,000 most common real words, as float16.

Then:

```sh
uv sync
uv run pytest
uv run python dev.py                  # API on http://localhost:8001 (in-memory games)
python3 -m http.server 8010           # page on http://localhost:8010/?api=http://localhost:8001
```

Deploy the API with `uv run modal deploy modal_app.py` and the model servers with `uv run modal deploy llm_server.py`.

To replay a reference game, start it with `uv run python scripts/agent_play.py new --seed 200 --partner smart --tag mytest`, then have the reference player (a Claude Sonnet session given `prompts/player.md`) play side `a` using only the `show`, `clue`, `guess` and `stop` commands that script prints. It must never read the game file, which holds both keys.

## Licence

MIT for the code. A game in the Duet style; not affiliated with Codenames or Czech Games Edition. The 354 board words are my own list.

- Word vectors: [GloVe](https://nlp.stanford.edu/projects/glove/) 42B (Common Crawl), by Jeffrey Pennington, Richard Socher and Christopher D. Manning (2014), released under the Public Domain Dedication and License.
- Vocabulary filter: the public-domain ENABLE word list.
- Advanced partner: Google's Gemma 4 31B-it (Apache 2.0), via Red Hat AI's FP8 quantisation.
