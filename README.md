# Clue Words Duet

A free online co-operative word game in the Duet style, played with an AI partner. You and your partner each hold a secret key to the same 25 words; take turns giving one-word clues to find all 15 agents within 9 turns without hitting an assassin.

**Play:** https://ryanward.dev/clue-words-duet/

The point of the project is the AI partner: comparing word-vector and LLM partners, measuring them with self-play, and eventually training one. Right now the partner is a **random placeholder** so the whole game works end to end before any AI goes in.

## How it's built

- `src/cluewords/game.py`: the rules, as pure functions over a JSON-friendly state. The same code runs the web game and (later) the self-play evaluation.
- `src/cluewords/players.py`: the player interface. A player implements `give_clue(view)` and `next_guess(view)`, and only ever sees what its side is allowed to know (`game.view`).
- `src/cluewords/api.py`: a small FastAPI app. The human is side `a`; the server plays side `b` and keeps its key hidden until the game ends.
- `modal_app.py`: the infrastructure, deployed on [Modal](https://modal.com): container image, game storage (a Modal Dict), the web endpoint, and cost limits (scale to zero, at most two containers, 60 s request timeout).
- `index.html`, `assets/`: a static front end on GitHub Pages that calls the API.

## Running locally

```sh
uv sync
uv run pytest
uv run python dev.py                  # API on http://localhost:8001 (in-memory games)
python3 -m http.server 8010           # page on http://localhost:8010/?api=http://localhost:8001
```

Deploy the API with `uv run modal deploy modal_app.py`.

## Licence

MIT for the code. An original game in the Duet style; not affiliated with Codenames or Czech Games Edition. The word list is our own.
