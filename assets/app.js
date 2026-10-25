// Front end for Clue Words Duet. All rules run on the server; this file draws the board and sends moves.

const params = new URLSearchParams(location.search);
const API = params.get("api") || "https://ryanward-io--clue-words-duet.us-east.modal.run";
const SAVE_KEY = "clue-words-duet:game";

const $ = (s) => document.querySelector(s);
const OUTCOME = { G: "agent", N: "bystander", A: "assassin" };
const RESULT = {
  won: "You found all 15 agents. You win!",
  assassin: "An assassin was hit. Game over.",
  out_of_turns: "Out of turns before all agents were found. Game over.",
};

let state = null;
let busy = false;
let setupOpen = true;

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : String(c));
  return n;
}

function save(id) {
  try { id ? localStorage.setItem(SAVE_KEY, id) : localStorage.removeItem(SAVE_KEY); } catch (_) { /* storage blocked */ }
}
function saved() {
  try { return localStorage.getItem(SAVE_KEY); } catch (_) { return null; }
}

async function call(path, body) {
  const res = await fetch(`${API}${path}`, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Server error (${res.status})`);
  }
  return data;
}

async function act(path, body, thinking = "Your partner is thinking…") {
  if (busy) return;
  busy = true;
  $("#error").textContent = "";
  renderAction(thinking);
  const slow = setTimeout(() => renderAction(`${thinking} (the server may be waking up; this can take a few seconds)`), 3000);
  const before = state;
  try {
    state = await call(path, body);
    save(state.phase === "over" ? null : state.id);
  } catch (err) {
    $("#error").textContent = err.message;
  } finally {
    clearTimeout(slow);
    busy = false;
  }
  render(before !== state ? before : null);
}

// ---------- rendering ----------

function render(before = null) {
  $("#setup").hidden = Boolean(state) && !setupOpen;
  $("#game").hidden = !state;
  if (!state) return;
  $("#found").textContent = `Agents found: ${state.found_count} / ${state.total_agents}`;
  $("#turns").textContent = `Turns left: ${state.turns_left}`;
  $("#partner-name").textContent = `Partner: ${state.partner}`;
  renderBoard(before);
  renderLog();
  renderAction();
}

function cardInfo(i) {
  const s = state;
  const mine = s.bystanders.a.includes(i);
  const theirs = s.bystanders.b.includes(i);
  const myKey = s.my_key[i];
  const partnerKey = s.partner_key?.[i];
  const guessable = s.phase === "guess" && s.clue?.giver === "b" && !s.found[i] && !theirs && !busy;
  const dead = !s.found[i] && ((mine && theirs) || (theirs && myKey !== "G"));
  const hitAssassin = s.result === "assassin" && s.history.at(-2)?.index === i;
  return { mine, theirs, myKey, partnerKey, guessable, dead, hitAssassin };
}

function renderBoard(before) {
  const board = $("#board");
  board.replaceChildren(...state.words.map((word, i) => {
    const c = cardInfo(i);
    const label = [word,
      state.found[i] && "agent found",
      c.myKey === "G" && "your agent", c.myKey === "A" && "your assassin",
      c.mine && "bystander on your key", c.theirs && "bystander on partner's key",
      c.partnerKey && `partner's key: ${OUTCOME[c.partnerKey]}`].filter(Boolean).join(", ");
    const changed = before && (before.found[i] !== state.found[i]
      || before.bystanders.a.includes(i) !== c.mine || before.bystanders.b.includes(i) !== c.theirs);
    const cls = ["card", state.found[i] && "found", c.dead && "dead", c.guessable && "guessable",
      c.hitAssassin && "assassin-hit", changed && "flash"].filter(Boolean).join(" ");
    const keyDots = el("span", { class: "key", "aria-hidden": "true" },
      c.myKey === "G" && el("i", { class: "dot agent" }), c.myKey === "A" && el("i", { class: "dot assassin" }),
      c.partnerKey === "G" && el("i", { class: "dot agent partner" }), c.partnerKey === "A" && el("i", { class: "dot assassin partner" }));
    const marks = el("span", { class: "marks", "aria-hidden": "true" },
      c.mine && el("i", { class: "mark mine" }, "✕"), c.theirs && el("i", { class: "mark theirs" }, "✕"));
    return el("button", {
      class: cls, type: "button", "aria-label": label, disabled: !c.guessable,
      onclick: () => act(`/api/games/${state.id}/guess`, { index: i }),
    }, keyDots, marks, el("span", { class: "word" }, word));
  }));
  fitWords();
}

function fitWords() {
  // Long words shrink until they fit on one line in their card.
  for (const card of document.querySelectorAll(".card")) {
    const word = card.querySelector(".word");
    word.style.fontSize = "";
    let size = parseFloat(getComputedStyle(card).fontSize);
    const room = card.clientWidth - 6;
    while (word.scrollWidth > room && size > 7) {
      size -= 0.5;
      word.style.fontSize = `${size}px`;
    }
  }
}
window.addEventListener("resize", () => { if (state) fitWords(); });

function renderLog() {
  const who = (side) => (side === "a" ? "You" : "Partner");
  const items = state.history.map((h) => {
    if (h.type === "clue") return el("li", {}, `${who(h.side)} gave the clue `, el("strong", {}, `${h.word} ${h.number}`));
    if (h.type === "guess") return el("li", {}, `${who(h.side)} guessed ${h.word}: `, el("span", { class: `outcome-${h.outcome}` }, OUTCOME[h.outcome]));
    if (h.type === "stop") return el("li", {}, `${who(h.side)} stopped guessing.`);
    return el("li", {}, el("strong", {}, RESULT[h.result]));
  });
  $("#log").replaceChildren(...items);
}

function renderAction(message) {
  const box = $("#action");
  if (message) { box.replaceChildren(el("p", {}, message)); return; }
  const s = state;
  if (s.phase === "over") {
    box.replaceChildren(
      el("p", { class: "big-clue" }, RESULT[s.result]),
      el("p", { class: "muted" }, "Your partner's key is now shown on the board (dashed dots)."),
      el("div", {}, el("button", { type: "button", onclick: () => { save(null); setupOpen = true; $("#setup").hidden = false; $("#setup").scrollIntoView(); } }, "New game")));
    return;
  }
  if (s.phase === "clue" && s.giver === "a") {
    const word = el("input", { type: "text", id: "clue-word", required: true, autocomplete: "off", spellcheck: "false", maxlength: 30 });
    const num = el("input", { type: "number", id: "clue-number", min: 0, max: 9, value: 1, required: true });
    const form = el("form", {
      class: "clue-form",
      onsubmit: (e) => { e.preventDefault(); act(`/api/games/${s.id}/clue`, { word: word.value, number: Number(num.value) }); },
    }, el("label", {}, "Clue word", word), el("label", {}, "Number", num), el("button", { type: "submit" }, "Give clue"));
    box.replaceChildren(el("p", {}, "Your turn to give a clue. Point your partner at your green-dot words, and steer clear of your assassins."), form);
    word.focus();
    return;
  }
  if (s.phase === "guess" && s.clue.giver === "b") {
    const stopBtn = el("button", {
      type: "button", class: "secondary", disabled: s.clue.guesses === 0 || busy,
      onclick: () => act(`/api/games/${s.id}/stop`, {}),
    }, "Stop guessing");
    box.replaceChildren(
      el("p", {}, "Your partner's clue:"),
      el("p", { class: "big-clue" }, `${s.clue.word} ${s.clue.number}`),
      el("p", { class: "muted" }, s.clue.guesses === 0
        ? "Click a highlighted word to guess."
        : `${s.clue.guesses} guessed this turn. Keep going or stop.`),
      el("div", {}, stopBtn));
    return;
  }
  box.replaceChildren(el("p", {}, "Waiting for your partner…"));
}

// ---------- start ----------

$("#start").addEventListener("click", () => {
  const first = document.querySelector('input[name="first"]:checked').value;
  state = null;
  setupOpen = false;
  render();
  $("#game").hidden = false;
  act("/api/games", { first, partner: $("#partner").value }, "Setting up the board…");
});

(async () => {
  const id = params.get("game") || saved();
  if (!id) return;
  try {
    state = await call(`/api/games/${id}`);
    setupOpen = false;
    render();
  } catch (_) {
    save(null);
  }
})();
