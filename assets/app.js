// Front end for Cluo Duo. All rules run on the server; this file draws the board and sends moves.

const params = new URLSearchParams(location.search);
const API = params.get("api") || "https://ryanward-io--cluo-duo.us-east.modal.run";
const SAVE_KEY = "cluo-duo:game";
const TIPS_KEY = "cluo-duo:tips";

const $ = (s) => document.querySelector(s);
const SKULL = '<svg class="skull" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" fill-rule="evenodd" d="M8 1C4.1 1 1.5 3.6 1.5 7c0 2 .9 3.4 2.3 4.2V13a1 1 0 0 0 1 1h.9v-1.5h1V14h2.6v-1.5h1V14h.9a1 1 0 0 0 1-1v-1.8c1.4-.8 2.3-2.2 2.3-4.2C14.5 3.6 11.9 1 8 1zM5.3 9.6a1.6 1.6 0 1 1 0-3.2 1.6 1.6 0 0 1 0 3.2zm5.4 0a1.6 1.6 0 1 1 0-3.2 1.6 1.6 0 0 1 0 3.2zM8 11.2l-.9-1.6h1.8z"/></svg>';
function skull(cls) {
  const i = el("i", { class: `assassin-icon ${cls}` });
  i.innerHTML = SKULL;
  return i;
}
const PARTNERS = { random: "Random", "glove-sum": "Primitive", "glove-bayes": "Primitive + Bayesian", "glove-rank": "Basic",
  smart: "Advanced" };
const GPU_PARTNERS = ["smart"]; // run on a GPU that may need a few minutes to wake up
const GPU_AWAKE_MS = 4 * 60 * 1000; // the GPU sleeps 5 minutes after its last move; allow a minute's margin
// Bystander marks: a pawn for your key, an upside-down pawn for your partner's.
const PAWN = '<svg class="pawn" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="4" r="2.7" fill="currentColor"/><path fill="currentColor" d="M5.4 7.4h5.2l-1 4h1.9l1.2 2.9H3.3l1.2-2.9h1.9z"/></svg>';
function pawn(cls) {
  const i = el("i", { class: `mark ${cls}` });
  i.innerHTML = PAWN;
  return i;
}
const GUESS_PACE_MS = 1000; // the partner's guesses are revealed one at a time, this far apart
const LOW_TURNS = 3; // warn when this few turns are left
const OUTCOME = { G: "agent", N: "bystander", A: "assassin" };
const RESULT = {
  won: "You found all 15 agents. You win!",
  assassin: "An assassin was hit. Game over.",
  out_of_turns: "Out of turns before all agents were found. Game over.",
  sudden_death: "A wrong guess in sudden death. Game over.",
};

let state = null;
let busy = false;
let setupOpen = true;
let generation = 0; // bumped when the player leaves a game, so late replies for it are ignored
let lastGpuReply = 0; // when a GPU partner last answered, to tell whether its GPU is still awake

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

// ---------- advanced tips: odds for the partner's key ----------

// How the two keys overlap (game.KEY_LAYOUT): for each colour on my key, how many words are
// each colour on my partner's. The layout is symmetric, so it reads the same from either side.
const LAYOUT = { G: { G: 3, N: 5, A: 1 }, N: { G: 5, N: 7, A: 1 }, A: { G: 1, N: 1, A: 1 } };
const NAMES = { G: "agent", N: "bystander", A: "assassin" };

function tipsOn() {
  try { return localStorage.getItem(TIPS_KEY) === "1"; } catch (_) { return false; }
}

// What the game so far has revealed about my partner's key: my guesses on their clues (and in
// sudden death) show the colour on their key; a found word that isn't my agent must be theirs.
function knownPartnerKey(s) {
  const known = {};
  for (const h of s.history) if (h.type === "guess" && h.side === "a") known[h.index] = h.outcome;
  s.found.forEach((f, i) => { if (f && s.my_key[i] !== "G") known[i] = "G"; });
  for (const i of s.bystanders.b) known[i] = "N";
  return known;
}

// P(colour on partner's key) for a word, given its colour on mine: the rules' counts, minus
// what's already been revealed, spread evenly over the words still unknown.
function partnerOdds(s) {
  const known = knownPartnerKey(s);
  const odds = {};
  for (const mine of "GNA") {
    const left = { ...LAYOUT[mine] };
    const unknown = [];
    s.my_key.forEach((k, i) => {
      if (k !== mine) return;
      if (i in known) left[known[i]] -= 1; else unknown.push(i);
    });
    const total = left.G + left.N + left.A || 1;
    odds[mine] = { G: left.G / total, N: left.N / total, A: left.A / total, words: unknown.filter((i) => !s.found[i]).length };
  }
  return odds;
}

const pct = (x) => `${Math.round(x * 100)}%`;

function renderTips() {
  const panel = $("#tips-panel");
  const s = state;
  panel.hidden = !tipsOn() || !s || s.phase === "over";
  if (panel.hidden) return;
  const left = s.total_agents - s.found_count;
  const pace = s.phase === "sudden_death"
    ? `Sudden death: ${left} agent${left === 1 ? "" : "s"} left, and every guess must be right.`
    : `${left} agents left in ${s.turns_left} turn${s.turns_left === 1 ? "" : "s"}: you need ${(left / Math.max(s.turns_left, 1)).toFixed(1)} per turn.`;
  const odds = partnerOdds(s);
  const rows = [["G", "Your agents"], ["N", "Your bystanders"], ["A", "Your assassins"]].map(([k, label]) => {
    const o = odds[k];
    return el("tr", {}, el("th", { scope: "row" }, label), el("td", {}, pct(o.G)), el("td", {}, pct(o.N)), el("td", { class: o.A >= 0.2 ? "risky" : "" }, pct(o.A)));
  });
  panel.replaceChildren(
    el("p", { class: "pace" }, pace),
    el("table", {},
      el("caption", {}, "What a word you can't see yet is likely to be on your partner's key"),
      el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", { scope: "col" }, "their agent"), el("th", { scope: "col" }, "bystander"), el("th", { scope: "col" }, "assassin"))),
      el("tbody", {}, ...rows)),
    el("p", { class: "muted hint" }, "From the rules: of your 9 agents, 3 are theirs too, 5 are their bystanders and 1 is their assassin; your 3 assassins are one of each for them. The odds update as guesses reveal their key."));
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
    if (data.detail?.unknown_word) throw Object.assign(new Error("unknown word"), { unknownWord: data.detail.unknown_word });
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw Object.assign(new Error(detail || `Server error (${res.status})`), { status: res.status });
  }
  return data;
}

// True while the server owes the partner's move.
function partnersTurn(s) {
  return (s.phase === "clue" && s.giver === "b") || (s.phase === "guess" && s.clue.giver === "a")
    || (s.phase === "sudden_death" && s.guesser === "b");
}

// A move request failed, but the server may still have made the move: a slow partner can outlast the
// connection. Fetch the game and, if it moved on, wait for the partner to finish (up to 10 minutes).
async function resync(before, err, gen) {
  if (!before) return null;
  for (let tries = 0; tries < 120; tries++) {
    if (gen !== generation) return null;
    const s = await call(`/api/games/${before.id}`).catch(() => null);
    const moved = s && s.history.length > before.history.length;
    if (moved && !partnersTurn(s)) return s;
    if (s && !moved && err.status >= 400 && err.status < 500) return null; // a real rule error: nothing happened
    await new Promise((r) => setTimeout(r, 5000));
  }
  return null;
}

async function act(path, body, thinking = "Your partner is thinking…") {
  if (busy) return;
  busy = true;
  $("#error").textContent = "";
  renderAction(thinking);
  const gpu = GPU_PARTNERS.includes(state?.partner ?? document.querySelector('input[name="partner"]:checked').value);
  const awake = Date.now() - lastGpuReply < GPU_AWAKE_MS;
  const started = Date.now();
  const before = state;
  // A GPU partner reports each step it takes (reading the clue, brainstorming, testing clues).
  let step = null;
  const poll = gpu && before ? setInterval(async () => {
    step = await call(`/api/games/${before.id}/progress`).catch(() => step);
  }, 2000) : null;
  // After 3 seconds, say what's happening and count the seconds, so the player can see it's still working.
  const slow = setInterval(() => {
    const secs = Math.round((Date.now() - started) / 1000);
    if (secs < 3) return;
    if (!gpu) return renderAction(`${thinking} (the server may be waking up; this can take a few seconds)`);
    // The server knows when the GPU last answered anyone; before its first reply, fall back on this browser's.
    const asleep = step ? !step.awake && !step.answered : !awake;
    renderAction(`${step?.text || thinking} ${secs}\u00a0s`,
      asleep && "Its GPU switches off when nobody's playing, so it may need a few minutes to start up again.");
  }, 1000);
  const gen = generation;
  let next = null;
  try {
    next = await call(path, body);
  } catch (err) {
    if (err.unknownWord && path.endsWith("/clue")) {
      const name = PARTNERS[state.partner] || "Your partner";
      clearInterval(slow);
      clearInterval(poll);
      busy = false;
      if (confirm(`${name} doesn't know the word ${err.unknownWord}, so it would have to guess blind. Give it anyway?`)) {
        return act(path, { ...body, anyway: true }, thinking);
      }
      render();
      return;
    }
    next = await resync(before, err, gen);
    if (!next) $("#error").textContent = err.message;
  } finally {
    clearInterval(slow);
    clearInterval(poll);
  }
  if (next && gpu && next.history.slice(before?.history.length ?? 0).some((h) => h.side === "b")) {
    lastGpuReply = Date.now(); // only a reply with a partner move proves the GPU is awake
  }
  if (gen !== generation) return; // the player quit while we were waiting
  if (next) {
    save(next.phase === "over" ? null : next.id);
    await replayPartner(before, next, gen);
    if (gen !== generation) return;
    state = next;
  }
  busy = false;
  render(next && before !== state ? before : null);
}

// The server answers with every move the partner made at once. Show its guesses one at a time
// so the player can follow them: rebuild the board as it was after each guess.
async function replayPartner(before, next, gen) {
  const start = before ? before.history.length : 0;
  const steps = next.history.map((h, n) => n).filter((n) => n >= start && next.history[n].type === "guess" && next.history[n].side === "b");
  if (!before || !steps.length) return;
  let shown = before;
  for (const n of steps) {
    await new Promise((r) => setTimeout(r, GUESS_PACE_MS));
    if (gen !== generation) return;
    const frame = upTo(next, n);
    state = frame;
    render(shown);
    renderAction("Your partner is guessing…");
    shown = frame;
  }
  await new Promise((r) => setTimeout(r, GUESS_PACE_MS));
}

// `next` as it looked just after history entry `n`: later guesses undone, later entries hidden.
function upTo(next, n) {
  const s = structuredClone(next);
  let giver = null;
  for (let k = 0; k < next.history.length; k++) {
    const h = next.history[k];
    if (h.type === "clue") giver = h.side;
    if (k <= n || h.type !== "guess") continue;
    const judge = h.sudden_death ? (h.side === "a" ? "b" : "a") : giver;
    if (h.outcome === "G") s.found[h.index] = false;
    if (h.outcome === "N") s.bystanders[judge] = s.bystanders[judge].filter((i) => i !== h.index);
  }
  s.history = next.history.slice(0, n + 1);
  s.found_count = s.found.filter(Boolean).length;
  s.phase = "replay";
  s.result = null;
  s.partner_key = null;
  return s;
}

// ---------- rendering ----------

function render(before = null) {
  $("#setup").hidden = Boolean(state) && !setupOpen;
  $("#game").hidden = !state;
  if (!state) return;
  $("#found").textContent = `Agents found: ${state.found_count} / ${state.total_agents}`;
  const turns = $("#turns");
  const sudden = state.phase === "sudden_death" || (state.phase === "over" && state.history.some((h) => h.type === "sudden_death"));
  turns.textContent = sudden ? "Sudden death" : `Turns left: ${state.turns_left}`;
  const low = sudden || (state.phase !== "over" && state.turns_left <= LOW_TURNS);
  if (low && !turns.classList.contains("low")) turns.classList.add("pulse");
  turns.classList.toggle("low", low);
  $("#partner-name").textContent = `Partner: ${PARTNERS[state.partner] || state.partner}`;
  renderBoard(before);
  renderTips();
  renderLog();
  renderAction();
}

function cardInfo(i) {
  const s = state;
  const mine = s.bystanders.a.includes(i);
  const theirs = s.bystanders.b.includes(i);
  const myKey = s.my_key[i];
  const partnerKey = s.partner_key?.[i];
  const myTurnToGuess = (s.phase === "guess" && s.clue?.giver === "b") || (s.phase === "sudden_death" && s.guesser === "a");
  const guessable = myTurnToGuess && !s.found[i] && !theirs && !busy;
  const dead = !s.found[i] && ((mine && theirs) || (theirs && myKey !== "G"));
  const hitAssassin = s.result === "assassin" && s.history.at(-2)?.index === i;
  return { mine, theirs, myKey, partnerKey, guessable, dead, hitAssassin };
}

function renderBoard(before) {
  const board = $("#board");
  const odds = tipsOn() && state.phase !== "over" ? partnerOdds(state) : null;
  const known = odds ? knownPartnerKey(state) : {};
  board.replaceChildren(...state.words.map((word, i) => {
    const c = cardInfo(i);
    const o = odds && c.guessable && !(i in known) ? odds[c.myKey] : null;
    const label = [word,
      state.found[i] && "agent found",
      c.myKey === "G" && "your agent", c.myKey === "A" && "your assassin",
      c.mine && "bystander on your key", c.theirs && "bystander on partner's key",
      c.partnerKey && `partner's key: ${OUTCOME[c.partnerKey]}`,
      o && `odds on partner's key: ${pct(o.G)} agent, ${pct(o.N)} bystander, ${pct(o.A)} assassin`].filter(Boolean).join(", ");
    const changed = before && (before.found[i] !== state.found[i]
      || before.bystanders.a.includes(i) !== c.mine || before.bystanders.b.includes(i) !== c.theirs);
    const cls = ["card", state.found[i] && "found", c.dead && "dead", c.guessable && "guessable",
      c.hitAssassin && "assassin-hit", changed && "flash"].filter(Boolean).join(" ");
    const keyDots = el("span", { class: "key", "aria-hidden": "true" },
      c.myKey === "G" && el("i", { class: "dot agent" }), c.myKey === "A" && skull("mine"),
      c.partnerKey === "G" && el("i", { class: "dot agent partner" }), c.partnerKey === "A" && skull("partner"));
    const marks = el("span", { class: "marks", "aria-hidden": "true" },
      c.mine && pawn("mine"), c.theirs && pawn("theirs"));
    const chance = o && el("span", { class: `chance${o.A >= 0.2 ? " risky" : ""}`, "aria-hidden": "true" }, pct(o.G));
    return el("button", {
      class: cls, type: "button", "aria-label": label, disabled: !c.guessable,
      title: o ? `Partner's key: ${pct(o.G)} agent · ${pct(o.N)} bystander · ${pct(o.A)} assassin` : null,
      onclick: () => act(`/api/games/${state.id}/guess`, { index: i }),
    }, keyDots, marks, el("span", { class: "word" }, word), chance);
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
    if (h.type === "sudden_death") return el("li", {}, el("strong", {}, "Out of turns: sudden death. No more clues."));
    return el("li", {}, el("strong", {}, RESULT[h.result]));
  });
  $("#log").replaceChildren(...items);
}

function renderAction(message, note) {
  const box = $("#action");
  if (message) { box.replaceChildren(el("p", {}, message), ...(note ? [el("p", { class: "muted" }, note)] : [])); return; }
  const s = state;
  if (s.phase === "over") {
    box.replaceChildren(
      el("p", { class: "big-clue" }, RESULT[s.result]),
      el("p", { class: "muted" }, "Your partner's key is now shown on the board (dashed dots)."),
      el("div", {}, el("button", { type: "button", onclick: () => { save(null); setupOpen = true; $("#setup").hidden = false; $("#setup").scrollIntoView(); } }, "New game")));
    return;
  }
  if (s.phase === "clue" && s.giver === "a") {
    const word = el("input", { type: "text", id: "clue-word", required: true, autocomplete: "off", spellcheck: "false", maxlength: 30,
      pattern: "[A-Za-z]+", title: "One word, letters only", "aria-describedby": "clue-help" });
    word.addEventListener("input", () => { word.value = word.value.replace(/[^A-Za-z]/g, ""); });
    const num = el("input", { type: "number", id: "clue-number", min: 0, max: 9, value: 1, required: true });
    const form = el("form", {
      class: "clue-form",
      onsubmit: (e) => { e.preventDefault(); act(`/api/games/${s.id}/clue`, { word: word.value, number: Number(num.value) }); },
    }, el("label", {}, "Clue word", word), el("label", {}, "Number", num), el("button", { type: "submit" }, "Give clue"));
    box.replaceChildren(el("p", {}, "Your turn to give a clue. Point your partner at your green-dot words, and steer clear of your assassins."), form,
      el("p", { id: "clue-help", class: "muted hint" }, "One word, letters only. It can't be a word on the board or contain one."));
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
  if (s.phase === "sudden_death" && s.guesser === "a") {
    box.replaceChildren(
      el("p", { class: "big-clue" }, "Sudden death"),
      el("p", {}, "No more clues. Click a word you think is one of your partner's agents. Earlier clues still help."),
      el("p", { class: "muted" }, "A right guess finds it; any wrong guess loses the game."));
    return;
  }
  box.replaceChildren(el("p", {}, "Waiting for your partner…"));
}

// ---------- start and quit ----------

function quitToMenu() {
  const inProgress = state && state.phase !== "over";
  if (inProgress && !confirm("Leave this game and go back to the menu? You won't be able to come back to it.")) return;
  generation += 1;
  busy = false;
  save(null);
  state = null;
  setupOpen = true;
  $("#error").textContent = "";
  render();
  $("#setup").scrollIntoView({ block: "start" });
  document.querySelector('input[name="partner"]:checked').focus();
}

$("#quit").addEventListener("click", quitToMenu);

for (const i of document.querySelectorAll("[data-pawn]")) i.innerHTML = PAWN;

// Each explainer's text lives once in the page (for screen readers); copy it into its tooltips.
for (const term of document.querySelectorAll(".term")) {
  term.dataset.tip = document.getElementById(term.getAttribute("aria-describedby")).textContent;
}

// ---------- tabs (Game, How to play, Why I built it) ----------

const TABS = ["game", "how", "why"];
function showTab(name, focus = false) {
  for (const t of TABS) {
    const btn = $(`#tab-btn-${t}`);
    btn.setAttribute("aria-selected", String(t === name));
    btn.tabIndex = t === name ? 0 : -1;
    $(`#tab-${t}`).hidden = t !== name;
  }
  if (focus) $(`#tab-btn-${name}`).focus();
  if (name === "game" && state) fitWords();
  history.replaceState(null, "", name === "game" ? location.pathname + location.search : `#${name}`);
}
for (const t of TABS) $(`#tab-btn-${t}`).addEventListener("click", () => showTab(t));
$(".tabs").addEventListener("keydown", (e) => {
  const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
  if (!step) return;
  const current = TABS.findIndex((t) => $(`#tab-btn-${t}`).getAttribute("aria-selected") === "true");
  showTab(TABS[(current + step + TABS.length) % TABS.length], true);
});
const tabFromHash = () => (TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "game");
showTab(tabFromHash());
window.addEventListener("hashchange", () => showTab(tabFromHash()));

// Selecting a GPU partner starts its GPU booting while the player finishes setting up.
for (const input of document.querySelectorAll('input[name="partner"]')) {
  input.addEventListener("change", () => {
    if (input.checked && GPU_PARTNERS.includes(input.value)) call("/api/warm", { partner: input.value }).catch(() => {});
  });
}

$("#tips").checked = tipsOn();
$("#tips").addEventListener("change", (e) => {
  try { localStorage.setItem(TIPS_KEY, e.target.checked ? "1" : "0"); } catch (_) { /* storage blocked */ }
  if (state) render();
});

$("#start").addEventListener("click", () => {
  const first = document.querySelector('input[name="first"]:checked').value;
  state = null;
  setupOpen = false;
  render();
  $("#game").hidden = false;
  const partner = document.querySelector('input[name="partner"]:checked').value;
  // Show the board first, then ask for the partner's opening clue, so the player can study the
  // board while a GPU partner wakes up.
  act("/api/games", { first, partner, sudden_death: $("#sudden-death").checked, turns: Number($("#turn-count").value),
    partner_moves: false }, "Setting up the board…").then(() => {
    if (state?.phase === "clue" && state.giver === "b") {
      act(`/api/games/${state.id}/partner`, {}, "Your partner is thinking of a clue. Take a look at the board while you wait.");
    }
  });
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
