// Figures for the "Why I built it" tab. Data comes from saved test runs (why-data.json) and the
// results table in the README; nothing here calls the game server.

const el = (tag, attrs = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : String(c));
  return n;
};
const svg = (tag, attrs = {}, ...children) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : String(c));
  return n;
};
const list = (words) => (words.length > 1 ? `${words.slice(0, -1).join(", ")} and ${words.at(-1)}` : words[0] ?? "");

// ---------- bar rows: progress, models, games, cost ----------
// rows: {label, note?, value | [lo, hi], text, hi?} or "sep". max: the axis end. ticks: [value, label] pairs.
// Each row reads as label, note, value; the bar itself is decoration.
function bars(host, rows, { max, ticks }) {
  const grid = el("div", { class: "bars" });
  for (const r of rows) {
    if (r === "sep") { grid.append(el("div", { class: "sep" })); continue; }
    const [lo, hi] = Array.isArray(r.value) ? r.value : [0, r.value];
    const bar = el("div", { class: `bar${lo ? " range" : ""}`, style: `left:${(lo / max) * 100}%;width:${((hi - lo) / max) * 100}%` });
    grid.append(el("div", { class: `row${r.hi ? " hi" : ""}` },
      el("div", { class: "label" }, r.label, r.note && el("small", {}, r.note)),
      el("div", { class: "track", "aria-hidden": "true" }, bar),
      el("div", { class: "value" }, r.text)));
  }
  grid.append(el("div", { class: "axis-pad" }),
    el("div", { class: "axis", "aria-hidden": "true" }, ticks.map(([v, t]) => el("span", { style: `left:${(v / max) * 100}%` }, t))),
    el("div", {}));
  host.replaceChildren(grid);
}

// Partners in the order I built them (selected milestones), each against the same reference player.
const PROGRESS = [
  { label: "Random", note: "a floor", value: 2.8, text: "2.8" },
  { label: "GloVe word vectors", note: "closest words to a clue, 4 boards only", value: 6.0, text: "6.0" },
  { label: "Basic", note: "GloVe + Bayesian guessing", value: 9.0, text: "9.0" },
  { label: "Qwen 3.6 35B-A3B gives clues", note: "GloVe checks them and guesses", value: 8.6, text: "8.6" },
  { label: "Gemma 4 31B gives clues", note: "same setup, better model", value: 10.4, text: "10.4" },
  { label: "Gemma guesses too", value: 11.0, text: "11.0" },
  { label: "Gemma alone (live Advanced)", note: "no GloVe, tests every clue idea", value: 11.2, text: "11.2", hi: true },
  "sep",
  { label: "Claude Sonnet as partner", note: "paid API, for comparison", value: 11.2, text: "11.2" },
];

const MODELS = [
  { label: "Mistral Small 3.2", note: "24B · Mistral AI", value: 8.5, text: "8.5" },
  { label: "Qwen 3.6 35B-A3B", note: "35B, 3B active · Alibaba", value: 8.6, text: "8.6" },
  { label: "Qwen 3.6 27B", note: "27B · Alibaba", value: 9.4, text: "9.4" },
  { label: "gpt-oss-120b", note: "120B, ~5B active · OpenAI", value: 10.4, text: "10.4" },
  { label: "Gemma 4 31B", note: "31B · Google · used live", value: 10.4, text: "10.4", hi: true },
];

// The games figure is plain HTML in index.html. Counts: reference games and partner versions from data/agent-games
// (boards 200-207); self-play totals rebuilt from session logs, since evaluate.py and compare.py never saved results.

// US$ per game. GPU: Modal's bill (see the caption). Claude: rough estimates from token use, not bills.
const COST = [
  { label: "Gemma 4 on a rented GPU", note: "Modal L40S, billed while awake", value: [0.27, 0.58], text: "$0.27–0.58", hi: true },
  { label: "Claude Sonnet, medium effort", note: "estimate", value: [0.15, 0.25], text: "$0.15–0.25" },
  { label: "Claude Opus, medium effort", note: "estimate", value: [0.3, 0.5], text: "$0.30–0.50" },
  { label: "Claude Opus, max effort", note: "estimate, ~2 min per clue", value: [1.0, 1.5], text: "$1.00–1.50" },
];

// ---------- one clue, step by step ----------
const OUTCOME = {
  ok: (i) => `its test found ${i.run === 1 ? "its 1 agent" : i.run === 2 ? "both agents" : `all ${i.run} agents`}`,
  bystander: (i) => `${i.run} right, then a bystander`,
  assassin: () => "led to an assassin: ruled out",
  unreadable: (i) => `the test reply hit the ${i.tokens}-token limit (${Math.round(i.seconds)} s) before naming any guesses, so it scored 0`,
};

// Some replies were cut off mid-JSON; show the reasoning text, not the brackets.
const reasonText = (r) => {
  const m = r.match(/^\s*\{\s*"reason"\s*:\s*"(.*?)"?\s*\}?\s*$/s);
  return (m ? m[1].replace(/\\"/g, '"') : r).replace(/\s+/g, " ").trim();
};

function clueWalk(host, data) {
  const { words, key, ideas } = data;
  const chosen = ideas.reduce((a, b) => (b.run > a.run ? b : a));
  let step = 0;
  let pick = chosen;
  const NOTES = [
    `Gemma sees its own key: 9 agents (green dots) and 3 assassins (skulls). It brainstorms ${ideas.length} clue ideas, each aimed at one to three of its agents. Pick an idea to see its targets outlined on the board.`,
    "For each idea, Gemma plays your side: it's given the board without the key and asked to guess, much as it guesses your clues. All the tests run at once. Underlined words are what it guessed.",
    `It picks the idea whose test found the most agents in a row, and the clue's number is that count. Only ${chosen.clue} got both its targets; MAMMAL was thrown out for leading straight to an assassin.`,
  ];
  // Buttons are built once and updated in place, so keyboard focus stays where it was.
  const stepButtons = ["1. Brainstorm", "2. Test each idea", "3. Choose"].map((s, i) =>
    el("button", { type: "button", onclick: () => { step = i; render(); } }, s));
  const note = el("p", { class: "walk-note", "aria-live": "polite" });
  const board = el("div", { class: "mini-board", "aria-hidden": "true" },
    words.map((w, i) => el("span", { class: key[i], title: w }, w)));
  const ideaItems = ideas.map((idea) => {
    const detail = el("span", { class: "detail" });
    const mark = el("span", { class: "chosen" });
    const button = el("button", { type: "button", onclick: () => { pick = idea; render(); } },
      el("span", { class: "clueword" }, idea.clue), mark, detail);
    return { idea, button, detail, mark };
  });
  const reason = el("p", { class: "walk-reason" });

  function render() {
    stepButtons.forEach((b, i) => (i === step ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
    note.textContent = NOTES[step];
    const targets = new Set(pick.targets);
    const guessed = new Set(step >= 1 ? pick.guesses ?? [] : []);
    [...board.children].forEach((n, i) => {
      n.classList.toggle("target", targets.has(words[i]));
      n.classList.toggle("guessed", guessed.has(words[i]));
    });
    for (const { idea, button, detail, mark } of ideaItems) {
      button.setAttribute("aria-pressed", String(idea === pick));
      button.classList.toggle("out", step === 2 && idea !== chosen && idea !== pick);
      mark.textContent = step === 2 && idea === chosen ? `  ✓ chosen: ${idea.clue} ${idea.run}` : "";
      detail.textContent = step === 0 ? `for ${list(idea.targets)}`
        : `for ${list(idea.targets)}: ${idea.guesses ? `guessed ${list(idea.guesses)}, ` : ""}${OUTCOME[idea.outcome](idea)}`;
    }
    const r = step >= 1 && pick.reason ? reasonText(pick.reason) : "";
    reason.textContent = r ? `Its test reasoning: “${r.slice(0, 240)}${r.length > 240 ? "…" : ""}”` : "";
  }
  render();
  host.replaceChildren(el("div", { class: "steps", role: "group", "aria-label": "Steps" }, stepButtons), note,
    el("div", { class: "clue-walk" },
      el("div", {}, board, el("p", { class: "mini-legend", "aria-hidden": "true" }, "Outlined: the idea's targets. Underlined: its test guesses.")),
      el("div", {}, el("ul", { class: "ideas", "aria-label": "Clue ideas" }, ideaItems.map((x) => el("li", {}, x.button))), reason)));
}

// ---------- wire up ----------
const $ = (s) => document.querySelector(s);
if ($("#fig-progress")) {
  const fifteen = [0, 5, 10, 15].map((v) => [v, String(v)]);
  bars($("#fig-progress .chart"), PROGRESS, { max: 15, ticks: fifteen });
  bars($("#fig-models .chart"), MODELS, { max: 15, ticks: fifteen });
  bars($("#fig-cost .chart"), COST, { max: 2, ticks: [[0, "$0"], [0.5, "$0.50"], [1, "$1"], [1.5, "$1.50"], [2, "$2"]] });
  try {
    const res = await fetch(new URL(`why-data.json${new URL(import.meta.url).search}`, import.meta.url));
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    clueWalk($("#fig-clue .chart"), data.clue);
  } catch {
    $("#fig-clue .chart").replaceChildren(el("p", { class: "fig-error" }, "This figure's data didn't load. Reload the page to try again."));
  }
}
