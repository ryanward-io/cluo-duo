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
// Each row reads as label, note, value; the bar itself is decoration, and every chart has a table too.
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
const rowsOf = (rows) => rows.filter((r) => r !== "sep").map((r) => [r.note ? `${r.label} (${r.note})` : r.label, r.text]);

// Partners in the order I built them (selected milestones), each against the same reference player.
const PROGRESS = [
  { label: "Random", note: "a floor", value: 2.8, text: "2.8" },
  { label: "GloVe word vectors", note: "closest words to a clue; 4 boards", value: 6.0, text: "6.0" },
  { label: "Basic", note: "GloVe + Bayesian guessing", value: 9.0, text: "9.0" },
  { label: "Qwen 3.6 35B-A3B gives clues", note: "GloVe checks them and guesses", value: 8.6, text: "8.6" },
  { label: "Gemma 4 31B gives clues", note: "same setup, better model", value: 10.4, text: "10.4" },
  { label: "Gemma guesses too", value: 11.0, text: "11.0" },
  { label: "Gemma alone, 28 Oct", note: "no GloVe; tests every clue idea", value: 13.0, text: "13.0" },
  { label: "Advanced (live)", note: "the same, with the speed fix", value: 11.2, text: "11.2", hi: true },
  "sep",
  { label: "Claude Sonnet as partner", note: "paid API; for comparison", value: 11.2, text: "11.2" },
];

const MODELS = [
  { label: "Mistral Small 3.2", note: "24B · Mistral AI", value: 8.5, text: "8.5" },
  { label: "Qwen 3.6 35B-A3B", note: "35B, 3B active · Alibaba", value: 8.6, text: "8.6" },
  { label: "Qwen 3.6 27B", note: "27B · Alibaba", value: 9.4, text: "9.4" },
  { label: "gpt-oss-120b", note: "120B, ~5B active · OpenAI", value: 10.4, text: "10.4" },
  { label: "Gemma 4 31B", note: "31B · Google", value: 10.4, text: "10.4", hi: true },
];

// Games played while building and testing. Reference and two-agent games are counted from
// data/agent-games (boards 200-207: 208 reference incl. the 10 Nov rerun, 4 Opus self-play). Self-play counts are rebuilt
// from session logs: evaluate.py and compare.py print their results and never saved them.
const SELF_PLAY_GLOVE = "about 17,000";
const GAMES = [
  { label: "Self-play, LLM partners", note: "20 games per version, on the GPU", value: 400, text: "about 400" },
  { label: "Two AI players", note: "refining the player guide", value: 172, text: "172" },
  { label: "Reference player + partner", note: "the games behind the results", value: 208, text: "208", hi: true },
];

// US$ per game. GPU: Modal's bill (see the caption). Claude: rough estimates from token use, not bills.
const COST = [
  { label: "Gemma 4 on a rented GPU", note: "Modal L40S, billed while awake", value: [0.27, 0.75], text: "$0.27–0.75", hi: true },
  { label: "Claude Sonnet, medium effort", note: "estimate", value: [0.15, 0.25], text: "$0.15–0.25" },
  { label: "Claude Opus, medium effort", note: "estimate", value: [0.3, 0.5], text: "$0.30–0.50" },
  { label: "Claude Opus, max effort", note: "estimate; ~2 min per clue", value: [1.0, 1.5], text: "$1.00–1.50" },
  { label: "DeepSeek V4.1 Flash", note: "Modal's quoted price; not tested", value: 0.01, text: "~$0.01" },
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
    `Gemma sees its own key: 9 agents (green dots) and 3 assassins (small squares). It brainstorms ${ideas.length} clue ideas, each aimed at one to three of its agents. Pick an idea to see its targets outlined on the board.`,
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
      el("div", {}, board, el("p", { class: "mini-legend" }, "Outlined: the idea's targets. Underlined: its test guesses.")),
      el("div", {}, el("ul", { class: "ideas", "aria-label": "Clue ideas" }, ideaItems.map((x) => el("li", {}, x.button))), reason)));
}

// ---------- speed: one dot per clue turn ----------
// Dots that would overlap are stacked up and down from the row's centre line (a small beeswarm).
function swarm(xs, gap) {
  const placed = [];
  return xs.map((x) => {
    for (let k = 0; ; k++) {
      const dy = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * gap;
      if (placed.every((p) => Math.hypot(p.x - x, p.dy - dy) >= gap)) { placed.push({ x, dy }); return dy; }
    }
  });
}

function strip(host, data) {
  const rows = [
    { label: "Before", sub: `${data.before.length} turns`, values: data.before },
    { label: "After", sub: `${data.after.length} turns`, values: data.after },
  ];
  const draw = () => {
    const W = Math.max(300, host.clientWidth), left = 74, right = 16, top = 26, rowH = 76, H = top + rows.length * rowH + 26;
    const max = 40, x = (s) => left + (s / max) * (W - left - right);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Time to test the clue ideas, per clue turn. Before the fix: ${data.before.length} turns, half near 7 seconds and half near 33. After: ${data.after.length} turns, all 5 to 8 seconds.` });
    for (let t = 0; t <= max; t += 10) {
      g.append(svg("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 4, y2: H - 22 }),
        svg("text", { class: "tick", x: x(t), y: H - 6, "text-anchor": "middle" }, `${t} s`));
    }
    g.append(svg("text", { class: "note", x: x(7), y: 12, "text-anchor": "middle" }, "normal"),
      svg("text", { class: "note", x: Math.min(x(33.5), W - right), y: 12, "text-anchor": "end" }, "one reply ran away"));
    rows.forEach((r, ri) => {
      const cy = top + ri * rowH + rowH / 2;
      g.append(svg("text", { class: "rowlabel", x: 0, y: cy - 2 }, r.label), svg("text", { class: "tick", x: 0, y: cy + 13 }, r.sub));
      const xs = r.values.map(x);
      const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
      const dys = swarm(order.map((i) => xs[i]), 11);
      order.forEach((i, k) => g.append(svg("circle", { class: `dot${r.values[i] > 20 ? " slow" : ""}`, cx: xs[i], cy: cy + dys[k], r: 5 })));
    });
    host.replaceChildren(g);
  };
  draw();
  new ResizeObserver(draw).observe(host);
}

function table(host, head, rows) {
  host.after(el("details", {}, el("summary", {}, "Show as a table"),
    el("table", {}, el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: i ? "num" : "" }, h)))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i ? "num" : "" }, c))))))));
}

// ---------- wire up ----------
const $ = (s) => document.querySelector(s);
if ($("#fig-progress")) {
  const fifteen = [0, 5, 10, 15].map((v) => [v, String(v)]);
  bars($("#fig-progress .chart"), PROGRESS, { max: 15, ticks: fifteen });
  table($("#fig-progress .chart"), ["Partner, in build order", "Agents found (of 15)"], rowsOf(PROGRESS));
  bars($("#fig-models .chart"), MODELS, { max: 15, ticks: fifteen });
  table($("#fig-models .chart"), ["Model giving the clues", "Agents found (of 15)"], rowsOf(MODELS));
  $("#fig-games .stat-glove").textContent = SELF_PLAY_GLOVE;
  bars($("#fig-games .chart"), GAMES, { max: 500, ticks: [0, 100, 200, 300, 400, 500].map((v) => [v, String(v)]) });
  table($("#fig-games .chart"), ["Kind of game", "Games"], [["Self-play, GloVe partners (on a CPU)", SELF_PLAY_GLOVE], ...rowsOf(GAMES)]);
  bars($("#fig-cost .chart"), COST, { max: 2, ticks: [[0, "$0"], [0.5, "$0.50"], [1, "$1"], [1.5, "$1.50"], [2, "$2"]] });
  table($("#fig-cost .chart"), ["Partner", "US$ per game"], rowsOf(COST));
  try {
    const res = await fetch(new URL(`why-data.json${new URL(import.meta.url).search}`, import.meta.url));
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    clueWalk($("#fig-clue .chart"), data.clue);
    strip($("#fig-speed .chart"), data.speed);
    table($("#fig-speed .chart"), ["Clue turn", "Seconds to test its ideas"],
      [...data.speed.before.map((s, i) => [`Before, turn ${i + 1}`, s.toFixed(1)]), ...data.speed.after.map((s, i) => [`After, turn ${i + 1}`, s.toFixed(1)])]);
  } catch {
    for (const id of ["#fig-clue", "#fig-speed"]) $(`${id} .chart`).replaceChildren(el("p", { class: "fig-error" }, "This figure's data didn't load. Reload the page to try again."));
  }
}
