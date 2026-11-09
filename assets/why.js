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

// ---------- tooltip shared by the charts ----------
const tip = el("div", { class: "chart-tip", role: "status", hidden: true });
document.body.append(tip);
function showTip(e, text) {
  tip.textContent = text;
  tip.hidden = false;
  const r = (e.target.getBoundingClientRect?.() ?? { left: e.clientX, top: e.clientY, width: 0 });
  tip.style.left = `${Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, r.left + r.width / 2 - tip.offsetWidth / 2))}px`;
  tip.style.top = `${Math.max(8, r.top - tip.offsetHeight - 6)}px`;
}
const hideTip = () => { tip.hidden = true; };

// ---------- bar rows: progress, models, cost ----------
// rows: {label, note?, value | [lo, hi], text, hi?} or "sep". max: the axis end. ticks: axis labels.
function bars(host, rows, { max, ticks, unit = "" }) {
  const grid = el("div", { class: "bars" });
  for (const r of rows) {
    if (r === "sep") { grid.append(el("div", { class: "sep" })); continue; }
    const [lo, hi] = Array.isArray(r.value) ? r.value : [0, r.value];
    const bar = el("div", { class: `bar${lo ? " range" : ""}`, style: `left:${(lo / max) * 100}%;width:${((hi - lo) / max) * 100}%` });
    const row = el("div", { class: `row${r.hi ? " hi" : ""}`, tabindex: "0", "aria-label": `${r.label}: ${r.text}`,
      onmouseenter: (e) => showTip(e, `${r.label}: ${r.text}`), onmouseleave: hideTip,
      onfocus: (e) => showTip(e, `${r.label}: ${r.text}`), onblur: hideTip },
      el("div", { class: "label" }, r.label, r.note && el("small", {}, r.note)),
      el("div", { class: "track" }, bar),
      el("div", { class: "value" }, r.text));
    grid.append(row);
  }
  grid.append(el("div", { class: "axis-pad" }), el("div", { class: "axis", "aria-hidden": "true" }, ticks.map((t) => el("span", {}, `${t}${unit}`))), el("div", {}));
  host.replaceChildren(grid);
}

const PROGRESS = [
  { label: "Random", note: "a floor", value: 2.8, text: "2.8" },
  { label: "GloVe word vectors", note: "closest words to a clue", value: 6.0, text: "6.0" },
  { label: "Basic", note: "GloVe + Bayesian guessing", value: 9.0, text: "9.0" },
  { label: "Qwen 3.6 gives clues", note: "GloVe checks them and guesses", value: 8.6, text: "8.6" },
  { label: "Gemma 4 gives clues", note: "same setup, better model", value: 10.4, text: "10.4" },
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
  { label: "gpt-oss-120b", note: "120B · OpenAI", value: 10.4, text: "10.4" },
  { label: "Gemma 4 31B", note: "31B · Google", value: 10.4, text: "10.4", hi: true },
];

// Games played while building and testing, by kind. Reference and two-agent games are counted from
// data/agent-games; self-play counts are rebuilt from session logs (evaluate.py prints, never saves).
const GAMES = [
  { label: "Self-play, GloVe partners", note: "tuning Basic's settings", value: 17260, text: "≈17,260" },
  { label: "Self-play, LLM partners", note: "20 games per version", value: 400, text: "≈400" },
  { label: "Two AI players", note: "refining the player guide", value: 170, text: "170" },
  { label: "Reference player + partner", note: "the yardstick for the results", value: 162, text: "162", hi: true },
];

// Rough per-game estimates (US$), not measured bills: see the caption.
const COST = [
  { label: "Gemma 4 on my own GPU", note: "L40S at ~$1.95/h, awake ~15–20 min", value: [0.4, 0.7], text: "$0.40–0.70", hi: true },
  { label: "Claude Sonnet, medium effort", value: [0.15, 0.25], text: "$0.15–0.25" },
  { label: "Claude Opus, medium effort", value: [0.3, 0.5], text: "$0.30–0.50" },
  { label: "Claude Opus, max effort", note: "~5 min per clue", value: [1.5, 2.5], text: "$1.50–2.50" },
  { label: "DeepSeek V4.1 Flash", note: "not tested as a partner", value: [0.02, 0.05], text: "$0.02–0.05" },
];

// ---------- one clue, step by step ----------
const OUTCOME = {
  ok: (i) => `its test found ${i.run === 1 ? "1 agent" : `all ${i.run}`}`,
  bystander: (i) => `${i.run} right, then a bystander`,
  assassin: () => "led to an assassin: ruled out",
  unreadable: (i) => `the test reply ran to ${i.tokens} tokens of blank space (${Math.round(i.seconds)} s) and couldn't be read`,
};

function clueWalk(host, data) {
  const { words, key, ideas } = data;
  const chosen = ideas.reduce((a, b) => (b.run > a.run ? b : a));
  let step = 0;
  let pick = chosen;
  const steps = ["1. Brainstorm", "2. Test each idea", "3. Choose"];
  const stepBar = el("div", { class: "steps", role: "group", "aria-label": "Steps" });
  const board = el("div", { class: "mini-board", "aria-hidden": "true" });
  const ideaList = el("ul", { class: "ideas" });
  const note = el("p", { class: "walk-note" });
  const reason = el("p", { class: "walk-reason" });

  function render() {
    stepBar.replaceChildren(...steps.map((s, i) => el("button", { type: "button", "aria-pressed": String(i === step), onclick: () => { step = i; render(); } }, s)));
    const targets = new Set(pick.targets);
    const guessed = new Set(step >= 1 ? pick.guesses ?? [] : []);
    board.replaceChildren(...words.map((w, i) => el("span", {
      class: [key[i], targets.has(w) && "target", guessed.has(w) && "guessed"].filter(Boolean).join(" "), title: w }, w)));
    ideaList.replaceChildren(...ideas.map((idea) => {
      const detail = step === 0 ? `for ${list(idea.targets)}`
        : `for ${list(idea.targets)}: ${idea.guesses ? `guessed ${list(idea.guesses)}, ` : ""}${OUTCOME[idea.outcome](idea)}`;
      const isChosen = step === 2 && idea === chosen;
      return el("li", {}, el("button", { type: "button", "aria-pressed": String(idea === pick), class: step === 2 && !isChosen ? "out" : "",
        onclick: () => { pick = idea; render(); } },
        el("span", { class: "clueword" }, idea.clue), isChosen && el("span", { class: "chosen" }, `  ✓ chosen: ${idea.clue} ${idea.run}`),
        el("span", { class: "detail" }, detail)));
    }));
    note.textContent = [
      `Gemma sees its own key: 9 agents (green dots) and 3 assassins (small squares). It brainstorms ${ideas.length} clue ideas, each aimed at one to three of its agents. Pick one to see its targets outlined.`,
      "For each idea, Gemma plays your side: it's given the board without the key and asked to guess, exactly as it would guess your clues. All the tests run at once. Underlined words are what it guessed.",
      `It picks the idea whose test found the most agents in a row, and the clue's number is that count. Only ${chosen.clue} got both its targets; MAMMAL was thrown out for leading straight to an assassin.`,
    ][step];
    reason.textContent = step >= 1 && pick.reason ? `Its test reasoning: “${pick.reason.replace(/\s+/g, " ").slice(0, 240)}${pick.reason.length > 240 ? "…" : ""}”` : "";
  }
  render();
  host.replaceChildren(stepBar, el("div", { class: "clue-walk" },
    el("div", {}, board, el("p", { class: "mini-legend" }, "Outlined: the idea's targets. Underlined: its test guesses.")),
    el("div", {}, ideaList, note, reason)));
}

// ---------- speed: one dot per clue turn ----------
function strip(host, data) {
  const rows = [
    { label: "Before the fix", values: data.before },
    { label: "After the fix", values: data.after },
  ];
  const draw = () => {
    const W = Math.max(300, host.clientWidth), left = 104, right = 14, rowH = 56, top = 10, H = top + rows.length * rowH + 26;
    const max = 40, x = (s) => left + (s / max) * (W - left - right);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Time to test the clue ideas, per clue turn. Before the fix: ${data.before.length} turns, half near 7 seconds and half near 33. After: ${data.after.length} turns, all 5 to 8 seconds.` });
    for (let t = 0; t <= max; t += 10) {
      g.append(svg("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: H - 22 }),
        svg("text", { class: "tick", x: x(t), y: H - 6, "text-anchor": "middle" }, `${t} s`));
    }
    rows.forEach((r, ri) => {
      const cy = top + ri * rowH + rowH / 2;
      g.append(svg("text", { class: "rowlabel", x: 0, y: cy + 4 }, r.label));
      r.values.forEach((v, i) => {
        const dy = ((i * 7) % 5 - 2) * 5;  // fixed jitter so near-equal times don't hide each other
        const d = svg("circle", { class: `dot${v > 20 ? " slow" : ""}`, cx: x(v), cy: cy + dy, r: 5, tabindex: "0" });
        const label = `${v.toFixed(1)} s to test the ideas${v > 20 ? ": one reply ran to the 600-token limit" : ""}`;
        d.addEventListener("mouseenter", (e) => showTip(e, label));
        d.addEventListener("focus", (e) => showTip(e, label));
        d.addEventListener("mouseleave", hideTip);
        d.addEventListener("blur", hideTip);
        g.append(d);
      });
    });
    g.append(svg("text", { class: "note", x: x(33.5), y: top + 8, "text-anchor": "middle" }, "one reply ran away"),
      svg("text", { class: "note", x: x(7.5), y: top + 8, "text-anchor": "middle" }, "normal"));
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
  bars($("#fig-progress .chart"), PROGRESS, { max: 15, ticks: [0, 5, 10, 15] });
  table($("#fig-progress .chart"), ["Partner", "Agents found (of 15)"], PROGRESS.filter((r) => r !== "sep").map((r) => [r.label, r.text]));
  bars($("#fig-models .chart"), MODELS, { max: 15, ticks: [0, 5, 10, 15] });
  bars($("#fig-games .chart"), GAMES, { max: 20000, ticks: ["0", "5,000", "10,000", "15,000", "20,000"] });
  table($("#fig-games .chart"), ["Kind of game", "Games"], GAMES.map((r) => [r.label, r.text]));
  bars($("#fig-cost .chart"), COST, { max: 3, ticks: ["$0", "$1", "$2", "$3"] });
  const data = await fetch(new URL(`why-data.json${new URL(import.meta.url).search}`, import.meta.url)).then((r) => r.json());
  clueWalk($("#fig-clue .chart"), data.clue);
  strip($("#fig-speed .chart"), data.speed);
  table($("#fig-speed .chart"), ["Clue turn", "Seconds to test its ideas"],
    [...data.speed.before.map((s, i) => [`Before, turn ${i + 1}`, s.toFixed(1)]), ...data.speed.after.map((s, i) => [`After, turn ${i + 1}`, s.toFixed(1)])]);
}
