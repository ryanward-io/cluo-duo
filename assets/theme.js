// Light/dark toggle, sharing ryanward.dev's saved choice (localStorage "theme"), so picking a theme
// on the blog or here applies to both. Loaded in <head> without defer, so a saved choice applies
// before first paint; with no saved choice the page follows the system setting.
(function () {
  var root = document.documentElement;
  try {
    var t = localStorage.getItem("theme");
    if (t === "dark" || t === "light") root.setAttribute("data-theme", t);
  } catch (e) { /* storage blocked */ }
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    function current() { return root.getAttribute("data-theme") || (mq.matches ? "dark" : "light"); }
    function label() { btn.setAttribute("aria-label", "Switch to " + (current() === "dark" ? "light" : "dark") + " theme"); }
    btn.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
      label();
    });
    mq.addEventListener("change", label);
    label();
  });
})();
