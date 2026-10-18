/* Runs synchronously in <head> (before first paint) so the saved theme and
 * motion preference apply without a flash. Kept tiny and dependency-free;
 * the full settings store lives in settings.js. */
(function () {
  var root = document.documentElement;
  var saved = {};
  try {
    saved = JSON.parse(localStorage.getItem("vc_settings_v1") || "{}") || {};
  } catch (err) {
    saved = {};
  }
  var theme = saved.theme || "system";
  if (theme === "system") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  root.setAttribute("data-theme", theme);
  if (saved.motion === "reduced" || saved.motion === "full") {
    root.setAttribute("data-motion", saved.motion);
  }
})();
