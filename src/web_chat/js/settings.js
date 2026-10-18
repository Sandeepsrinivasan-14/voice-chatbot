/* Persistent user settings (localStorage, always wrapped: storage can throw
 * in private windows or when blocked) plus the settings drawer bindings. */

import { emit, on } from "./store.js";

const KEY = "vc_settings_v1";

export const DEFAULTS = Object.freeze({
  language: "auto", // auto | en | hi | ta -- STT language and reply/voice language
  speak: true, // read replies aloud
  rate: 1, // playback speed
  volume: 1,
  sensitivity: 3, // 1 (needs louder speech) .. 5 (picks up quiet speech)
  pauseMs: 800, // trailing silence before auto-stop
  theme: "system", // system | dark | light
  motion: "auto", // auto | reduced | full
  showMeta: true, // confidence under voice messages
});

function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "{}");
    const clean = {};
    for (const [k, v] of Object.entries(raw || {})) {
      if (k in DEFAULTS && typeof v === typeof DEFAULTS[k]) clean[k] = v;
    }
    return clean;
  } catch (err) {
    return {};
  }
}

export const settings = { ...DEFAULTS, ...load() };

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch (err) {
    /* storage unavailable: settings simply last for this tab */
  }
}

export function setSetting(key, value) {
  if (!(key in DEFAULTS) || settings[key] === value) return;
  settings[key] = value;
  persist();
  applySideEffects(key);
  emit("setting", { key, value });
}

export function resetSettings() {
  for (const [k, v] of Object.entries(DEFAULTS)) {
    if (settings[k] !== v) setSetting(k, v);
  }
}

/* Mic sensitivity -> how far above the learned noise floor speech must sit. */
export function speechMargin() {
  return { 1: 5.0, 2: 4.0, 3: 3.0, 4: 2.4, 5: 1.9 }[settings.sensitivity] || 3.0;
}

/* ---- theme / motion ---------------------------------------------------- */
const lightQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
const motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

export function resolvedTheme() {
  if (settings.theme === "system") return lightQuery && lightQuery.matches ? "light" : "dark";
  return settings.theme;
}

export function motionReduced() {
  if (settings.motion === "reduced") return true;
  if (settings.motion === "full") return false;
  return !!(motionQuery && motionQuery.matches);
}

function applySideEffects(key) {
  const root = document.documentElement;
  if (key === "theme") {
    root.setAttribute("data-theme", resolvedTheme());
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", resolvedTheme() === "light" ? "#eceffd" : "#07081a");
  }
  if (key === "motion") {
    if (settings.motion === "auto") root.removeAttribute("data-motion");
    else root.setAttribute("data-motion", settings.motion);
  }
}

if (lightQuery) {
  lightQuery.addEventListener("change", () => {
    if (settings.theme === "system") applySideEffects("theme");
  });
}

export function toggleTheme() {
  setSetting("theme", resolvedTheme() === "light" ? "dark" : "light");
}

/* ---- drawer bindings ---------------------------------------------------- */
const FORMATTERS = {
  rate: (v) => Number(v).toFixed(2).replace(/0$/, "") + "x",
  volume: (v) => Math.round(v * 100) + "%",
  sensitivity: (v) => ["", "Low", "Fairly low", "Balanced", "High", "Very high"][v],
  pauseMs: (v) => (v / 1000).toFixed(1) + "s",
};

function syncInputs() {
  document.querySelectorAll("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (input.type === "radio") input.checked = String(settings[key]) === input.value;
    else if (input.type === "checkbox") input.checked = !!settings[key];
    else input.value = settings[key];
  });
  document.querySelectorAll("[data-output-for]").forEach((out) => {
    const key = out.dataset.outputFor;
    out.textContent = FORMATTERS[key] ? FORMATTERS[key](settings[key]) : settings[key];
  });
}

export function initSettings() {
  applySideEffects("theme");
  applySideEffects("motion");
  syncInputs();

  document.querySelectorAll("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    input.addEventListener("input", () => {
      let value;
      if (input.type === "radio") {
        if (!input.checked) return;
        value = input.value;
      } else if (input.type === "checkbox") {
        value = input.checked;
      } else {
        value = Number(input.value);
      }
      setSetting(key, value);
      syncInputs();
    });
  });

  // Keep the drawer honest when a setting changes elsewhere (shortcuts etc.).
  on("setting", syncInputs);
}
