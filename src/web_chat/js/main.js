/* Entry point: wires the DOM to the modules, owns the animation loop, the
 * keyboard shortcuts and the connection indicator. The WebGL orb is loaded
 * lazily after first paint so the page is usable immediately. */

import * as api from "./api.js";
import { isOutputRouted, stopSpeaking } from "./speech.js";
import { ensureMic, sampleFrame } from "./audio.js";
import { handleClip, handleUserText, newConversation } from "./conversation.js";
import { initExport } from "./exporter.js";
import { PhaseMixer } from "./palette.js";
import {
  cancelRecording,
  maybeRearmWakeListening,
  pttRelease,
  setClipHandler,
  startRecording,
  startWakeListening,
  stopRecording,
  stopWakeListening,
} from "./recorder.js";
import { initSettings, motionReduced, resetSettings, resolvedTheme, setSetting, settings, toggleTheme } from "./settings.js";
import { on, refreshPhase, store } from "./store.js";
import * as transcript from "./transcript.js";
import { $, isTypingTarget, setIdleCaption, setStatus, toast } from "./ui.js";
import { createVisualizer } from "./visualizer.js";

const root = document.documentElement;
const btnMic = $("btn-mic");
const btnSend = $("btn-send");
const btnHandsFree = $("btn-handsfree");
const btnWakeWord = $("btn-wakeword");
const btnSpeak = $("btn-speak");
const textInput = $("text-input");
const messagesEl = $("messages");
const stageEl = $("stage");
const meshEl = document.querySelector(".mesh");

const LANG_LABEL = { auto: "Auto-detect", en: "English", hi: "Hindi", ta: "Tamil" };
const PHASE_TITLE = { idle: "Ready", listening: "Listening", thinking: "Thinking", speaking: "Speaking" };

transcript.rememberEmptyState();
initSettings();
const exporter = initExport();
setClipHandler(handleClip);

/* ---- phase labels -------------------------------------------------------- */
function idleCaption() {
  if (store.wakeArmed) return 'Say "hey chatbot" to start';
  if (store.handsFree) return "Hands-free is on";
  return "Hold the mic or press Space to talk";
}

function renderPhase() {
  const title =
    store.phase === "idle" && store.wakeArmed ? "Waiting for wake word" : PHASE_TITLE[store.phase] || "Ready";
  $("phase-title").textContent = title;
  setIdleCaption(idleCaption());
  const recording = store.recording && !store.recordingIsWakeCheck;
  btnMic.classList.toggle("recording", recording);
  btnMic.setAttribute("aria-pressed", String(recording));
  btnMic.setAttribute(
    "aria-label",
    recording ? "Stop recording" : "Talk: hold to speak, or tap to start and stop",
  );
}

on("phase", renderPhase);
on("recording", renderPhase);
on("wake-armed", renderPhase);

on("busy", () => {
  const locked = store.busy && !store.quietBusy;
  btnSend.disabled = locked;
  btnMic.disabled = locked;
  btnSend.setAttribute("aria-disabled", String(locked));
});

/* ---- connection status ---------------------------------------------------- */
function applyWhisperDevice(device) {
  if (device) setStatus("ok", "Ready · STT on " + String(device).toUpperCase());
}
on("stt-device", applyWhisperDevice);

async function checkHealth() {
  try {
    const data = await api.health();
    // Whisper loads lazily on the first transcription request in some setups,
    // so the device can still be null: show plain "Ready" rather than a
    // "STT on ?" that would look broken. It fills in once known.
    if (data.whisper_device) applyWhisperDevice(data.whisper_device);
    else setStatus("ok", "Ready");
  } catch (err) {
    setStatus("err", "Server unreachable");
  }
}
checkHealth();
setInterval(() => {
  if (!document.hidden) checkHealth();
}, 20000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) checkHealth();
});

/* ---- composer -------------------------------------------------------------- */
function autosize() {
  textInput.style.height = "auto";
  textInput.style.height = Math.min(textInput.scrollHeight, 140) + "px";
  textInput.style.overflowY = textInput.scrollHeight > 140 ? "auto" : "hidden";
}

function submitText(text) {
  if (store.busy) {
    toast("One moment, still working on the last message.");
    return;
  }
  handleUserText(text);
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = textInput.value;
  if (!text.trim()) return;
  textInput.value = "";
  autosize();
  submitText(text);
});

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
textInput.addEventListener("input", autosize);

messagesEl.addEventListener("click", (e) => {
  const suggest = e.target.closest(".suggest");
  if (suggest) submitText(suggest.textContent);
});

/* ---- push-to-talk / tap-to-talk ------------------------------------------ */
let pressAt = 0;
let pressActive = false;

// Press and release are timed with the events' own timestamps, not with when
// our handlers happen to run: a busy main thread (or a slow mic start) must
// never turn a quick tap into a "hold".
function pttPress(timeStamp) {
  if (store.recording) {
    // Already recording in tap mode: a press ends it.
    stopRecording();
    pressActive = false;
    return;
  }
  if (store.busy && !store.quietBusy) return;
  pressAt = timeStamp;
  pressActive = true;
  btnMic.classList.add("held");
  startRecording({ ptt: true });
}

function pttUp(e) {
  btnMic.classList.remove("held");
  if (!pressActive) return;
  pressActive = false;
  pttRelease((e && e.timeStamp ? e.timeStamp : performance.now()) - pressAt);
}

btnMic.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || btnMic.disabled) return;
  e.preventDefault();
  try {
    btnMic.setPointerCapture(e.pointerId);
  } catch (err) {
    /* capture is a nicety */
  }
  pttPress(e.timeStamp);
});
btnMic.addEventListener("pointerup", pttUp);
btnMic.addEventListener("pointercancel", pttUp);
btnMic.addEventListener("contextmenu", (e) => e.preventDefault());
// Keyboard / assistive-tech activation arrives as a click with detail 0 (no
// pointer involved): that is plain tap-to-talk. Pointer clicks (detail > 0)
// were already handled by the press/release pair above.
btnMic.addEventListener("click", (e) => {
  if (e.detail !== 0) return;
  if (store.recording) stopRecording();
  else if (!store.busy) startRecording();
});

/* ---- mode toggles ---------------------------------------------------------- */
function setPressed(button, value) {
  button.setAttribute("aria-pressed", String(value));
}

async function toggleHandsFree() {
  store.handsFree = !store.handsFree;
  setPressed(btnHandsFree, store.handsFree);
  renderPhase();
  toast(
    store.handsFree
      ? "Hands-free on: listening now and after every reply. Headphones help it avoid hearing its own voice."
      : "Hands-free off.",
  );
  if (store.handsFree) {
    // Start listening immediately, like tapping the mic: otherwise turning
    // this on gives no feedback until the FIRST reply ends, which reads as broken.
    if (!store.recording && !store.busy) startRecording();
  } else {
    if (store.recording && !store.recordingIsWakeCheck && !pressActive) stopRecording();
    maybeRearmWakeListening(); // no-op unless wake word is also on
  }
}

async function toggleWakeWord() {
  if (!store.wakeWord) {
    try {
      await ensureMic();
    } catch (err) {
      toast("Microphone access denied: " + err.message, true);
      return;
    }
    store.wakeWord = true;
    setPressed(btnWakeWord, true);
    toast('Wake word on: say "hey chatbot". Background speech is transcribed locally and discarded unless it matches.');
    if (!store.recording && !store.busy) startWakeListening();
  } else {
    store.wakeWord = false;
    setPressed(btnWakeWord, false);
    stopWakeListening();
    toast("Wake word off.");
  }
  renderPhase();
}

function toggleSpeak() {
  setSetting("speak", !settings.speak);
}

btnHandsFree.addEventListener("click", toggleHandsFree);
btnWakeWord.addEventListener("click", toggleWakeWord);
btnSpeak.addEventListener("click", toggleSpeak);

function renderSpeak() {
  setPressed(btnSpeak, settings.speak);
  btnSpeak.setAttribute("aria-label", settings.speak ? "Spoken replies on" : "Spoken replies off");
  btnSpeak.title = "Speak replies (V)";
}

let clearArmedUntil = 0;
function requestNewConversation() {
  if (transcript.entries.length === 0) {
    toast("This conversation is already empty.");
    return;
  }
  const now = performance.now();
  if (now > clearArmedUntil) {
    clearArmedUntil = now + 4000;
    toast("Press New again to clear this conversation. Export it first if you want to keep it.");
    return;
  }
  clearArmedUntil = 0;
  newConversation();
  toast("Started a new conversation.");
}
$("btn-new").addEventListener("click", requestNewConversation);

/* ---- header: theme, export, settings, help ---------------------------------- */
const settingsDialog = $("settings");
const shortcutsDialog = $("shortcuts");

function renderTheme() {
  const light = resolvedTheme() === "light";
  const btn = $("btn-theme");
  btn.querySelector("use").setAttribute("href", light ? "#i-moon" : "#i-sun");
  btn.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
}

$("btn-theme").addEventListener("click", toggleTheme);
$("btn-export").addEventListener("click", () => exporter.open());
$("btn-settings").addEventListener("click", () => openDialog(settingsDialog));
$("btn-help").addEventListener("click", () => openDialog(shortcutsDialog));
$("settings-export").addEventListener("click", () => {
  settingsDialog.close();
  exporter.open();
});
$("settings-reset").addEventListener("click", () => {
  resetSettings();
  toast("Settings reset to defaults.");
});

function openDialog(dialog) {
  if (!dialog.open) dialog.showModal();
}

// Click on the dimmed backdrop (a click whose target is the dialog element
// itself, outside its box) closes the dialog.
document.querySelectorAll("dialog").forEach((dialog) => {
  dialog.addEventListener("click", (e) => {
    if (e.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (!inside) dialog.close();
  });
});

function anyDialogOpen() {
  return !!document.querySelector("dialog[open]");
}

/* ---- settings side effects ---------------------------------------------------- */
function renderLanguageChip() {
  $("lang-chip").textContent = LANG_LABEL[settings.language] || settings.language;
}

on("setting", ({ key }) => {
  if (key === "language") renderLanguageChip();
  if (key === "theme") renderTheme();
  if (key === "speak") {
    renderSpeak();
    if (!settings.speak) stopSpeaking();
    toast(settings.speak ? "Spoken replies on." : "Spoken replies off. Replies stay as text.");
  }
  if (key === "showMeta") messagesEl.classList.toggle("show-meta-off", !settings.showMeta);
});
renderLanguageChip();
renderTheme();
renderSpeak();
messagesEl.classList.toggle("show-meta-off", !settings.showMeta);

/* ---- keyboard shortcuts --------------------------------------------------------- */
function isInteractive(target) {
  return !!(target && target.closest && target.closest("button, a, summary, [role='button']"));
}

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;

  if (e.key === "Escape") {
    if (anyDialogOpen()) return; // the dialog closes itself
    if (cancelRecording()) return;
    stopSpeaking(); // also aborts a reply that is still streaming
    return;
  }
  if (anyDialogOpen()) return;

  if (e.key === " " || e.code === "Space") {
    if (isTypingTarget(e.target) || isInteractive(e.target)) return;
    e.preventDefault();
    if (!e.repeat) pttPress(e.timeStamp);
    return;
  }
  if (isTypingTarget(e.target)) return;

  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  switch (key) {
    case "m":
      if (store.recording) stopRecording();
      else if (!store.busy) startRecording();
      break;
    case "h":
      toggleHandsFree();
      break;
    case "w":
      toggleWakeWord();
      break;
    case "v":
      toggleSpeak();
      break;
    case "n":
      requestNewConversation();
      break;
    case "e":
      exporter.open();
      break;
    case "s":
      openDialog(settingsDialog);
      break;
    case "t":
      toggleTheme();
      break;
    case "/":
      e.preventDefault();
      textInput.focus();
      break;
    case "?":
      openDialog(shortcutsDialog);
      break;
    default:
      return;
  }
  e.preventDefault();
});

document.addEventListener("keyup", (e) => {
  if ((e.key === " " || e.code === "Space") && pressActive) pttUp(e);
});

// Losing focus mid-hold (alt-tab, a dialog opening) must never leave the mic stuck on.
window.addEventListener("blur", () => {
  if (pressActive) pttUp();
});

/* ---- animation loop -------------------------------------------------------------- */
const mixer = new PhaseMixer();
const viz = createVisualizer($("wave"), $("spectro"));
let orb = null;

const levels = { level: 0, bass: 0, mid: 0, treble: 0 };
const tilt = { x: 0, y: 0, tx: 0, ty: 0 };
let lastNow = performance.now();
let animTime = 0;
let lastCssLevel = -1;

stageEl.addEventListener("pointermove", (e) => {
  const r = stageEl.getBoundingClientRect();
  tilt.tx = ((e.clientX - r.left) / r.width - 0.5) * 2;
  tilt.ty = ((e.clientY - r.top) / r.height - 0.5) * 2;
});
stageEl.addEventListener("pointerleave", () => {
  tilt.tx = 0;
  tilt.ty = 0;
});

function bandAverage(freq, from, to) {
  let sum = 0;
  for (let i = from; i < to; i++) sum += freq[i];
  return sum / ((to - from) * 255);
}

function smoothTo(current, target, dt) {
  const rate = target > current ? 24 : 7; // fast attack, gentle release
  return current + (target - current) * (1 - Math.exp(-dt * rate));
}

function frame(now) {
  requestAnimationFrame(frame);
  const reduced = motionReduced();
  // Reduced motion: keep the audio-reactive meters honest but update gently.
  if (reduced && now - lastNow < 90) return;
  const dt = Math.min((now - lastNow) / 1000, 0.1);
  const frameMs = now - lastNow;
  lastNow = now;
  if (!reduced) animTime += dt;

  // Where does the signal come from right now?
  let source = null;
  if (store.phase === "listening" || store.wakeArmed) source = "mic";
  else if (store.phase === "speaking" && isOutputRouted()) source = "out";
  const data = source ? sampleFrame(source) : null;

  let targetLevel = 0;
  let bass = 0;
  let mid = 0;
  let treble = 0;
  if (data) {
    targetLevel = Math.min(1, Math.pow(data.rms * 5.5, 0.65));
    if (source === "mic" && store.phase !== "listening") targetLevel *= 0.35;
    bass = bandAverage(data.freq, 1, 7);
    mid = bandAverage(data.freq, 7, 41);
    treble = bandAverage(data.freq, 41, 120);
  } else if (store.phase === "speaking") {
    // Playing without a real analyser tap (suspended context): pulse plausibly.
    const wob = Math.abs(Math.sin(animTime * 7.3)) * Math.abs(Math.sin(animTime * 2.1 + 1));
    targetLevel = 0.2 + 0.3 * wob;
    bass = 0.3 * wob;
    mid = 0.4 * wob;
  } else if (store.phase === "thinking") {
    targetLevel = 0.1 + 0.05 * Math.sin(animTime * 3);
  }
  levels.level = smoothTo(levels.level, targetLevel, dt);
  levels.bass = smoothTo(levels.bass, Math.min(1, bass * 1.4), dt);
  levels.mid = smoothTo(levels.mid, Math.min(1, mid * 1.6), dt);
  levels.treble = smoothTo(levels.treble, Math.min(1, treble * 2.4), dt);

  mixer.update(dt, store.phase);

  if (Math.abs(levels.level - lastCssLevel) > 0.012) {
    lastCssLevel = levels.level;
    const v = levels.level.toFixed(3);
    stageEl.style.setProperty("--lvl", v);
    meshEl.style.setProperty("--lvl", v);
  }

  viz.draw(data, mixer, animTime, levels.level, dt);

  if (orb) {
    const k = 1 - Math.exp(-dt * 6);
    tilt.x += (tilt.tx * 0.55 - tilt.x) * k;
    tilt.y += (tilt.ty * 0.55 - tilt.y) * k;
    orb.render({
      time: animTime,
      level: levels.level,
      bands: [levels.bass, levels.mid, levels.treble],
      phases: mixer.weights,
      colors: mixer.colors,
      tilt: [tilt.x, tilt.y],
      halo: root.getAttribute("data-theme") === "light" ? 0.6 : 1,
      frameMs,
    });
  }
}

new ResizeObserver(() => {
  viz.resize();
  if (orb) orb.resize();
}).observe(stageEl);
viz.resize();
requestAnimationFrame(frame);

/* ---- lazy WebGL orb ----------------------------------------------------------------- */
async function loadOrb() {
  try {
    const mod = await import("./orb.js");
    const canvas = $("orb");
    const created = mod.createOrb(canvas);
    if (!created) return; // no WebGL: the CSS orb stays
    orb = created;
    orb.resize();
    canvas.classList.add("ready");
    $("orb-fallback").classList.add("hidden");
  } catch (err) {
    console.warn("[orb] could not load; using the CSS orb", err);
  }
}
if ("requestIdleCallback" in window) requestIdleCallback(loadOrb, { timeout: 1500 });
else setTimeout(loadOrb, 250);

renderPhase();
refreshPhase();

// Read-only view of the state machine, handy for debugging in the console.
window.__voiceChat = Object.freeze({
  state: () => ({ ...store }),
  hasOrb: () => !!orb,
});
