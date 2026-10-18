/* Microphone capture: push-to-talk, tap-to-talk with silence auto-stop (VAD),
 * and idle wake-word listening. Produces a recorded Blob and hands it to the
 * conversation layer via setClipHandler(); it never talks to the network. */

import { ensureMic, ensureMicAnalyser, getStream, playChime, readMicRms } from "./audio.js";
import { settings, speechMargin } from "./settings.js";
import { emit, refreshPhase, store } from "./store.js";
import { setLive, toast } from "./ui.js";
import { stopSpeaking } from "./speech.js";

// ---- silence-detection auto-stop (VAD) -----------------------------------
// Deliberately a simple RMS-energy threshold, not a ported webrtcvad/WASM
// model -- the same "boring, tunable code over an ML dependency" philosophy
// as src/learning/'s deterministic correction engine. The values mirror the
// CLI's tuned ones in src/pipeline.py / src/config.py so both interfaces feel
// the same. No pre-roll padding is needed (unlike the CLI's
// PRE_SPEECH_PADDING_FRAMES): MediaRecorder captures continuously from the
// moment recording starts, so there is no "trim before speech" problem.
const VAD_CHECK_MS = 30; // analysis cadence, matches pipeline.py FRAME_MS
const VAD_MIN_SPEECH_MS = 150; // ignore blips shorter than this (MIN_SPEECH_FRAMES)
const VAD_MAX_RECORD_MS = 15000; // hard safety cap (MAX_RECORD_SECONDS)
const PTT_MAX_RECORD_MS = 30000; // a held button may talk for longer
const PTT_HOLD_MS = 350; // held at least this long -> release ends the clip

// A single hardcoded RMS threshold needed hand-tuning against one specific
// mic/room and silently failed on anyone else's -- hands-free and wake-word
// rely ENTIRELY on it to auto-detect speech. So it adapts: a slow-moving
// estimate of the ambient noise floor (updated only during frames judged
// non-speech) sets the bar, calibrating to whatever mic/room/gain this
// browser actually has.
let noiseFloorEma = 0.004;
const NOISE_FLOOR_EMA_ALPHA = 0.05; // slow, so one loud moment cannot drag the floor up
const VAD_RMS_ABS_FLOOR = 0.006; // a silent room must not make electrical noise read as speech

function currentSpeechThreshold() {
  return Math.max(VAD_RMS_ABS_FLOOR, noiseFloorEma * speechMargin());
}

let clipHandler = () => {};
export function setClipHandler(fn) {
  clipHandler = fn;
}

let mediaRecorder = null;
let chunks = [];
let vadTimerId = null;
let wakeTimerId = null;
let pttMode = false; // button/Space held: VAD must not end the clip
let starting = false; // getUserMedia / recorder setup in flight
let releaseAfterStart = null; // a release that arrived before recording began
let discardNext = false; // Escape: stop, but throw the clip away

/* ---- VAD while recording ------------------------------------------------ */
function checkVadFrame(state) {
  const rms = readMicRms();
  const isSpeech = rms > currentSpeechThreshold();
  if (!isSpeech) noiseFloorEma += NOISE_FLOOR_EMA_ALPHA * (rms - noiseFloorEma);

  state.elapsedMs += VAD_CHECK_MS;
  if (isSpeech) {
    state.speechMs += VAD_CHECK_MS;
    state.silenceMs = 0;
    if (state.speechMs >= VAD_MIN_SPEECH_MS) state.speechDetected = true;
  } else {
    state.silenceMs += VAD_CHECK_MS;
  }

  const trailingSilenceHit = !pttMode && state.speechDetected && state.silenceMs >= settings.pauseMs;
  const maxDurationHit = state.elapsedMs >= (pttMode ? PTT_MAX_RECORD_MS : VAD_MAX_RECORD_MS);
  if (trailingSilenceHit || maxDurationHit) stopRecording();
}

function startVad() {
  try {
    ensureMicAnalyser();
  } catch (err) {
    return; // Web Audio unavailable/blocked: degrade to manual-stop only
  }
  const state = { elapsedMs: 0, speechMs: 0, silenceMs: 0, speechDetected: false };
  vadTimerId = setInterval(() => checkVadFrame(state), VAD_CHECK_MS);
}

function stopVad() {
  if (vadTimerId !== null) {
    clearInterval(vadTimerId);
    vadTimerId = null;
  }
}

/* ---- wake-word idle listening ------------------------------------------ */
// Reuses the SAME analyser as the in-recording VAD -- no second getUserMedia
// or AudioContext. While armed it only watches for "has speech started"; the
// instant it has, it hands off to the normal recording pipeline, so the clip
// is captured and auto-stopped exactly like a real command. The transcript is
// then checked for the wake phrase instead of running the normal flow.
// Same cumulative-speechMs style as checkVadFrame (a brief dip below the
// threshold does not reset progress) and the same adaptive threshold.
function checkWakeFrame(state) {
  const rms = readMicRms();
  const isSpeech = rms > currentSpeechThreshold();
  if (!isSpeech) noiseFloorEma += NOISE_FLOOR_EMA_ALPHA * (rms - noiseFloorEma);
  if (isSpeech) state.speechMs += VAD_CHECK_MS;
  if (state.speechMs >= VAD_MIN_SPEECH_MS) {
    stopWakeListening();
    startRecording({ wakeCheck: true });
  }
}

export function startWakeListening() {
  if (!store.wakeWord || store.recording || starting || store.busy || wakeTimerId !== null) return;
  try {
    ensureMicAnalyser();
  } catch (err) {
    return; // Web Audio unavailable/blocked: degrade silently
  }
  const state = { speechMs: 0 };
  wakeTimerId = setInterval(() => checkWakeFrame(state), VAD_CHECK_MS);
  store.wakeArmed = true;
  emit("wake-armed", true);
}

export function stopWakeListening() {
  if (wakeTimerId !== null) {
    clearInterval(wakeTimerId);
    wakeTimerId = null;
  }
  if (store.wakeArmed) {
    store.wakeArmed = false;
    emit("wake-armed", false);
  }
}

// Re-arm point for every place the app returns to fully idle. Hands-free (a
// conversation already in progress) always wins: it keeps things going
// without requiring the wake phrase again, like real assistants.
export function maybeRearmWakeListening() {
  if (store.wakeWord && !store.handsFree && !store.recording && !store.busy) startWakeListening();
}

export { playChime };

/* ---- recording ----------------------------------------------------------- */
function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

export async function startRecording({ ptt = false, wakeCheck = false } = {}) {
  if (store.recording || starting) return;
  starting = true;
  pttMode = ptt;
  releaseAfterStart = null;
  stopWakeListening();
  stopSpeaking("new-turn");
  try {
    await ensureMic();
  } catch (err) {
    starting = false;
    toast("Microphone access denied: " + err.message, true);
    return;
  }
  chunks = [];
  discardNext = false;
  const mimeType = pickMimeType();
  const stream = getStream();
  mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.onstop = onRecordingStopped;
  store.recordingIsWakeCheck = wakeCheck;
  mediaRecorder.start();
  store.recording = true;
  starting = false;
  emit("recording", true);
  refreshPhase();
  if (!wakeCheck) setLive(ptt ? "Listening... release to send" : "Listening... pause to send, or tap to stop");
  startVad();

  if (releaseAfterStart) {
    // The user already let go while the mic was still spinning up.
    const { heldMs } = releaseAfterStart;
    releaseAfterStart = null;
    pttRelease(heldMs);
  }
}

export function stopRecording() {
  if (!store.recording) return;
  stopVad();
  store.recording = false;
  mediaRecorder.stop();
  emit("recording", false);
  refreshPhase();
}

/* Escape: end the recording but do not send what was captured. */
export function cancelRecording() {
  if (!store.recording) return false;
  discardNext = true;
  stopRecording();
  return true;
}

/* The press that began a push-to-talk recording has ended. A real hold sends
 * the clip on release; a quick tap instead turns it into tap-to-talk, where
 * the silence detector (or another tap) ends it. */
export function pttRelease(heldMs) {
  if (starting) {
    releaseAfterStart = { heldMs };
    return;
  }
  if (!store.recording) return;
  if (heldMs >= PTT_HOLD_MS) stopRecording();
  else {
    pttMode = false;
    setLive("Listening... pause to send, or tap to stop");
  }
}

export function isStarting() {
  return starting;
}

function onRecordingStopped() {
  const wasWakeCheck = store.recordingIsWakeCheck;
  store.recordingIsWakeCheck = false;
  const discard = discardNext;
  discardNext = false;
  const mimeType = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
  const blob = new Blob(chunks, { type: mimeType });
  chunks = [];
  refreshPhase();

  if (discard) {
    setLive("");
    maybeRearmWakeListening();
    return;
  }
  if (blob.size < 800) {
    setLive("");
    if (wasWakeCheck) maybeRearmWakeListening();
    else toast("Recording too short.", true);
    return;
  }
  clipHandler({ blob, mimeType, wasWakeCheck });
}
