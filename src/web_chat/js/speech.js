/* Spoken replies: a queue of per-sentence /api/tts calls drained strictly in
 * order, with real barge-in (starting a new turn cancels playback AND the
 * in-flight network calls).
 *
 * Replies are spoken sentence-by-sentence while the LLM is still streaming,
 * so sentence 1 plays while sentence 3 is still being generated. */

import * as api from "./api.js";
import { prepareAudio, resumeContext } from "./audio.js";
import { settings } from "./settings.js";
import { refreshPhase, store } from "./store.js";
import { sleep, toast } from "./ui.js";

// Without a client-side cap, a stuck request leaves "Speaking..." on screen
// forever with no sound and no explanation. Tamil gets a much longer
// allowance: it runs in its own worker process (pipeline.py's
// synthesize_tamil_via_worker), and the FIRST request after that worker
// (re)spawns pays interpreter startup + imports (torch, coqui-tts, numba JIT)
// + model load from scratch -- measured at ~35s. Every later request is fast
// (~1.5-2s per sentence, CPU-only), but the client cannot tell a cold spawn
// from a genuine hang, so this stays generous with real margin.
const TTS_TIMEOUT_MS = 15000;
// Stays above pipeline.py's _INDIC_WORKER_TIMEOUT_S (90s) so a truly hung
// worker is reported by the SERVER's own clear error, not a bare AbortError.
const TTS_TIMEOUT_MS_TA = 100000;

// Mirrors pipeline.py's _SENTENCE_END_RE (same four classes).
export const SENTENCE_END_RE = /[.!?।॥]/;

let currentAudio = null; // { audio, url, routed } for the clip playing now
// Resolves the in-flight playAndAwaitEnded promise, if any. Pausing an Audio
// element never fires "ended", so without this stopSpeaking() mid-chunk would
// leave the pump loop's `await` hanging forever.
let currentPlaybackResolve = null;
// Bumped on every new turn / barge-in. A pump loop compares its own captured
// value against the live one, which is immune to the reset race a plain
// boolean "aborted" flag would have if a new turn started while an old,
// still-unwinding pump had not yet noticed it should stop.
let ttsGeneration = 0;
let ttsQueue = []; // [{ text, language }], drained strictly in order
let ttsQueueRunning = false;
let ttsAbortController = null; // cancels in-flight/future /api/tts fetches for this turn
// Kept SEPARATE from ttsAbortController so one slow or timed-out TTS chunk can
// never cut off the LLM's still-streaming text reply.
let llmAbortController = null;
let llmStreamFinished = false; // /api/respond's reader loop + final flush are done
let replyAudioBlobs = []; // this turn's blobs, in order, for Replay
let queueDrainedResolvers = []; // fired once the whole queue finishes
// True once this turn's speech actually started playing. Lets stopSpeaking()
// tell "the user barged in on audio that was already talking" (expected, stay
// silent) from "the previous reply never got a chance to play before the next
// turn began" (worth telling the user about: Tamil synthesis can take far
// longer than a person takes to read a reply and ask the next question).
let anyAudioStartedThisTurn = false;

function updateSpeakingState() {
  const active = currentAudio !== null || (anyAudioStartedThisTurn && ttsQueueRunning);
  if (store.speaking === active) return;
  store.speaking = active;
  refreshPhase();
}

/* ---- per-turn lifecycle ------------------------------------------------ */

// Called once at the start of everything the assistant is about to say.
export function beginNewSpeechTurn() {
  ttsGeneration++;
  ttsAbortController = new AbortController();
  llmAbortController = new AbortController();
  llmStreamFinished = false;
  replyAudioBlobs = [];
  anyAudioStartedThisTurn = false;
  return ttsGeneration;
}

export const getLlmSignal = () => (llmAbortController ? llmAbortController.signal : undefined);
export const getReplyBlobs = () => replyAudioBlobs;
export function finishLlmStream() {
  llmStreamFinished = true;
}
export const isSpeechQueued = () => ttsQueueRunning || ttsQueue.length > 0;
export const isOutputRouted = () => currentAudio !== null && currentAudio.routed;

export function onQueueDrained(fn) {
  queueDrainedResolvers.push(fn);
}

function fireDrained() {
  const resolvers = queueDrainedResolvers;
  queueDrainedResolvers = [];
  resolvers.forEach((resolve) => resolve());
}

// reason "new-turn": the user started talking/typing again, as opposed to an
// explicit stop press or an about-to-replay call, where cutting audio off is
// the whole point and needs no explanation.
export function stopSpeaking(reason) {
  const silentlyLosingUnplayedReply =
    reason === "new-turn" && !anyAudioStartedThisTurn && (ttsQueueRunning || ttsQueue.length > 0);
  if (currentAudio) {
    currentAudio.audio.pause();
    currentAudio.audio.currentTime = 0;
    URL.revokeObjectURL(currentAudio.url);
    currentAudio = null;
  }
  ttsGeneration++; // invalidates any still-draining pump/replay loop immediately
  ttsQueue = [];
  queueDrainedResolvers = []; // the interrupted reply's re-arm hooks are moot
  if (currentPlaybackResolve) {
    const resolve = currentPlaybackResolve;
    currentPlaybackResolve = null;
    resolve(); // unblocks a loop that is mid-`await playAndAwaitEnded(...)`
  }
  // Real (server-visible) barge-in: aborting cancels the in-flight network
  // calls, not just local playback, so an interrupted reply stops burning the
  // shared GPU/Ollama instead of finishing unheard.
  if (ttsAbortController) {
    ttsAbortController.abort();
    ttsAbortController = null;
  }
  if (llmAbortController) {
    llmAbortController.abort();
    llmAbortController = null;
  }
  updateSpeakingState();
  if (silentlyLosingUnplayedReply) {
    toast("Didn't get to speak the previous reply before you moved on. Its text is still above.", false);
  }
}

/* ---- playback ----------------------------------------------------------- */

// Plays one blob and resolves once it has actually finished (native "ended"),
// or stopSpeaking() interrupts it, or autoplay is blocked -- it never hangs
// the caller on a rejected .play().
async function playAndAwaitEnded(blob) {
  anyAudioStartedThisTurn = true;
  let resolveFn;
  const done = new Promise((resolve) => {
    resolveFn = resolve;
  });
  currentPlaybackResolve = resolveFn;
  await resumeContext();
  if (currentPlaybackResolve !== resolveFn) return; // stopped while we awaited

  const handle = prepareAudio(blob, { rate: settings.rate, volume: settings.volume });
  currentAudio = handle;
  updateSpeakingState();
  const finish = () => {
    URL.revokeObjectURL(handle.url);
    if (currentAudio === handle) currentAudio = null;
    if (currentPlaybackResolve === resolveFn) currentPlaybackResolve = null;
    updateSpeakingState();
    resolveFn();
  };
  handle.audio.addEventListener("ended", finish, { once: true });
  handle.audio.play().catch(finish); // autoplay blocked: do not hang the queue
  return done;
}

export async function playAudioBlobsSequentially(blobs) {
  stopSpeaking(); // stop anything currently playing/queued first
  const myGeneration = ttsGeneration;
  for (const blob of blobs) {
    if (myGeneration !== ttsGeneration) break; // a barge-in during replay stops replay too
    await playAndAwaitEnded(blob);
    if (myGeneration !== ttsGeneration) break;
  }
}

// The /api/tts call shared by every playback path. Uses ttsAbortController
// (not llmAbortController) so a slow chunk can only cut off remaining speech,
// never the LLM's text stream.
async function fetchTtsBlob(text, language) {
  if (!ttsAbortController) return null;
  const controller = ttsAbortController;
  const signal = controller.signal;
  const timeoutMs = language === "ta" ? TTS_TIMEOUT_MS_TA : TTS_TIMEOUT_MS;
  const timeoutId = setTimeout(() => {
    if (!signal.aborted) {
      controller.timedOut = true;
      controller.abort();
    }
  }, timeoutMs);
  try {
    return await api.tts(text, language, signal);
  } catch (err) {
    if (err.name === "AbortError") {
      // This call's OWN timeout is worth telling the user about; a deliberate
      // barge-in aborting it is expected, and noise if reported.
      if (controller.timedOut) {
        toast("Speech synthesis is taking too long. Showing text only for this reply.", true);
      }
    } else {
      toast("TTS failed: " + err.message, true);
    }
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}

// Drains ttsQueue strictly in order: fetch chunk N's audio, play it fully,
// then chunk N+1 -- whose TEXT may still be arriving from the LLM, so it
// polls briefly rather than giving up when the queue is momentarily empty.
// Only one instance runs at a time; later enqueueTtsChunk() calls just add to
// the queue the running instance is already draining.
async function pumpTtsQueue(myGeneration) {
  if (ttsQueueRunning) return;
  ttsQueueRunning = true;
  while (myGeneration === ttsGeneration) {
    if (ttsQueue.length === 0) {
      if (llmStreamFinished) break; // truly done: no more sentences are coming
      await sleep(50);
      continue;
    }
    const item = ttsQueue.shift();
    const idx = replyAudioBlobs.findIndex((b) => b === null);
    const blob = await fetchTtsBlob(item.text, item.language);
    if (myGeneration !== ttsGeneration) break;
    if (blob) {
      if (idx >= 0) replyAudioBlobs[idx] = blob;
      await playAndAwaitEnded(blob);
      if (myGeneration !== ttsGeneration) break;
    }
  }
  ttsQueueRunning = false;
  updateSpeakingState();
  if (ttsQueue.length > 0) {
    // A newer turn's chunks arrived while this instance was exiting
    // (generation mismatch): hand off to a fresh pump for the CURRENT
    // generation instead of stranding them.
    pumpTtsQueue(ttsGeneration);
    return;
  }
  if (myGeneration === ttsGeneration) fireDrained();
}

export function enqueueTtsChunk(text, language) {
  if (!text.trim()) return;
  ttsQueue.push({ text, language });
  replyAudioBlobs.push(null); // placeholder: keeps indices aligned as chunks finish
  pumpTtsQueue(ttsGeneration);
}

// Single-utterance playback (the "please repeat that" reply). Awaits only the
// network fetch, then plays in the background; callers that need to know
// whether it actually spoke get that from the returned blobs array.
export async function synthesizeAndPlay(text, language) {
  beginNewSpeechTurn();
  llmStreamFinished = true; // exactly one chunk, nothing more is ever coming
  replyAudioBlobs = [null];
  const myGeneration = ttsGeneration;
  const blob = await fetchTtsBlob(text, language);
  if (myGeneration !== ttsGeneration || !blob) return null;
  replyAudioBlobs[0] = blob;
  playAndAwaitEnded(blob).then(() => {
    if (myGeneration !== ttsGeneration) return;
    fireDrained();
  });
  return replyAudioBlobs;
}
