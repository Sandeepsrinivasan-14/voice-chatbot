/* The conversation flow: a typed message or a recorded clip becomes a
 * transcript, then a streamed LLM reply that is spoken sentence-by-sentence.
 * This is where recorder.js, speech.js and transcript.js are wired together;
 * the behaviour deliberately mirrors the original single-file UI. */

import * as api from "./api.js";
import { maybeRearmWakeListening, playChime, startRecording, stopWakeListening } from "./recorder.js";
import {
  SENTENCE_END_RE,
  beginNewSpeechTurn,
  enqueueTtsChunk,
  finishLlmStream,
  getLlmSignal,
  getReplyBlobs,
  onQueueDrained,
  stopSpeaking,
  synthesizeAndPlay,
} from "./speech.js";
import { settings } from "./settings.js";
import { emit, refreshPhase, store } from "./store.js";
import * as transcript from "./transcript.js";
import { setLive, toast } from "./ui.js";
import { matchWakePhrase } from "./wake.js";

// Local, no-auth identity for the continuous-learning layer (src/learning/).
// profileId survives across tabs/restarts on this browser; sessionId is fresh
// per tab load. Neither is sent anywhere off this machine -- they are
// partition keys the server uses to keep one person's corrections from
// affecting anyone else's. A server with learning disabled (the default)
// ignores both.
function getOrCreateId(storage, key) {
  try {
    let id = storage.getItem(key);
    if (!id) {
      id = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
      storage.setItem(key, id);
    }
    return id;
  } catch (err) {
    return "local"; // storage unavailable (private mode, etc.): the server falls back too
  }
}
const profileId = getOrCreateId(localStorage, "vc_profile_id");
const sessionId = getOrCreateId(sessionStorage, "vc_session_id");

let history = []; // [{role, content}, ...] sent to /api/respond

// Replies are chunked sentence-by-sentence for these languages; any other
// detected language is spoken as one call for the whole reply. Tamil used to
// be excluded because its old engine took 20-150s per call; the current
// FastPitch+HiFiGAN engine is ~1.5-2s/sentence on CPU, so it streams too.
const CHUNKED_LANGUAGES = new Set(["en", "hi", "ta"]);

export function setBusy(value, quiet = false) {
  store.busy = value;
  store.quietBusy = value && quiet;
  refreshPhase();
  emit("busy", value);
}

export function newConversation() {
  stopSpeaking();
  history = [];
  transcript.clear();
  setLive("");
}

/* ---- hands-free / wake-word re-arm hooks ------------------------------- */

// Fires once the WHOLE reply has finished playing, not just the first chunk:
// with multi-sentence streaming speech, re-arming on chunk 1's end would
// reopen the mic while chunks 2+ are still playing.
function hookHandsFreeRestart() {
  onQueueDrained(restartHandsFree);
}

// Only acts if hands-free is NOT also driving the conversation (hands-free
// wins: it already keeps things going without the wake phrase). Armed only
// once the reply finished speaking, so an echo-tainted false detection cannot
// cut the reply off mid-sentence via startRecording()'s stopSpeaking().
function hookWakeWordRestart() {
  onQueueDrained(() => {
    if (store.wakeWord && !store.handsFree && !store.recording && !store.busy) maybeRearmWakeListening();
  });
}

function restartHandsFree() {
  if (store.handsFree && !store.recording && !store.busy) startRecording();
}

/* After a reply that was never spoken (spoken replies turned off), the same
 * "keep the conversation going" logic runs immediately. */
function continueWithoutSpeech() {
  if (store.handsFree) restartHandsFree();
  else maybeRearmWakeListening();
}

/* ---- the LLM turn ------------------------------------------------------- */

// Streams the LLM's reply and speaks it sentence-by-sentence as it arrives,
// instead of waiting for the whole reply to generate AND synthesize before
// any audio plays. Sentence 1's /api/tts call fires while the LLM is still
// generating sentence 3.
async function sendToLLM(userText, language) {
  language = language || (settings.language === "auto" ? "en" : settings.language);
  const chunked = CHUNKED_LANGUAGES.has(language);
  const speak = settings.speak;
  setBusy(true);
  setLive("Thinking...");
  const bubble = transcript.beginAssistant();
  let fullText = "";
  let sentenceBuffer = ""; // unflushed tail since the last sentence boundary
  let anyChunkQueued = false;
  let completed = false;
  beginNewSpeechTurn();
  const thisReplyBlobs = getReplyBlobs(); // the array object itself (see addReplyActions)
  try {
    const res = await api.respond({ message: userText, history, language }, getLlmSignal());
    if (!res.ok) {
      let message = "HTTP " + res.status;
      try {
        message = (await res.json()).error || message;
      } catch (err) {
        /* keep the status text */
      }
      throw new Error(message);
    }
    if (!res.body) throw new Error("Streaming not supported by this browser.");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let first = true;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const piece = decoder.decode(value, { stream: true });
      if (!piece) continue;
      if (first) {
        first = false;
        setLive("Writing the reply...");
      }
      fullText += piece;
      bubble.push(piece);

      if (!speak || !chunked) continue; // accumulate; spoken as one call below
      // Slice off every complete sentence found so far and queue it for
      // speech immediately -- do not wait for the reply to finish.
      sentenceBuffer += piece;
      let match;
      while ((match = SENTENCE_END_RE.exec(sentenceBuffer))) {
        const boundary = match.index + 1;
        const sentence = sentenceBuffer.slice(0, boundary).trim();
        sentenceBuffer = sentenceBuffer.slice(boundary);
        if (sentence) {
          enqueueTtsChunk(sentence, language);
          anyChunkQueued = true;
        }
      }
    }
    if (!fullText.trim()) fullText = "(no response)";
    if (speak) {
      if (chunked) {
        if (sentenceBuffer.trim()) {
          enqueueTtsChunk(sentenceBuffer.trim(), language);
          anyChunkQueued = true;
        }
      } else if (fullText.trim()) {
        enqueueTtsChunk(fullText.trim(), language);
        anyChunkQueued = true;
      }
    }
    finishLlmStream(); // tells the pump nothing more is coming, so it can stop polling
    history.push({ role: "user", content: userText });
    history.push({ role: "assistant", content: fullText });
    bubble.finish(fullText);
    transcript.addReplyActions(bubble.bubble, anyChunkQueued ? thisReplyBlobs : null, () => fullText);
    completed = true;

    if (anyChunkQueued) {
      hookHandsFreeRestart();
      hookWakeWordRestart();
    }
  } catch (err) {
    finishLlmStream();
    if (err.name === "AbortError") {
      bubble.abort(fullText);
    } else {
      bubble.fail("Error: " + err.message);
      toast(err.message, true);
    }
  } finally {
    setLive("");
    setBusy(false);
    if (!anyChunkQueued) {
      if (completed && !speak) continueWithoutSpeech();
      else maybeRearmWakeListening(); // nothing was ever queued: safe to re-arm now
    }
  }
}

export async function handleUserText(text) {
  text = text.trim();
  if (!text) return;
  stopSpeaking("new-turn");
  stopWakeListening();
  transcript.addMessage("user", text, { language: settings.language });
  await sendToLLM(text);
}

/* ---- a recorded clip ---------------------------------------------------- */

export async function handleClip({ blob, mimeType, wasWakeCheck }) {
  setBusy(true, wasWakeCheck);
  if (!wasWakeCheck) setLive("Transcribing...");
  try {
    const form = new FormData();
    const ext = mimeType.includes("ogg") ? "ogg" : "webm";
    form.append("audio", blob, `clip.${ext}`);
    form.append("profile_id", profileId);
    form.append("session_id", sessionId);
    form.append("language", settings.language);
    const data = await api.transcribe(form);
    emit("stt-device", data.whisper_device); // fills in "STT on ?" once known

    if (wasWakeCheck) {
      await handleWakeCheck(data);
      return;
    }

    if (!data.accepted) {
      await handleRejected(data);
      return;
    }

    // data.event_id is only present when the server has the continuous-
    // learning layer enabled (CONFIG.learning_enabled): the correction
    // affordance only appears in that case. data.text is already the
    // auto-corrected transcript if the server applied one with high enough
    // confidence; data.original_text carries the raw STT output.
    let meta = `confidence ${data.avg_logprob.toFixed(2)}`;
    if (data.auto_corrected) meta += " · auto-corrected from memory";
    const correction = data.event_id
      ? {
          eventId: data.event_id,
          profileId,
          sessionId,
          avgLogprob: data.avg_logprob,
          rawText: data.original_text || data.text,
          suggestions: data.correction_suggestions || [],
          language: data.language,
        }
      : null;
    transcript.addMessage("user", data.text, { meta, correction, language: data.language });
    setBusy(false); // sendToLLM re-locks
    await sendToLLM(data.text, data.language);
  } catch (err) {
    toast(err.message, true);
    setLive("");
    setBusy(false);
    maybeRearmWakeListening(); // a network hiccup must not permanently kill wake-listening
  }
}

async function handleWakeCheck(data) {
  const { hasWake, remainder } = matchWakePhrase(data.text || "");
  if (!hasWake) {
    setLive("");
    setBusy(false);
    maybeRearmWakeListening();
    return;
  }
  if (remainder) {
    // One-shot "hey chatbot, what's the weather": the remainder is the real command.
    transcript.addMessage("user", remainder, { meta: "via wake phrase", language: data.language });
    setBusy(false); // sendToLLM re-locks
    await sendToLLM(remainder, data.language);
    return;
  }
  // The wake phrase alone: chime, then open a normal full recording.
  await playChime();
  setBusy(false);
  startRecording();
}

async function handleRejected(data) {
  transcript.addMessage("rejected", '"' + data.text + '"', {
    meta: `low confidence (${data.avg_logprob.toFixed(2)}) -- not sent to the model`,
  });
  setLive("");
  const askAgain = "Sorry, could you say that again?";
  const bubble = transcript.addMessage("assistant", askAgain);
  const clip = settings.speak ? await synthesizeAndPlay(askAgain) : null;
  if (clip) {
    transcript.addReplyActions(bubble, clip, () => askAgain);
    hookHandsFreeRestart();
    hookWakeWordRestart();
  }
  setBusy(false);
  if (!clip) continueWithoutSpeech();
}
