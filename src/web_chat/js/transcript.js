/* The conversation view: message bubbles (including the streaming assistant
 * bubble), per-message actions, the inline transcript-correction form, and
 * the plain-data log that powers transcript export. */

import * as api from "./api.js";
import { playAudioBlobsSequentially, stopSpeaking } from "./speech.js";
import { emit } from "./store.js";
import { $, el, icon, toast } from "./ui.js";

const messagesEl = $("messages");
let emptyState = $("empty-state");

/* The exportable record of the conversation: plain data, no DOM. */
export const entries = [];

function now() {
  return new Date();
}

function clock(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function nearBottom() {
  return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 140;
}

export function scrollToBottom(force) {
  if (force || nearBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideEmptyState() {
  if (emptyState && emptyState.parentNode) emptyState.remove();
}

function pushEntry(role, text, extra) {
  entries.push({ role, text, time: now(), ...extra });
  emit("transcript", entries.length);
}

function buildRow(role) {
  hideEmptyState();
  const row = el("div", "row " + role);
  const avatar = el("div", "avatar");
  avatar.appendChild(icon(role === "user" ? "user" : role === "rejected" ? "help" : "spark"));
  const bubble = el("div", "bubble");
  row.append(avatar, bubble);
  return { row, bubble };
}

function metaLine(bubble, parts, time) {
  const meta = el("div", "bubble-meta");
  for (const part of parts) {
    if (!part) continue;
    meta.appendChild(el("span", part.cls || "", part.text));
  }
  meta.appendChild(el("span", "", clock(time)));
  bubble.appendChild(meta);
  return meta;
}

/* A finished message (user text, a rejected low-confidence transcript, or a
 * fixed assistant line like "could you say that again?"). */
export function addMessage(role, text, opts = {}) {
  const { row, bubble } = buildRow(role);
  bubble.appendChild(el("div", "bubble-text", text));
  const time = now();
  const parts = [];
  if (opts.meta) parts.push({ text: opts.meta, cls: "meta-conf" });
  metaLine(bubble, parts, time);
  messagesEl.appendChild(row);
  scrollToBottom(true);
  if (role !== "rejected") pushEntry(role, text, { language: opts.language, meta: opts.meta });
  if (opts.correction) attachCorrectionUI(bubble, text, opts.correction);
  return bubble;
}

/* The assistant's streaming bubble: starts as typing dots, fills in as
 * pieces arrive, and is finalised (or failed) exactly once. */
export function beginAssistant() {
  const { row, bubble } = buildRow("assistant");
  const textEl = el("div", "bubble-text");
  const dots = el("span", "typing-dots");
  dots.append(el("i"), el("i"), el("i"));
  const sr = el("span", "sr-only", "The assistant is thinking");
  textEl.append(dots, sr);
  bubble.appendChild(textEl);
  bubble.classList.add("streaming");
  messagesEl.appendChild(row);
  messagesEl.setAttribute("aria-busy", "true");
  scrollToBottom(true);

  let started = false;
  let pieces = 0;

  return {
    bubble,
    // Append one streamed piece. Each piece fades in; very long replies are
    // periodically collapsed back into a single text node to keep the DOM small.
    push(piece) {
      if (!started) {
        textEl.textContent = "";
        started = true;
      }
      const span = el("span", "tok", piece);
      textEl.appendChild(span);
      if (++pieces > 240) {
        const merged = document.createTextNode(textEl.textContent);
        textEl.textContent = "";
        textEl.appendChild(merged);
        pieces = 0;
      }
      scrollToBottom();
    },
    finish(fullText) {
      bubble.classList.remove("streaming");
      textEl.textContent = fullText;
      messagesEl.setAttribute("aria-busy", "false");
      const time = now();
      metaLine(bubble, [], time);
      pushEntry("assistant", fullText, {});
      scrollToBottom();
    },
    fail(message) {
      bubble.classList.remove("streaming");
      textEl.textContent = message;
      messagesEl.setAttribute("aria-busy", "false");
    },
    // Aborted by barge-in: keep whatever text already arrived.
    abort(partialText) {
      bubble.classList.remove("streaming");
      messagesEl.setAttribute("aria-busy", "false");
      if (!started) {
        row.remove();
        return;
      }
      textEl.textContent = partialText;
      metaLine(bubble, [{ text: "interrupted" }], now());
      pushEntry("assistant", partialText, { interrupted: true });
    },
  };
}

/* Replay / stop / copy under an assistant reply. `audioBlobs` is the live
 * array speech.js keeps filling in, so a click shortly after the reply
 * appears still plays whatever has finished synthesising by then. */
export function addReplyActions(bubble, audioBlobs, text) {
  const actions = el("div", "bubble-actions");

  if (audioBlobs) {
    const replay = el("button", "mini-btn");
    replay.type = "button";
    replay.append(icon("replay"), document.createTextNode("Replay"));
    // .filter(Boolean) runs at CLICK time, not here.
    replay.addEventListener("click", () => playAudioBlobsSequentially(audioBlobs.filter(Boolean)));
    const stop = el("button", "mini-btn");
    stop.type = "button";
    stop.append(icon("stop"), document.createTextNode("Stop"));
    stop.addEventListener("click", () => stopSpeaking());
    actions.append(replay, stop);
  }

  const copy = el("button", "mini-btn");
  copy.type = "button";
  copy.append(icon("copy"), document.createTextNode("Copy"));
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text());
      toast("Copied to clipboard");
    } catch (err) {
      toast("Could not copy: " + err.message, true);
    }
  });
  actions.appendChild(copy);
  bubble.appendChild(actions);
}

export function clear() {
  entries.length = 0;
  messagesEl.textContent = "";
  emptyState = restoreEmptyState();
  messagesEl.appendChild(emptyState);
  emit("transcript", 0);
}

let emptyTemplate = null;
export function rememberEmptyState() {
  emptyTemplate = emptyState ? emptyState.cloneNode(true) : null;
}
function restoreEmptyState() {
  return emptyTemplate ? emptyTemplate.cloneNode(true) : el("div", "empty");
}

/* ---- transcript correction (continuous-learning feedback) ---------------- */
// Only attached to voice-transcribed user bubbles, and only when the server
// returned an event_id (i.e. the learning layer is enabled server-side).
// Typed messages have no STT output to correct.
function escapeForWordRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function applyPhrase(text, originalPhrase, correctedPhrase) {
  const pattern = new RegExp("\\b" + escapeForWordRegex(originalPhrase) + "\\b", "i");
  return pattern.test(text) ? text.replace(pattern, correctedPhrase) : text;
}

function attachCorrectionUI(bubble, displayedText, correction) {
  const actions = el("div", "bubble-actions");
  const btn = el("button", "mini-btn");
  btn.type = "button";
  btn.append(icon("edit"), document.createTextNode("Correct"));
  btn.addEventListener("click", () => openCorrectionForm(bubble, displayedText, correction, btn));
  actions.appendChild(btn);
  bubble.appendChild(actions);

  const suggestions = correction.suggestions || [];
  if (suggestions.length) {
    // Surface only the strongest suggestion, as a low-confidence "did you
    // mean": the user still reviews and saves it like any other correction.
    const top = suggestions[0];
    const note = el(
      "div",
      "suggestion-note",
      `Did you mean "${top.corrected_phrase}" instead of "${top.original_phrase}"? `,
    );
    const review = el("button", "", "Review");
    review.type = "button";
    review.addEventListener("click", () => {
      const prefill = applyPhrase(displayedText, top.original_phrase, top.corrected_phrase);
      openCorrectionForm(bubble, displayedText, correction, btn, prefill);
    });
    note.appendChild(review);
    bubble.appendChild(note);
  }
}

function openCorrectionForm(bubble, displayedText, correction, triggerBtn, prefill) {
  const existing = bubble.querySelector(".correction-form");
  if (existing) {
    existing.remove();
    return;
  }
  const form = el("div", "correction-form");
  const textarea = el("textarea");
  textarea.setAttribute("aria-label", "Corrected transcript");
  textarea.value = prefill !== undefined ? prefill : displayedText;
  const actions = el("div", "correction-actions");
  const save = el("button", "mini-btn", "Save correction");
  save.type = "button";
  // The feedback record's original_text is always the raw STT output
  // (correction.rawText), even if the bubble currently shows an
  // already-auto-corrected version: the immutable record is about what the
  // model actually said, not what the user saw.
  save.addEventListener("click", () =>
    submitCorrection(bubble, correction.rawText, textarea.value, correction, triggerBtn, form),
  );
  const cancel = el("button", "mini-btn", "Cancel");
  cancel.type = "button";
  cancel.addEventListener("click", () => form.remove());
  actions.append(save, cancel);
  form.append(textarea, actions);
  bubble.appendChild(form);
  textarea.focus();
}

async function submitCorrection(bubble, originalText, correctedRaw, correction, triggerBtn, form) {
  const correctedText = correctedRaw.trim();
  if (!correctedText || correctedText === originalText) {
    form.remove();
    return;
  }
  try {
    await api.feedback({
      event_id: correction.eventId,
      profile_id: correction.profileId,
      session_id: correction.sessionId,
      original_text: originalText,
      corrected_text: correctedText,
      avg_logprob: correction.avgLogprob,
      language: correction.language,
    });
    form.remove();
    if (triggerBtn) triggerBtn.remove();
    bubble.querySelectorAll(".suggestion-note").forEach((n) => n.remove());
    const saved = el("div", "bubble-meta");
    saved.appendChild(el("span", "correction-saved", "Correction saved. This helps future transcriptions for you."));
    bubble.appendChild(saved);
  } catch (err) {
    toast("Could not save correction: " + err.message, true);
  }
}
