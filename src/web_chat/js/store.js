/* Tiny shared state + event bus. Deliberately not a framework: the whole UI
 * is a handful of flags, and every module reads/writes them through here. */

export const store = {
  phase: "idle", // idle | listening | thinking | speaking (derived, see refreshPhase)
  recording: false, // a clip is being recorded right now
  recordingIsWakeCheck: false, // ...but only to check it for the wake phrase
  busy: false, // transcribing / waiting on the model
  quietBusy: false, // ...but only a background wake-phrase check: not worth animating
  speaking: false, // assistant audio is playing
  handsFree: false,
  wakeWord: false,
  wakeArmed: false, // idle-listening for the wake phrase
};

const bus = new EventTarget();

export function on(type, fn) {
  const handler = (e) => fn(e.detail);
  bus.addEventListener(type, handler);
  return () => bus.removeEventListener(type, handler);
}

export function emit(type, detail) {
  bus.dispatchEvent(new CustomEvent(type, { detail }));
}

/* The orb, background and labels all follow ONE derived phase so they can
 * never disagree. A silent wake-check recording is not "listening" to the
 * user -- it is background work -- so it keeps the app looking idle. */
export function refreshPhase() {
  let next = "idle";
  if (store.recording && !store.recordingIsWakeCheck) next = "listening";
  else if (store.speaking) next = "speaking";
  else if (store.busy && !store.quietBusy) next = "thinking";
  if (next === store.phase) return;
  store.phase = next;
  document.documentElement.setAttribute("data-phase", next);
  emit("phase", next);
}
