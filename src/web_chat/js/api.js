/* The backend contract (src/chat_web.py). Nothing else in the UI talks to the
 * network, which keeps the surface that must stay in sync with the server
 * in one small file. */

async function readJson(res) {
  try {
    return await res.json();
  } catch (err) {
    return {};
  }
}

export async function health() {
  const res = await fetch("/api/health", { cache: "no-store" });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

export async function transcribe(form) {
  const res = await fetch("/api/transcribe", { method: "POST", body: form });
  const data = await readJson(res);
  if (!res.ok || data.error) throw new Error(data.error || "HTTP " + res.status);
  return data;
}

/* Returns the raw streaming Response; the caller reads the body. */
export function respond(payload, signal) {
  return fetch("/api/respond", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function tts(text, language, signal) {
  const res = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language: language || "en" }),
    signal,
  });
  if (!res.ok) {
    const data = await readJson(res);
    throw new Error(data.error || "HTTP " + res.status);
  }
  return res.blob();
}

export async function feedback(payload) {
  const res = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJson(res);
  if (!res.ok || data.error) throw new Error(data.error || "HTTP " + res.status);
  return data;
}
