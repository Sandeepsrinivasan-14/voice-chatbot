/* Wake-phrase matching: pure text logic, kept free of DOM and audio so it can
 * be unit-tested under plain Node (tests/test_frontend_logic.py). */

export const WAKE_PHRASES = ["hey chatbot", "hi chatbot", "hey chat bot", "ok chatbot"];
const WAKE_LEADING_SLOP_CHARS = 12; // tolerate a little leading noise before the phrase

const WAKE_PHRASE_REGEXES = WAKE_PHRASES.map((phrase) => {
  const words = phrase.split(" ").map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  // Tolerates STT inserting punctuation/extra whitespace between words,
  // e.g. "Hey, Chatbot" or "hey  chatbot".
  return new RegExp(words.join("[,.\\s]+") + "\\b", "i");
});

// Matches against text.toLowerCase() (same length/index as the original for
// this alphabet) so the match position slices the ORIGINAL text for the
// remainder, preserving real capitalisation/punctuation for the LLM.
export function matchWakePhrase(rawText) {
  const lower = rawText.toLowerCase();
  let best = null;
  for (const re of WAKE_PHRASE_REGEXES) {
    const m = re.exec(lower);
    if (m && (best === null || m.index < best.index)) best = m;
  }
  if (!best || best.index > WAKE_LEADING_SLOP_CHARS) return { hasWake: false, remainder: "" };
  const remainder = rawText.slice(best.index + best[0].length).replace(/^[,.\s]+/, "").trim();
  return { hasWake: true, remainder };
}
