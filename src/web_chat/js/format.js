/* Transcript serialisation: pure functions (no DOM) so the output formats can
 * be unit-tested under plain Node (tests/test_frontend_logic.py). */

const ROLE_LABEL = { user: "You", assistant: "Assistant" };

export function stamp(date) {
  const p = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}`;
}

function hhmm(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function buildExport(format, log, generatedAt = new Date()) {
  const when = generatedAt.toLocaleString();
  if (format === "json") {
    return JSON.stringify(
      {
        app: "Local Voice Chatbot",
        exported_at: generatedAt.toISOString(),
        messages: log.map((e) => ({
          role: e.role,
          text: e.text,
          time: e.time.toISOString(),
          ...(e.language ? { language: e.language } : {}),
          ...(e.meta ? { note: e.meta } : {}),
          ...(e.interrupted ? { interrupted: true } : {}),
        })),
      },
      null,
      2,
    );
  }
  if (format === "md") {
    const lines = ["# Voice chat transcript", "", `_Exported ${when}. Recorded and generated entirely on this machine._`, ""];
    for (const e of log) {
      lines.push(`**${ROLE_LABEL[e.role] || e.role}** · ${hhmm(e.time)}`, "", e.text.replace(/\n/g, "  \n"), "");
    }
    return lines.join("\n");
  }
  const lines = [`Voice chat transcript (exported ${when})`, ""];
  for (const e of log) lines.push(`[${hhmm(e.time)}] ${ROLE_LABEL[e.role] || e.role}: ${e.text}`);
  return lines.join("\n") + "\n";
}
