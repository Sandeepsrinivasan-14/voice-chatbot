/* Transcript export. Everything is built in the browser from the in-memory
 * conversation log; nothing is uploaded. */

import { buildExport, stamp } from "./format.js";
import { entries } from "./transcript.js";
import { $, toast } from "./ui.js";

const MIME = { md: "text/markdown", txt: "text/plain", json: "application/json" };

function download(filename, content, mime) {
  const blob = new Blob([content], { type: mime + ";charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e2) {
      ok = false;
    }
    area.remove();
    return ok;
  }
}

export function initExport() {
  const dialog = $("export-dialog");
  const summary = $("export-summary");

  function open() {
    if (entries.length === 0) {
      toast("Nothing to export yet. Say or type something first.");
      return;
    }
    const n = entries.length;
    summary.textContent = `${n} message${n === 1 ? "" : "s"} from this conversation.`;
    if (!dialog.open) dialog.showModal();
  }

  dialog.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const format = btn.dataset.export;
      const now = new Date();
      if (format === "copy") {
        const ok = await copyText(buildExport("txt", entries, now));
        toast(ok ? "Transcript copied to clipboard" : "Could not copy the transcript", !ok);
      } else {
        download(`voice-chat-${stamp(now)}.${format}`, buildExport(format, entries, now), MIME[format]);
        toast(`Saved voice-chat-${stamp(now)}.${format}`);
      }
      dialog.close();
    });
  });

  return { open };
}
