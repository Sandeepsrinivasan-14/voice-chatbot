/* Small DOM helpers shared by every module: element lookup, icons, toasts,
 * and the two accessibility announcers. */

export const $ = (id) => document.getElementById(id);

const SVG_NS = "http://www.w3.org/2000/svg";

export function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "ico");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", "#i-" + name);
  svg.appendChild(use);
  return svg;
}

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ---- screen-reader announcements -------------------------------------- */
let lastAnnouncement = "";
export function announce(text) {
  const region = $("sr-status");
  if (!region || !text || text === lastAnnouncement) return;
  lastAnnouncement = text;
  // Clearing first makes a repeated identical string announce again later.
  region.textContent = "";
  setTimeout(() => {
    region.textContent = text;
  }, 30);
}

export function resetAnnouncement() {
  lastAnnouncement = "";
}

/* ---- live caption under the phase title -------------------------------- */
export function setLive(text) {
  const caption = $("phase-caption");
  if (!caption) return;
  caption.dataset.live = text ? "1" : "";
  caption.textContent = text || caption.dataset.idle || "";
  if (text) announce(text);
  else resetAnnouncement();
}

export function setIdleCaption(text) {
  const caption = $("phase-caption");
  caption.dataset.idle = text;
  if (!caption.dataset.live) caption.textContent = text;
}

export function getLive() {
  const caption = $("phase-caption");
  return caption.dataset.live ? caption.textContent : "";
}

/* ---- connection status pill -------------------------------------------- */
export function setStatus(state, text) {
  const dot = $("status-dot");
  dot.className = "status-dot" + (state ? " " + state : "");
  $("status-text").textContent = text;
}

/* ---- toasts ------------------------------------------------------------- */
export function toast(message, isError) {
  const host = $("toasts");
  const node = el("div", "toast" + (isError ? " error" : ""), message);
  node.setAttribute("role", isError ? "alert" : "status");
  host.appendChild(node);
  while (host.children.length > 3) host.firstElementChild.remove();
  setTimeout(
    () => {
      node.classList.add("leaving");
      setTimeout(() => node.remove(), 260);
    },
    isError ? 5200 : 3600,
  );
}

export function isTypingTarget(target) {
  if (!target || !target.tagName) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}
