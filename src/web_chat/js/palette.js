/* Per-phase colour palettes shared by the WebGL orb and the canvas
 * visualisers. Values are 0..1 RGB. The CSS custom properties in theme.css
 * (--p1/--p2/--p3) mirror these so every layer stays in tune. */

function hex(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export const PHASES = ["idle", "listening", "thinking", "speaking"];

const PALETTE = {
  idle: [hex("#4f46e5"), hex("#a855f7"), hex("#38bdf8")],
  listening: [hex("#0891b2"), hex("#10b981"), hex("#67e8f9")],
  thinking: [hex("#7c3aed"), hex("#f59e0b"), hex("#fcd34d")],
  speaking: [hex("#db2777"), hex("#6366f1"), hex("#f9a8d4")],
};

/* Smoothly cross-fades weights and colours so a phase change glides. */
export class PhaseMixer {
  constructor() {
    this.weights = [1, 0, 0, 0];
    this.colors = PALETTE.idle.map((c) => c.slice());
  }

  update(dt, phase) {
    const k = 1 - Math.exp(-dt * 3.2);
    for (let i = 0; i < 4; i++) {
      const target = PHASES[i] === phase ? 1 : 0;
      this.weights[i] += (target - this.weights[i]) * k;
    }
    for (let c = 0; c < 3; c++) {
      for (let ch = 0; ch < 3; ch++) {
        let v = 0;
        for (let i = 0; i < 4; i++) v += PALETTE[PHASES[i]][c][ch] * this.weights[i];
        this.colors[c][ch] = v;
      }
    }
  }

  css(index, alpha = 1) {
    const [r, g, b] = this.colors[index];
    return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${alpha})`;
  }
}
