/* Live waveform + scrolling spectrogram. Both are drawn from the analyser
 * frame the main loop samples (microphone while listening, assistant audio
 * while speaking); with no input they relax into a calm resting line. */

const SPECTRO_BINS = 72; // ~0-3.4 kHz at 48 kHz / fftSize 1024: where speech lives

export function createVisualizer(waveCanvas, spectroCanvas) {
  const wave = waveCanvas.getContext("2d");
  const spectro = spectroCanvas.getContext("2d", { willReadFrequently: false });
  let dpr = 1;
  let amp = 0; // smoothed overall amplitude, so the line eases in/out
  let scrollCarry = 0;
  let spectroDirty = true;

  function fit(canvas) {
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      return true;
    }
    return false;
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    fit(waveCanvas);
    if (fit(spectroCanvas)) spectroDirty = true;
  }

  /* frame: { time: Float32Array, freq: Uint8Array } | null
   * mixer: PhaseMixer (colours); t: seconds; level: 0..1 smoothed loudness */
  function draw(frame, mixer, t, level, dt) {
    if (waveCanvas.clientWidth > 0) drawWave(frame, mixer, t, level, dt);
    if (spectroCanvas.clientWidth > 0) drawSpectro(frame, mixer, dt);
  }

  function drawWave(frame, mixer, t, level, dt) {
    const w = waveCanvas.width;
    const h = waveCanvas.height;
    wave.clearRect(0, 0, w, h);
    amp += (Math.min(1, level * 1.6) - amp) * (1 - Math.exp(-dt * 14));

    const mid = h / 2;
    const points = Math.min(96, Math.floor(w / (3 * dpr)));
    const grad = wave.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, mixer.css(0, 0.0));
    grad.addColorStop(0.2, mixer.css(0, 0.95));
    grad.addColorStop(0.5, mixer.css(2, 1));
    grad.addColorStop(0.8, mixer.css(1, 0.95));
    grad.addColorStop(1, mixer.css(1, 0.0));

    // The envelope keeps the line pinned at the edges and fullest in the centre.
    const sample = (i) => {
      const x = i / (points - 1);
      const env = Math.sin(Math.PI * x) ** 1.4;
      let v;
      if (frame && frame.time) {
        const idx = Math.min(frame.time.length - 1, Math.floor(x * (frame.time.length - 1)));
        v = frame.time[idx] * 3.2;
      } else {
        v = 0;
      }
      const idle = Math.sin(x * 9 + t * 1.6) * 0.06 + Math.sin(x * 4.5 - t * 1.1) * 0.04;
      return (v * 0.9 + idle * (1 - amp * 0.8)) * env * mid * 0.95;
    };

    for (const pass of [{ width: 5, alpha: 0.22 }, { width: 2, alpha: 1 }]) {
      wave.beginPath();
      wave.lineWidth = pass.width * dpr;
      wave.globalAlpha = pass.alpha;
      wave.strokeStyle = grad;
      wave.lineJoin = "round";
      wave.lineCap = "round";
      for (let i = 0; i < points; i++) {
        const x = (i / (points - 1)) * w;
        const y = mid - sample(i);
        if (i === 0) wave.moveTo(x, y);
        else {
          const px = ((i - 1) / (points - 1)) * w;
          const py = mid - sample(i - 1);
          wave.quadraticCurveTo(px, py, (px + x) / 2, (py + y) / 2);
        }
      }
      wave.stroke();
    }
    wave.globalAlpha = 1;
  }

  function drawSpectro(frame, mixer, dt) {
    const w = spectroCanvas.width;
    const h = spectroCanvas.height;
    if (spectroDirty) {
      spectro.clearRect(0, 0, w, h);
      spectroDirty = false;
    }
    // Scroll left at a fixed on-screen speed regardless of frame rate.
    scrollCarry += dt * 60 * dpr;
    const step = Math.floor(scrollCarry);
    if (step < 1) return;
    scrollCarry -= step;
    spectro.globalCompositeOperation = "copy";
    spectro.drawImage(spectroCanvas, -step, 0);
    spectro.globalCompositeOperation = "source-over";
    spectro.clearRect(w - step, 0, step, h);

    const binH = h / SPECTRO_BINS;
    for (let b = 0; b < SPECTRO_BINS; b++) {
      const mag = frame && frame.freq ? frame.freq[b + 1] / 255 : 0;
      if (mag < 0.06) continue;
      const v = Math.min(1, mag * 1.25);
      // Cool colour for quiet bins, warming through the accent to near-white.
      const mixIdx = v > 0.72 ? 2 : v > 0.38 ? 1 : 0;
      spectro.fillStyle = mixer.css(mixIdx, 0.15 + v * 0.85);
      spectro.fillRect(w - step, h - (b + 1) * binH, step, Math.ceil(binH) + 0.5);
    }
  }

  return { draw, resize };
}
