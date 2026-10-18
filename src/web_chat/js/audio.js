/* The one AudioContext: microphone capture + level analysis, and routing of
 * assistant speech through an analyser so the orb and waveform can react to
 * what is actually being played. */

import { sleep } from "./ui.js";

let ctx = null;
let stream = null;
let micAnalyser = null;
let micSource = null;
let outAnalyser = null;

const FFT = 1024;
const timeBuf = new Float32Array(FFT);
const outTimeBuf = new Float32Array(FFT);
const freqBuf = new Uint8Array(FFT / 2);
const outFreqBuf = new Uint8Array(FFT / 2);

export function getContext() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = new AC();
  }
  return ctx;
}

/* Browsers keep a fresh context suspended until a user gesture. Resolve to
 * whether it is actually running, but never hang the caller waiting. */
export async function resumeContext() {
  let c;
  try {
    c = getContext();
  } catch (err) {
    return false;
  }
  if (c.state === "suspended") {
    try {
      await Promise.race([c.resume(), sleep(400)]);
    } catch (err) {
      /* fall through to the state check */
    }
  }
  return c.state === "running";
}

export async function ensureMic() {
  if (stream && stream.getTracks().some((t) => t.readyState === "live")) return stream;
  // Explicit echo-cancellation / noise-suppression matter more in hands-free
  // mode, where listening restarts moments after this machine's speakers
  // stop: without them the mic risks transcribing the assistant's own voice.
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  micAnalyser = null; // belongs to the previous stream, if any
  micSource = null;
  return stream;
}

export function getStream() {
  return stream;
}

export function ensureMicAnalyser() {
  if (micAnalyser) return micAnalyser;
  const c = getContext();
  micSource = c.createMediaStreamSource(stream);
  micAnalyser = c.createAnalyser();
  micAnalyser.fftSize = FFT;
  micAnalyser.smoothingTimeConstant = 0.55;
  micSource.connect(micAnalyser); // deliberately NOT connected to the speakers
  return micAnalyser;
}

function ensureOutAnalyser() {
  if (outAnalyser) return outAnalyser;
  const c = getContext();
  outAnalyser = c.createAnalyser();
  outAnalyser.fftSize = FFT;
  outAnalyser.smoothingTimeConstant = 0.55;
  outAnalyser.connect(c.destination);
  return outAnalyser;
}

/* Root-mean-square of the latest mic samples (the voice-activity signal). */
export function readMicRms() {
  micAnalyser.getFloatTimeDomainData(timeBuf);
  let sum = 0;
  for (let i = 0; i < timeBuf.length; i++) sum += timeBuf[i] * timeBuf[i];
  return Math.sqrt(sum / timeBuf.length);
}

/* One frame of data for the visualisers. `kind` is "mic" or "out". */
export function sampleFrame(kind) {
  const analyser = kind === "mic" ? micAnalyser : outAnalyser;
  if (!analyser) return null;
  const time = kind === "mic" ? timeBuf : outTimeBuf;
  const freq = kind === "mic" ? freqBuf : outFreqBuf;
  analyser.getFloatTimeDomainData(time);
  analyser.getByteFrequencyData(freq);
  let sum = 0;
  for (let i = 0; i < time.length; i++) sum += time[i] * time[i];
  return { rms: Math.sqrt(sum / time.length), time, freq };
}

/* Creates the <audio> element for one reply chunk, routed through the output
 * analyser when the context is running. `routed` tells the visualiser
 * whether real data will be available or it should fall back to a
 * procedural pulse. */
export function prepareAudio(blob, { rate = 1, volume = 1 } = {}) {
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.playbackRate = rate;
  audio.volume = volume;
  let routed = false;
  try {
    if (ctx && ctx.state === "running") {
      const source = ctx.createMediaElementSource(audio);
      source.connect(ensureOutAnalyser());
      routed = true;
    }
  } catch (err) {
    routed = false;
  }
  return { audio, url, routed };
}

export function playChime() {
  return new Promise((resolve) => {
    try {
      const c = getContext();
      const osc = c.createOscillator();
      const gain = c.createGain();
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.15, c.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + 0.15);
      osc.connect(gain).connect(c.destination);
      osc.start();
      osc.stop(c.currentTime + 0.15);
      osc.onended = resolve;
    } catch (err) {
      resolve(); // the chime is a nicety, never block on it
    }
  });
}
