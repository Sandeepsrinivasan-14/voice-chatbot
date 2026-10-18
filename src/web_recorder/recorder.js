const MODULATION_META = {
  normal:    { color: '#60a5fa', text: 'Say it normally, like regular conversation.' },
  whispered: { color: '#a78bfa', text: 'Whisper it as quietly as you comfortably can.' },
  shouted:   { color: '#f87171', text: 'Say it loudly, as if shouting across a room.' },
  slow:      { color: '#2dd4bf', text: 'Say it very slowly, stretching the word out.' },
  fast:      { color: '#fbbf24', text: 'Say it very fast, rushed.' },
};
const MAX_RECORD_MS = 8000;
const MIN_RECORD_MS = 300;

let combos = [];
let idx = 0;
let stream = null;
let mediaRecorder = null;
let chunks = [];
let currentBlob = null;
let recordState = 'idle'; // idle | countdown | recording | reviewing
let recordStartTs = 0;
let autoStopTimer = null;

const el = (id) => document.getElementById(id);
const startScreen = el('screen-start');
const recordScreen = el('screen-record');
const doneScreen = el('screen-done');
const progressTrack = el('progress-track');

function showToast(msg, isError) {
  const t = document.createElement('div');
  t.className = 'toast' + (isError ? ' error' : '');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3200);
}

function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg',
  ];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return '';
}

async function loadPlan(words) {
  const url = '/api/plan' + (words ? ('?words=' + encodeURIComponent(words)) : '');
  const res = await fetch(url);
  const data = await res.json();
  combos = data.combos;
  renderWordChips(data.words);
  const firstNotDone = combos.findIndex((c) => !c.done);
  idx = firstNotDone === -1 ? 0 : firstNotDone;
  const doneCount = combos.filter((c) => c.done).length;
  if (doneCount > 0 && doneCount < combos.length) {
    el('resume-hint').hidden = false;
    el('resume-hint').textContent = `Found ${doneCount}/${combos.length} clips already recorded -- you'll resume from where you left off.`;
  }
  buildProgressTrack();
}

function renderWordChips(words) {
  const row = el('word-chip-row');
  row.innerHTML = '';
  words.forEach((w) => {
    const chip = document.createElement('span');
    chip.className = 'word-chip';
    chip.textContent = w;
    row.appendChild(chip);
  });
}

function buildProgressTrack() {
  // Note: does NOT reveal the track -- it's built as soon as the plan
  // loads (including the page-load call that only checks for a resume
  // hint) but stays hidden until initMicAndBegin() actually shows the
  // record screen.
  progressTrack.innerHTML = '';
  const byWord = {};
  combos.forEach((c, i) => {
    (byWord[c.word] = byWord[c.word] || []).push(i);
  });
  Object.values(byWord).forEach((indices) => {
    const group = document.createElement('div');
    group.className = 'progress-group';
    indices.forEach((i) => {
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.dataset.idx = i;
      group.appendChild(dot);
    });
    progressTrack.appendChild(group);
  });
  refreshProgressDots();
}

function refreshProgressDots() {
  progressTrack.querySelectorAll('.dot').forEach((dot) => {
    const i = Number(dot.dataset.idx);
    dot.classList.toggle('done', !!combos[i].done);
    dot.classList.toggle('current', i === idx);
  });
}

async function initMicAndBegin() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showToast('Microphone access denied or unavailable: ' + err.message, true);
    return;
  }
  startScreen.hidden = true;
  recordScreen.hidden = false;
  progressTrack.hidden = false;
  renderCurrentCombo();
}

function renderCurrentCombo() {
  if (idx >= combos.length) { showDone(); return; }
  const combo = combos[idx];
  const meta = MODULATION_META[combo.modulation];
  el('word-counter').textContent = `${idx + 1} / ${combos.length}`;
  el('word-display').textContent = combo.word.toUpperCase();
  el('badge-modulation').textContent = combo.modulation;
  el('badge-modulation').style.color = meta.color;
  el('badge-modulation').style.background = meta.color + '33';
  el('instruction').textContent = meta.text;
  document.documentElement.style.setProperty('--accent', meta.color);
  document.documentElement.style.setProperty('--accent-dim', meta.color + '33');
  el('playback-controls').hidden = true;
  el('record-status').textContent = 'Tap to record';
  recordState = 'idle';
  currentBlob = null;
  refreshProgressDots();
}

function setRecordButtonState(state) {
  const btn = el('btn-record');
  btn.classList.remove('recording', 'countdown');
  if (state === 'recording') btn.classList.add('recording');
  if (state === 'countdown') btn.classList.add('countdown');
}

function startCountdown() {
  recordState = 'countdown';
  setRecordButtonState('countdown');
  const numEl = el('countdown-num');
  numEl.hidden = false;
  let n = 3;
  numEl.textContent = n;
  el('record-status').textContent = 'Get ready...';
  const iv = setInterval(() => {
    n -= 1;
    if (n <= 0) {
      clearInterval(iv);
      numEl.hidden = true;
      beginRecording();
    } else {
      numEl.textContent = n;
    }
  }, 550);
}

function beginRecording() {
  chunks = [];
  const mimeType = pickMimeType();
  try {
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  } catch (err) {
    showToast('Recording failed to start: ' + err.message, true);
    recordState = 'idle';
    setRecordButtonState('idle');
    return;
  }
  mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStopped;
  mediaRecorder.start();
  recordState = 'recording';
  recordStartTs = Date.now();
  setRecordButtonState('recording');
  el('record-status').textContent = 'Recording... tap to stop';
  autoStopTimer = setTimeout(() => stopRecording(), MAX_RECORD_MS);
}

function stopRecording() {
  if (autoStopTimer) { clearTimeout(autoStopTimer); autoStopTimer = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
}

function onRecordingStopped() {
  const elapsed = Date.now() - recordStartTs;
  const mimeType = mediaRecorder.mimeType || 'audio/webm';
  const blob = new Blob(chunks, { type: mimeType });
  setRecordButtonState('idle');
  if (elapsed < MIN_RECORD_MS || blob.size < 500) {
    showToast('That was too short -- try again.', true);
    recordState = 'idle';
    el('record-status').textContent = 'Tap to record';
    return;
  }
  currentBlob = blob;
  recordState = 'reviewing';
  el('record-status').textContent = 'Recorded! Review below.';
  el('playback-controls').hidden = false;
}

function playCurrentBlob() {
  if (!currentBlob) return;
  const url = URL.createObjectURL(currentBlob);
  const audio = new Audio(url);
  audio.play();
  audio.onended = () => URL.revokeObjectURL(url);
}

function retake() {
  currentBlob = null;
  recordState = 'idle';
  el('playback-controls').hidden = true;
  el('record-status').textContent = 'Tap to record';
}

async function keepAndNext() {
  if (!currentBlob) return;
  const combo = combos[idx];
  const btn = el('btn-keep');
  btn.disabled = true;
  btn.textContent = 'Saving...';
  try {
    const form = new FormData();
    const ext = (currentBlob.type.includes('ogg')) ? 'ogg' : 'webm';
    form.append('audio', currentBlob, `clip.${ext}`);
    form.append('word', combo.word);
    form.append('modulation', combo.modulation);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || ('HTTP ' + res.status));
    combo.done = true;
    idx += 1;
    renderCurrentCombo();
  } catch (err) {
    showToast('Save failed: ' + err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Keep & Next →';
  }
}

function showDone() {
  recordScreen.hidden = true;
  doneScreen.hidden = false;
  progressTrack.hidden = true;
  if (stream) stream.getTracks().forEach((t) => t.stop());
}

el('btn-start').addEventListener('click', async () => {
  const btn = el('btn-start');
  btn.disabled = true;
  try {
    // Re-load the plan with whatever word list is currently in the input
    // (it may differ from the default shown on page load) BEFORE asking
    // for the mic, so `combos` reflects the user's actual choice.
    await loadPlan(el('words-input').value);
    await initMicAndBegin();
  } finally {
    btn.disabled = false;
  }
});

el('btn-record').addEventListener('click', () => {
  if (recordState === 'idle') startCountdown();
  else if (recordState === 'recording') stopRecording();
});

el('btn-play').addEventListener('click', playCurrentBlob);
el('btn-retake').addEventListener('click', retake);
el('btn-keep').addEventListener('click', keepAndNext);

// Load the plan once up front (default word list) purely so the start
// screen can show a "resume from where you left off" hint immediately.
loadPlan();
