const els = {
  mic: document.getElementById('mic'),
  status: document.getElementById('status'),
  transcript: document.getElementById('transcript'),
  meta: document.getElementById('meta'),
  rtf: document.getElementById('rtf'),
  decode: document.getElementById('decode'),
  onset: document.getElementById('onset'),
  gpu: document.getElementById('gpu'),
  gpuWrap: document.getElementById('gpu-wrap'),
  gpuState: document.getElementById('gpu-state'),
};

// A single dropped chunk is normal jitter, not a problem worth shouting about.
// The old threshold was "any drop at all", which put a permanent red
// "dropped 0.02s - falling behind" over the status line for 20ms of audio.
const DROP_WARN_S = 0.25;

function setGpuState(state, detail) {
  if (!els.gpuState) return;
  const label = {
    off: '',
    waiting: 'waiting for first line…',
    active: 'active',
    fallback: 'unavailable — using local',
  }[state] || '';
  els.gpuState.textContent = label;
  els.gpuState.className = 'gpu-state ' + state;
  els.gpuState.title = detail || '';
}

let ws = null;
let ctx = null;
let stream = null;
let running = false;

// seq -> <p> element, so a final can replace the partial it supersedes.
const lines = new Map();

function setStatus(text, kind = '') {
  els.status.textContent = text;
  els.status.className = 'status ' + kind;
}

function renderLine(seq, text, committed, source) {
  let el = lines.get(seq);
  if (!el) {
    el = document.createElement('p');
    el.className = 'line';
    els.transcript.appendChild(el);
    lines.set(seq, el);
  }
  el.textContent = text;
  el.classList.toggle('partial', !committed);
  if (committed && source) {
    // Which engine won arbitration. "vosk" means the offline model was not
    // confident enough to overrule what you already saw.
    const tag = document.createElement('span');
    tag.className = 'src ' + source;
    tag.textContent = source;
    el.appendChild(tag);
  }
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function handle(msg) {
  switch (msg.type) {
    case 'loading':
      setStatus(`loading ${msg.model}…`, 'busy');
      break;
    case 'ready':
      setStatus('listening', 'live');
      els.meta.textContent = `${msg.profile} · ${msg.model} · ${msg.device}`;
      // Only offer the GPU tier if the server actually has it configured;
      // a dead toggle is worse than no toggle.
      els.gpuWrap.hidden = !msg.gpu_available;
      setGpuState('off');
      break;
    case 'config':
      els.gpu.checked = !!msg.gpu;
      // "waiting" not "active": the toggle only states an intent. Whether the
      // GPU actually answers is only known when a final comes back, and a cold
      // container can take ~25s to produce the first one.
      setGpuState(msg.gpu ? 'waiting' : 'off');
      if (msg.error) { setStatus(msg.error, 'warn'); setGpuState('fallback', msg.error); }
      break;
    case 'gpu_status':
      if (msg.ok) {
        setGpuState('active', `${msg.model} · probe ${msg.latency_ms}ms`);
        // A slow probe means the container was cold and this call just warmed
        // it. Worth showing: it is the difference between "broken" and "the
        // first sentence will be slow".
        const cold = msg.latency_ms > 4000 ? ` · warmed in ${(msg.latency_ms / 1000).toFixed(1)}s` : '';
        els.gpuState.textContent = `ready (${msg.model})${cold}`;
      } else {
        setGpuState('fallback', msg.error);
      }
      break;
    case 'partial':
      renderLine(msg.seq, msg.text, false);
      break;
    case 'final':
      // The final carries a new seq; drop the partial that preceded it.
      for (const [seq, el] of lines) {
        if (el.classList.contains('partial')) {
          el.remove();
          lines.delete(seq);
        }
      }
      renderLine(msg.seq, msg.text, true, msg.source);
      break;
    case 'dropped':
      if (msg.seconds >= DROP_WARN_S) {
        setStatus(`dropped ${msg.seconds}s — falling behind`, 'warn');
        // Recover the status line instead of leaving a stale red warning.
        setTimeout(() => { if (running) setStatus('listening', 'live'); }, 4000);
      }
      break;
    case 'stats':
      // Only finals carry a meaningful RTF now: partials come from the
      // streaming engine, which consumes audio incrementally rather than
      // decoding a window, so there is no window to divide by.
      if (msg.kind === 'final') {
        els.rtf.textContent = msg.rtf.toFixed(2);
        els.rtf.className = 'v ' + (msg.rtf < 1 ? 'ok' : 'bad');
        els.decode.textContent = `${msg.decode_ms}ms (${msg.model})`;
        // Ground the GPU indicator in what actually produced the line, not in
        // what the toggle asked for.
        if (els.gpu.checked) {
          if (msg.fell_back) setGpuState('fallback', 'remote call failed; decoded locally');
          else setGpuState(msg.tier === 'gpu' ? 'active' : 'fallback', msg.model);
        }
      }
      if (msg.first_partial_ms) els.onset.textContent = `${msg.first_partial_ms}ms`;
      break;
  }
}

async function start() {
  setStatus('requesting mic…', 'busy');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    setStatus('microphone denied', 'bad');
    return;
  }

  ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule('/static/worklet.js');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { if (running) stop(); setStatus('disconnected', 'bad'); };
  ws.onerror = () => setStatus('connection failed', 'bad');

  await new Promise((res, rej) => { ws.onopen = res; setTimeout(rej, 5000); })
    .catch(() => setStatus('connection timed out', 'bad'));

  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'capture');
  node.port.onmessage = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const f32 = e.data;
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(i16.buffer);
  };
  src.connect(node);
  // The worklet emits no audio; connecting to the destination only keeps the
  // graph alive in browsers that garbage-collect dangling nodes.
  node.connect(ctx.destination);

  running = true;
  els.mic.textContent = 'Stop';
  els.mic.classList.add('on');
}

function stop() {
  running = false;
  els.mic.textContent = 'Start listening';
  els.mic.classList.remove('on');
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (ctx) ctx.close();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  ws = ctx = stream = null;
  setStatus('idle');
}

els.mic.addEventListener('click', () => (running ? stop() : start()));

// GPU tier is opt-in per session: an idle page must never spend credits.
els.gpu.addEventListener('change', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config', gpu: els.gpu.checked }));
    setGpuState(els.gpu.checked ? 'waiting' : 'off');
  }
});
