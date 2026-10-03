// Capture worklet: mono float32 in at the context rate, 16kHz frames out.
//
// Chrome will usually honour an AudioContext({sampleRate: 16000}) request, in
// which case the resampler below is a no-op passthrough. It is kept because
// Safari and most Firefox builds ignore the hint and hand you 44.1k or 48k,
// and silently sending those samples as if they were 16k makes every word
// come out slurred and slow.

const TARGET_RATE = 16000;
const FRAME = 320; // 20ms at 16kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.pos = 0;
    this.out = new Float32Array(FRAME);
    this.filled = 0;
    this.tail = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    // Prepend whatever we could not consume last block so interpolation does
    // not restart at a block boundary and click.
    let buf = input[0];
    if (this.tail.length) {
      const merged = new Float32Array(this.tail.length + buf.length);
      merged.set(this.tail, 0);
      merged.set(buf, this.tail.length);
      buf = merged;
    }

    while (this.pos + 1 < buf.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      // Linear interpolation is fine for speech at these rates.
      this.out[this.filled++] = buf[i] * (1 - frac) + buf[i + 1] * frac;

      if (this.filled === FRAME) {
        this.port.postMessage(this.out.slice(0));
        this.filled = 0;
      }
      this.pos += this.ratio;
    }

    const consumed = Math.floor(this.pos);
    this.tail = buf.slice(consumed);
    this.pos -= consumed;
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);
