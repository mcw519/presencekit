class MicrophoneBlocks extends AudioWorkletProcessor {
  block = new Float32Array(320);
  used = 0;

  process(inputs) {
    const mono = inputs[0]?.[0];
    if (!mono) return true;
    for (const sample of mono) {
      // Match the PCM16 representation that the original browser sends to PresenceKit.
      this.block[this.used++] = Math.trunc(Math.max(-1, Math.min(1, sample)) * 32767) / 32768;
      if (this.used === this.block.length) {
        this.port.postMessage(this.block, [this.block.buffer]);
        this.block = new Float32Array(320);
        this.used = 0;
      }
    }
    return true;
  }
}

registerProcessor('microphone-blocks', MicrophoneBlocks);

// The AudioContext already runs at 16 kHz. Drop stale queued samples after stalls.
class MonitorAudio extends AudioWorkletProcessor {
  queue = [];
  offset = 0;
  size = 0;
  playing = false;

  constructor() {
    super();
    this.port.onmessage = ({data}) => {
      if (data === null) {
        this.queue.length = 0;
        this.size = this.offset = 0;
        this.playing = false;
        return;
      }
      this.queue.push(data);
      this.size += data.length;
      while (this.size > 1920 && this.queue.length > 1) {
        this.size -= this.queue.shift().length - this.offset;
        this.offset = 0;
      }
    };
  }

  process(_, outputs) {
    const output = outputs[0][0];
    if (this.size >= 640) this.playing = true;
    if (!this.playing) return true;
    let written = 0;
    while (written < output.length && this.queue.length) {
      const block = this.queue[0];
      const count = Math.min(output.length - written, block.length - this.offset);
      output.set(block.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      this.size -= count;
      if (this.offset === block.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (!this.size) this.playing = false;
    return true;
  }
}

registerProcessor('monitor-audio', MonitorAudio);
