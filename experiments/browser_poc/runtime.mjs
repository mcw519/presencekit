import {BLOCK, RATE, defaults} from './audio-core.mjs';
import {decodeWav} from './vendor/audio.js';

export const FRAME_SECONDS = 0.5;
const blockSeconds = BLOCK / RATE;

class WorkerLink {
  id = 0;
  pending = new Map();

  constructor(path) {
    this.worker = new Worker(new URL(path, import.meta.url), {type: 'module'});
    this.worker.onmessage = ({data}) => {
      const request = this.pending.get(data.id);
      if (!request) return;
      this.pending.delete(data.id);
      if (data.error) request.reject(Error(data.error));
      else request.resolve(data);
    };
    this.worker.onerror = event => this.close(Error(event.message));
  }

  call(type, data, transfer = []) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      try {
        this.worker.postMessage({id, type, ...data}, transfer);
      } catch (error) {
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  close(reason = Error('Stopped')) {
    this.worker.terminate();
    for (const {reject} of this.pending.values()) reject(reason);
    this.pending.clear();
  }
}

export class LocalRuntime {
  audio = new WorkerLink('./audio-worker.mjs');
  vision = new WorkerLink('./vision-worker.mjs');
  snapshot = null;

  init(settings) {
    const config = {...defaults, ...settings};
    return Promise.all([this.audio.call('init', {config}), this.vision.call('init', {config})]);
  }

  async frame(bitmap, time) {
    const result = await this.vision.call('frame', {bitmap, time}, [bitmap]);
    this.snapshot = result.snapshot;
    return result;
  }

  block(samples, time) {
    return this.audio.call('block', {samples, time, snapshot: this.snapshot}, [samples.buffer]);
  }

  close() {
    this.audio.close();
    this.vision.close();
  }
}

export async function replayFixture(url) {
  const base = new URL(url, location.href);
  const fetchFile = async path => {
    const response = await fetch(new URL(path, base));
    if (!response.ok) throw Error('Cannot load ' + path);
    return response;
  };
  const [fixtureResponse, audioResponse] = await Promise.all([fetchFile('fixture.json'), fetchFile('raw.wav')]);
  const fixture = await fixtureResponse.json();
  const {samples, sampleRate} = decodeWav(await audioResponse.arrayBuffer());
  if (sampleRate !== RATE) throw Error('Replay needs 16 kHz');
  const frames = await Promise.all(fixture.frames.map(async frame => ({
    time: frame.time, bitmap: await createImageBitmap(await (await fetchFile(frame.file)).blob()),
  })));
  const runtime = new LocalRuntime();
  const out = new Float32Array(samples.length).fill(NaN), vi = out.slice();
  const speech = new Uint8Array(samples.length), allowed = speech.slice();
  const audioMs = [], latencyMs = [], videoMs = [], trace = [], views = [], pending = [];
  let audioBacklog = 0, maxAudioBacklog = 0, error;
  let start;
  const schedule = (time, run) => {
    const delay = Math.max(0, start + time * 1000 - performance.now());
    pending.push(new Promise(resolve => setTimeout(resolve, delay)).then(run).catch(err => { error ??= err; }));
  };

  try {
    await runtime.init(fixture.settings);
    start = performance.now();
    let videoQueue = Promise.resolve();
    for (const {bitmap, time} of frames) {
      schedule(time, () => {
        videoQueue = videoQueue.then(async () => {
          const {snapshot, computeMs} = await runtime.frame(bitmap, time);
          videoMs.push(computeMs);
          views.push(snapshot);
        });
        return videoQueue;
      });
    }
    for (let offset = 0; offset < samples.length + 5 * BLOCK; offset += BLOCK) {
      const block = new Float32Array(BLOCK);
      block.set(samples.subarray(offset, offset + BLOCK));
      const time = (offset + BLOCK) / RATE;
      schedule(time, async () => {
        audioBacklog++;
        maxAudioBacklog = Math.max(maxAudioBacklog, audioBacklog);
        const r = await runtime.block(block, time);
        audioBacklog--;
        latencyMs.push(performance.now() - (start + time * 1000));
        audioMs.push(r.computeMs);
        const left = Math.max(0, r.start), right = Math.min(samples.length, r.start + r.out.length);
        if (right <= left) return;
        const from = left - r.start, to = right - r.start;
        for (let i = left; i < right; i++) {
          if (Number.isFinite(out[i]) || r.raw[from + i - left] !== samples[i]) throw Error('Audio timestamp mismatch');
        }
        out.set(r.out.subarray(from, to), left);
        vi.set(r.vi.subarray(from, to), left);
        speech.fill(Number(r.speech), left, right);
        allowed.fill(Number(r.allowed), left, right);
        trace.push({left, right, decision: r.decision, view: r.view});
      });
    }
    await Promise.all(pending);
    if (error) throw error;
    if (!out.every(Number.isFinite) || videoMs.length !== frames.length) throw Error('Incomplete 2 fps replay');
    return {out, vi, speech, allowed, trace, views, audioMs, latencyMs, videoMs, maxAudioBacklog};
  } finally {
    runtime.close();
    frames.forEach(frame => frame.bitmap.close());
  }
}

export function wav(samples) {
  const data = new ArrayBuffer(44 + samples.length * 2), view = new DataView(data);
  for (const [offset, tag] of [[0, 'RIFF'], [8, 'WAVE'], [12, 'fmt '], [36, 'data']]) {
    Array.from(tag).forEach((letter, i) => view.setUint8(offset + i, letter.charCodeAt(0)));
  }
  view.setUint32(4, data.byteLength - 8, true);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((value, i) => view.setInt16(44 + 2 * i, Math.round(value * 32768), true));
  return new Blob([data], {type: 'audio/wav'});
}
