import {fft} from './vendor/runtime.js';
import {RATE} from './audio-core.mjs';
import {localize} from './i18n.mjs';

const STEP = 640; // 40 ms; each signal column shares the same aligned samples.
const COLUMNS = 125;
const FFT_SIZE = 512;
const BINS = 64;
const hann = Float64Array.from({length: FFT_SIZE}, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_SIZE));
const normalization = 2 / hann.reduce((sum, value) => sum + value);
const TONES = {
  agree: ['keptSpeech', 'speech'],
  false_reject: ['keptIsolation', 'neutral'],
  false_accept: ['cutFar', 'cut'],
  leak: ['cutEmpty', 'cut'],
  silence: ['quiet', 'neutral'],
  acoustic_only: ['audioOnly', 'neutral'],
};
const COLORS = {near: '#a6eddc', far: '#ffd184', speech: '#58caae', cut: '#df7889', recover: '#f4bb65'};
const get = id => document.getElementById(id);

function surface(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  const scale = Math.min(2, devicePixelRatio);
  const w = Math.round(width * scale), h = Math.round(height * scale);
  if (!w || !h) return null;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return {ctx: canvas.getContext('2d'), w, h, scale};
}

function clear(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function push(history, item, size) {
  history.push(item);
  if (history.length > size) history.shift();
}

function spectrum(block) {
  const re = Float64Array.from(hann, (value, i) => value * block[STEP - FFT_SIZE + i]);
  const im = new Float64Array(FFT_SIZE);
  fft(re, im);
  return Uint8Array.from({length: BINS}, (_, i) => {
    const bin = i * 2;
    const mag = Math.max(Math.hypot(re[bin], im[bin]), Math.hypot(re[bin + 1], im[bin + 1])) * normalization;
    const db = 20 * Math.log10(Math.max(1e-6, mag));
    return Math.round(Math.max(0, Math.min(255, (db + 85) * (255 / 65))));
  });
}

class Lane {
  block = new Float32Array(STEP);
  history = [];

  constructor(name, color) {
    this.wave = get('wave-' + name);
    this.spec = get('spec-' + name);
    this.db = get('db-' + name);
    this.color = color;
    this.pixels = document.createElement('canvas');
    this.pixels.width = COLUMNS;
    this.pixels.height = BINS;
  }

  complete() {
    let low = 0, high = 0, power = 0;
    for (const value of this.block) {
      low = Math.min(low, value);
      high = Math.max(high, value);
      power += value * value;
    }
    push(this.history, {low, high, power: power / STEP, spectrum: spectrum(this.block)}, COLUMNS);
  }

  level() {
    if (!this.history.length) return null;
    const recent = this.history.slice(-6);
    const rmsPower = recent.reduce((sum, entry) => sum + entry.power, 0) / recent.length;
    return 10 * Math.log10(Math.max(1e-10, rmsPower));
  }

  reset() {
    this.history = [];
    this.db.textContent = '—';
    clear(this.wave);
    clear(this.spec);
  }

  draw() {
    const wave = surface(this.wave);
    if (wave) {
      const {ctx, w, h} = wave;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#ffffff20';
      ctx.fillRect(0, Math.floor(h / 2), w, 1);
      const dx = w / COLUMNS;
      ctx.fillStyle = 'rgb(' + this.color.join(' ') + ')';
      this.history.forEach(({low, high}, i) => {
        const top = Math.round((0.5 - Math.min(0.35, high) / 0.7) * h);
        const bottom = Math.round((0.5 - Math.max(-0.35, low) / 0.7) * h);
        if (bottom > top) ctx.fillRect((COLUMNS - this.history.length + i) * dx, top, Math.max(1, dx - 0.5), bottom - top);
      });
    }
    const spec = surface(this.spec);
    if (!spec) return;
    const source = this.pixels.getContext('2d');
    const pixels = source.createImageData(COLUMNS, BINS);
    this.history.forEach((entry, i) => {
      const x = COLUMNS - this.history.length + i;
      entry.spectrum.forEach((value, bin) => {
        const index = 4 * ((BINS - bin - 1) * COLUMNS + x);
        for (let c = 0; c < 3; c++) pixels.data[index + c] = this.color[c];
        pixels.data[index + 3] = value;
      });
    });
    source.putImageData(pixels, 0, 0);
    spec.ctx.clearRect(0, 0, spec.w, spec.h);
    spec.ctx.imageSmoothingEnabled = false;
    spec.ctx.drawImage(this.pixels, 0, 0, spec.w, spec.h);
  }
}

export class MonitorView {
  used = 0;
  raw = new Lane('raw', [200, 246, 249]);
  out = new Lane('out', [189, 187, 255]);
  decisions = [];
  tracks = [];

  constructor(camera, nearRange) {
    this.camera = camera;
    this.nearRange = nearRange;
    this.overlay = get('overlay');
    this.timeline = get('timeline');
    this.resize = new ResizeObserver(() => this.drawOverlay());
    this.resize.observe(this.overlay);
  }

  reset() {
    this.raw.reset();
    this.out.reset();
    this.used = 0;
    this.decisions = [];
    this.last = null;
    this.showFaces(null);
    clear(this.timeline);
    for (const id of ['speech', 'reduction', 'near-count', 'far-count']) {
      const element = get(id);
      delete element.dataset.i18n;
      element.textContent = '—';
    }
    localize(get('decision'), 'standby');
    get('decision').dataset.tone = 'neutral';
  }

  collect(result) {
    this.last = result;
    const from = Math.max(0, -result.start);
    for (let i = from; i < result.out.length; i++) {
      this.raw.block[this.used] = result.raw[i];
      this.out.block[this.used] = result.out[i];
      if (++this.used === STEP) {
        this.raw.complete();
        this.out.complete();
        this.used = 0;
      }
    }
    if (result.out.length > from) {
      push(this.decisions, {
        speech: result.speech, allowed: result.allowed, decision: result.decision,
        start: (result.start + from) / RATE, end: (result.start + result.out.length) / RATE,
      }, 510);
    }
  }

  drawSignals() {
    this.raw.draw();
    this.out.draw();
    const rawLevel = this.raw.level(), outLevel = this.out.level();
    if (rawLevel !== null) {
      this.raw.db.textContent = rawLevel.toFixed(0) + ' dB';
      this.out.db.textContent = outLevel.toFixed(0) + ' dB';
      const difference = Math.round((rawLevel - outLevel) * 10) / 10;
      get('reduction').textContent = difference.toFixed(1) + ' dB';
    }
    if (this.last) {
      const [label, tone] = this.last.allowed ? ['recovered', 'recover'] : TONES[this.last.decision];
      localize(get('decision'), label);
      get('decision').dataset.tone = tone;
      localize(get('speech'), this.last.speech ? 'speaking' : 'quiet');
    }
    this.drawTimeline();
  }

  drawTimeline() {
    const canvas = surface(this.timeline);
    if (!canvas) return;
    const {ctx, w, h} = canvas;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff20';
    ctx.fillRect(0, Math.floor(h / 2), w, 1);
    const end = this.decisions.at(-1)?.end ?? 0;
    for (const block of this.decisions) {
      const left = Math.round(w * (1 + (block.start - end) / 10));
      const right = Math.round(w * (1 + (block.end - end) / 10));
      if (block.speech) {
        ctx.fillStyle = COLORS.speech;
        ctx.fillRect(left, 4, Math.max(1, right - left), h / 2 - 8);
      }
      if (block.allowed || ['leak', 'false_accept'].includes(block.decision)) {
        ctx.fillStyle = block.allowed ? COLORS.recover : COLORS.cut;
        ctx.fillRect(left, h / 2 + 4, Math.max(1, right - left), h / 2 - 8);
      }
    }
  }

  showFaces(snapshot) {
    this.tracks = snapshot?.tracks ?? [];
    const near = this.tracks.filter(track => track.distance <= this.nearRange).length;
    get('near-count').textContent = snapshot ? String(near) : '—';
    get('far-count').textContent = snapshot ? String(this.tracks.length - near) : '—';
    get('people').replaceChildren(...this.tracks.map((track, index) => {
      const li = document.createElement('li');
      const label = document.createElement('span');
      const value = document.createElement('small');
      const isNear = track.distance <= this.nearRange;
      li.className = 'person';
      label.className = isNear ? 'near' : 'far';
      localize(label, isNear ? 'trackNear' : 'trackFar', {index: index + 1});
      value.textContent = track.distance.toFixed(2) + ' m';
      li.append(label, value);
      return li;
    }));
    this.drawOverlay();
  }

  drawOverlay() {
    const canvas = surface(this.overlay);
    if (!canvas) return;
    const {ctx, w, h, scale} = canvas;
    ctx.clearRect(0, 0, w, h);
    const {videoWidth, videoHeight} = this.camera;
    if (!videoWidth || !videoHeight) return;
    const frameScale = Math.min(w / videoWidth, h / videoHeight);
    const fw = videoWidth * frameScale, fh = videoHeight * frameScale;
    const ox = (w - fw) / 2, oy = (h - fh) / 2;
    ctx.lineWidth = 2 * scale;
    ctx.font = `${12 * scale}px ui-monospace, monospace`;
    ctx.textBaseline = 'middle';
    this.tracks.forEach((track, index) => {
      const [x, y, right, bottom] = track.box;
      const left = ox + x * fw, top = oy + y * fh;
      const tint = track.distance <= this.nearRange ? COLORS.near : COLORS.far;
      ctx.strokeStyle = tint;
      ctx.strokeRect(left, top, (right - x) * fw, (bottom - y) * fh);
      const text = `#${index + 1} · ${track.distance.toFixed(2)} m`;
      const labelTop = Math.max(oy, top - 20 * scale);
      ctx.fillStyle = '#131629';
      ctx.fillRect(left, labelTop, ctx.measureText(text).width + 12 * scale, 20 * scale);
      ctx.fillStyle = tint;
      ctx.fillText(text, left + 6 * scale, labelTop + 10 * scale);
    });
  }
}
