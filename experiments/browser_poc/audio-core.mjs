// PureSound supplies the streaming output. These rules mirror PR #9's audio mode.
export const RATE = 16000;
export const HOP = 160;
export const BLOCK = 320;

function power(samples) {
  return samples.reduce((sum, value) => sum + value * value, 0) / samples.length;
}

function bandPowers(samples, highHz) {
  const n = samples.length;
  const win = samples.map((x, i) => x * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1))));
  let low = 0, high = 0, windowPower = 0;
  for (let i = 0; i < n; i++) windowPower += (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1))) ** 2;
  for (let k = Math.ceil(150 * n / RATE); k * RATE / n < highHz; k++) {
    const step = 2 * Math.PI * k / n;
    const cos = Math.cos(step), sin = Math.sin(step);
    let r = 1, q = 0, re = 0, im = 0;
    for (const x of win) {
      re += x * r;
      im += x * q;
      [r, q] = [cos * r - sin * q, cos * q + sin * r];
    }
    const bin = re * re + im * im;
    if (k * RATE / n < 1800) low += bin;
    else high += bin;
  }
  return {low, high, normalizedLow: low / (n * windowPower)};
}

class Onset {
  recent = [];
  onset = null;
  qualified = false;
  start = null;
  history = null;

  step(hop, end, rising, speech) {
    this.recent.push(hop);
    if (this.recent.length > 26) this.recent.shift();
    if (rising) {
      this.onset = null;
      this.qualified = false;
      this.start = end - 6 * HOP;
      this.history = this.recent.length === 26 ? this.join(this.recent.slice(0, 20)) : null;
    } else if (!speech) {
      this.qualified = false;
      this.history = null;
    }
    if (!this.history || end > this.start + 2240) {
      this.history = null;
      return;
    }
    const attack = this.join(this.recent.slice(-6));
    const rise = 10 * Math.log10(Math.max(power(attack), 1e-12) / Math.max(power(this.history), 1e-12));
    if (rise >= 8) {
      const spec = bandPowers(attack, 6000);
      const tilt = 10 * Math.log10(Math.max(spec.high, 1e-12) / Math.max(spec.low, 1e-12));
      if (tilt <= -12 || (rise >= 18 && tilt <= 0 && 10 * Math.log10(
        Math.max(spec.normalizedLow, 1e-18) / Math.max(bandPowers(this.history, 1800).normalizedLow, 1e-18)
      ) >= 15)) {
        this.onset = end - 6 * HOP;
        this.qualified = true;
        this.history = null;
      }
    } else if (rising) {
      this.history = null;
    }
  }

  join(hops) {
    const result = new Float64Array(hops.length * HOP);
    hops.forEach((hop, i) => result.set(hop, i * HOP));
    return result;
  }

  allows(start, end) {
    return this.onset !== null && end > this.onset && start < this.onset + 2560;
  }
}

class Vad {
  floor = -70;
  speech = false;
  run = 0;
  quiet = 0;

  constructor(cfg) {
    this.cfg = cfg;
  }

  step(samples) {
    const level = 10 * Math.log10(power(samples) + 1e-10);
    this.floor = Math.min(level, this.floor + 0.03);
    const active = level > Math.max(this.cfg.clean_vad_min_db, this.floor + this.cfg.clean_vad_margin_db);
    if (active) {
      this.quiet = 0;
      this.run++;
      if (this.run * HOP / RATE >= 0.06) this.speech = true;
    } else {
      this.run = 0;
      if (++this.quiet * HOP / RATE >= this.cfg.clean_vad_hang) this.speech = false;
    }
    return active;
  }
}

class CrossCheck {
  lastPerson = -Infinity;
  lastNear = -Infinity;
  lastFar = -Infinity;
  lastAnyFar = -Infinity;

  view(snapshot, now, cfg) {
    if (!snapshot || now - snapshot.time > cfg.xcheck_stale || snapshot.uncertain) {
      return {usable: false};
    }
    const distOk = cfg.distance_enabled && !cfg.xcheck_require_calib;
    const tracks = snapshot.tracks;
    const near = tracks.some(t => !distOk || t.distance === null || t.distance <= cfg.near_range);
    const far = distOk && tracks.some(t => t.distance > cfg.near_range);
    if (tracks.length) this.lastPerson = now;
    if (near) this.lastNear = now;
    if (far) this.lastAnyFar = now;
    if (far && !near) this.lastFar = now;
    const recentlyNear = now - this.lastNear <= cfg.xcheck_hold;
    const holdFar = cfg.xcheck_hold > 0 ? Math.max(cfg.xcheck_hold, cfg.xcheck_stale) : 0;
    return {
      usable: true, distOk, near: recentlyNear,
      person: now - this.lastPerson <= cfg.xcheck_hold,
      allFar: !recentlyNear && now - this.lastFar <= cfg.xcheck_hold,
      anyFar: distOk && now - this.lastAnyFar <= holdFar,
    };
  }
}

export const defaults = Object.freeze({
  near_range: 1.3, distance_enabled: true, distance_calib: 1, distance_ema: 0.3,
  camera_hfov: 60, face_width_m: 0.15, face_roi: [0.2, 0.05, 0.8, 0.95],
  crop_margin: 0.1, roi_crop: true, max_width: 640, min_conf: 0.4, coast_seconds: 2,
  xcheck_enabled: true, xcheck_hold: 0.75, xcheck_stale: 2, xcheck_require_calib: false,
  xcheck_floor_db: -40, xcheck_attack: 0.01, xcheck_release: 0.05, xcheck_recover_mix: 0.3,
  clean_vad_min_db: -50, clean_vad_margin_db: 10, clean_vad_hang: 0.3,
});

export class PresenceAudio {
  onset = new Onset();
  crossCheck = new CrossCheck();
  inputSamples = 0;
  emitted = 0;
  activeAt = -RATE;
  agreed = false;
  dry = new Float32Array(0);
  dryStart = 0;
  mix = 1;
  gain = 1;
  cleanTail = new Float32Array(0);

  constructor(runtime, manifest, config) {
    this.runtime = runtime;
    this.delay = manifest.streaming_delay_frames * manifest.hop_length;
    this.cfg = config;
    this.rawVad = new Vad(config);
    this.cleanVad = new Vad(config);
  }

  decide(view, start) {
    const cfg = this.cfg;
    const clean = this.cleanVad.speech;
    const floor = cfg.xcheck_floor_db <= -60 ? 0 : 10 ** (cfg.xcheck_floor_db / 20);
    if (!cfg.xcheck_enabled || !view.usable) return ['acoustic_only', clean, 1, 1, false];
    if (clean && !view.person) return ['leak', false, 1, floor, false];
    if (clean && view.allFar) return ['false_accept', false, 1, floor, false];
    if (clean) {
      this.agreed = true;
      return ['agree', true, 1, 1, false];
    }
    if (!this.rawVad.speech || !view.near) return ['silence', false, 1, 1, false];
    const solo = view.distOk && !view.anyFar && (this.onset.qualified || this.agreed)
      && this.inputSamples - this.activeAt <= 2400;
    const allowed = this.onset.allows(start, start + BLOCK) || solo;
    return ['false_reject', allowed, allowed ? cfg.xcheck_recover_mix : 1, 1, allowed];
  }

  async step(x, snapshot, now) {
    if (x.length !== BLOCK) throw Error('Expected 20 ms of mono 16 kHz input');
    for (let i = 0; i < BLOCK; i += HOP) {
      const hop = x.subarray(i, i + HOP);
      const wasSpeaking = this.rawVad.speech;
      const active = this.rawVad.step(hop);
      const end = this.inputSamples + i + HOP;
      if (active) this.activeAt = end;
      if (!this.rawVad.speech) this.agreed = false;
      this.onset.step(hop, end, this.rawVad.speech && !wasSpeaking, this.rawVad.speech);
    }
    this.inputSamples += BLOCK;
    const start = this.emitted - this.delay;
    const view = this.crossCheck.view(snapshot, now, this.cfg);
    const [decision, speech, targetMix, gain, allowed] = this.decide(view, start);
    this.mix += (targetMix - this.mix) * (1 - Math.exp(-0.02 / 0.05));
    this.dry = append(this.dry, x);
    const vi = await this.runtime.processSamples(x);
    this.emitted += vi.length;
    this.cleanTail = append(this.cleanTail, vi);
    let offset = 0;
    for (; offset + HOP <= this.cleanTail.length; offset += HOP) {
      this.cleanVad.step(this.cleanTail.subarray(offset, offset + HOP));
    }
    this.cleanTail = this.cleanTail.slice(offset);
    const tau = gain > this.gain ? this.cfg.xcheck_attack : this.cfg.xcheck_release;
    const decay = Math.exp(-1 / (Math.max(tau, 0.001) * RATE));
    const raw = new Float32Array(vi.length);
    const out = vi.map((sample, i) => {
      const ref = this.dry[start + i - this.dryStart] ?? 0;
      raw[i] = ref;
      this.gain = gain + (this.gain - gain) * decay;
      const mixed = sample * this.mix + ref * (1 - this.mix);
      return Math.trunc(Math.min(1, Math.max(-1, mixed * this.gain)) * 32767) / 32768;
    });
    const keep = Math.max(0, start + vi.length - this.dryStart);
    this.dry = this.dry.slice(keep);
    this.dryStart += keep;
    return {start, vi, out, raw, decision, speech, allowed, view};
  }
}

function append(left, right) {
  const result = new Float32Array(left.length + right.length);
  result.set(left);
  result.set(right, left.length);
  return result;
}
