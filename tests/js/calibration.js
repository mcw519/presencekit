// Distance calibration card: presets, 3s countdown, reset, enable shortcut, and server ack sync.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const staticRoot = path.resolve(__dirname, '../../presencekit/static');
const appSrc = fs.readFileSync(path.join(staticRoot, 'js/app.js'), 'utf8');
const i18nEn = fs.readFileSync(path.join(staticRoot, 'i18n/en.js'), 'utf8');
const i18nZh = fs.readFileSync(path.join(staticRoot, 'i18n/zh-TW.js'), 'utf8');

function mkEl(id, extra = {}) {
  const el = {
    id,
    value: extra.value || '',
    textContent: '',
    innerHTML: '',
    hidden: false,
    disabled: false,
    dataset: extra.dataset || {},
    listeners: {},
    classes: new Set(extra.classes || []),
    addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); },
    querySelectorAll() { return this._qsa ? this._qsa() : []; },
    insertAdjacentHTML() {},
    scrollIntoView() {},
  };
  el.classList = {
    toggle(c, force) {
      if (force === undefined) el.classes.has(c) ? el.classes.delete(c) : el.classes.add(c);
      else if (force) el.classes.add(c);
      else el.classes.delete(c);
    },
    add(c) { el.classes.add(c); },
    remove(c) { el.classes.delete(c); },
    contains(c) { return el.classes.has(c); },
  };
  return el;
}

const p05 = mkEl('p05', { dataset: { m: '0.5' } });
const p10 = mkEl('p10', { dataset: { m: '1.0' }, classes: ['is-active'] });
const p15 = mkEl('p15', { dataset: { m: '1.5' } });
[p05, p10, p15].forEach(el => { el.closest = () => el; });

const els = {
  calPresets: mkEl('calPresets'),
  calM: mkEl('calM', { value: '1.0' }),
  calB: mkEl('calB'),
  calDelayB: mkEl('calDelayB'),
  calReset: mkEl('calReset'),
  calEnable: mkEl('calEnable'),
  calBadge: mkEl('calBadge'),
  calHint: mkEl('calHint'),
  calibCard: mkEl('calibCard'),
  mod_distance: mkEl('mod_distance'),
  calJump: mkEl('calJump'),
};
els.calPresets._qsa = () => [p05, p10, p15];

let sent = [], toasts = [], intervals = [], nextId = 1;
const sandbox = {
  console,
  I18N: {},
  $: id => els[id] || null,
  t: (k, vars = {}) => (sandbox.I18N['en'][k] || k).replace(/\{(\w+)\}/g, (_, n) => vars[n] ?? ''),
  toast: msg => toasts.push(msg),
  cfg: { distance_enabled: true, distance_calib: 1.0 },
  running: true,
  last: { tracks: [{ id: 1, visible: true, distance: 0.68 }] },
  wsSend: o => sent.push(o),
  setCfg: o => { Object.assign(sandbox.cfg, o); sandbox.updateCalibUI(); },
  drawer: () => {},
  setInterval: (fn, ms) => { const id = nextId++; intervals.push({ id, fn, ms }); return id; },
  clearInterval: id => { intervals = intervals.filter(x => x.id !== id); },
};
vm.createContext(sandbox);
vm.runInContext(`${i18nEn}\n${i18nZh}`, sandbox);
vm.runInContext(appSrc.slice(appSrc.indexOf('let calTimer=null')), sandbox);
sandbox.addCalib();

assert.equal(els.calBadge.textContent, '#1 · 0.68 m · ×1.00');
assert.equal(els.calReset.disabled, true);
assert.equal(els.calEnable.hidden, true);

els.calPresets.listeners.click[0]({ target: p05 });
assert.equal(els.calM.value, '0.5');
assert.equal(p05.classList.contains('is-active'), true);
assert.equal(p10.classList.contains('is-active'), false);

els.calB.onclick();
assert.equal(JSON.stringify(sent.pop()), JSON.stringify({ type: 'calibrate', meters: 0.5 }));

els.calDelayB.onclick();
assert.equal(intervals.length, 1);
assert.match(els.calHint.textContent, /0\.5 m.*3 s/);
assert.equal(els.calHint.classList.contains('is-counting'), true);
intervals[0].fn();
assert.match(els.calHint.textContent, /2 s/);
intervals[0].fn();
assert.match(els.calHint.textContent, /1 s/);
intervals[0].fn();
assert.equal(intervals.length, 0);
assert.equal(els.calHint.classList.contains('is-counting'), false);
assert.equal(JSON.stringify(sent.pop()), JSON.stringify({ type: 'calibrate', meters: 0.5 }));

// Clicking countdown again cancels without sending.
els.calDelayB.onclick();
assert.equal(intervals.length, 1);
els.calDelayB.onclick();
assert.equal(intervals.length, 0);
assert.equal(els.calHint.classList.contains('is-counting'), false);

// Reset restores factor to 1.
sandbox.cfg.distance_calib = 1.45;
sandbox.updateCalibUI();
assert.equal(els.calReset.disabled, false);
assert.equal(els.calBadge.classList.contains('is-set'), true);
els.calReset.onclick();
assert.equal(sandbox.cfg.distance_calib, 1);
assert.equal(els.calReset.disabled, true);

// Disabled distance shows enable button and disables calibration buttons.
sandbox.cfg.distance_enabled = false;
sandbox.updateCalibUI();
assert.equal(els.calEnable.hidden, false);
assert.equal(els.calB.disabled, true);
assert.equal(els.calDelayB.disabled, true);
els.calEnable.onclick();
assert.equal(sandbox.cfg.distance_enabled, true);
assert.equal(els.calEnable.hidden, true);
