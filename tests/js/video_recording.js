const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = name => fs.readFileSync(path.join(__dirname, '../../presencekit/static/js', name + '.js'), 'utf8');
const calls = [], nodes = {}, timers = new Map(), urls = new Map(), revoked = [], notices = [], controls = [], downloads = [];
let next = 0, clock = 0;
const element = id => nodes[id] ??= {classList: {toggle() {}}, hidden: true};
const ctx = new Proxy({measureText: text => ({width: text.length}),}, {
  get: (target, name) => target[name] ?? ((...args) => calls.push([name, ...args])),
});
const makeTrack = () => ({readyState: 'live', stopped: false, stop() { this.stopped = true; }});
const mic = makeTrack(), output = makeTrack(), cameraTracks = [];
const browser = {
  document: {
    createElement(tag) {
      if (tag === 'a') return {click() { downloads.push(this); }, remove() {}};
      assert.equal(tag, 'canvas');
      return {
        getContext: () => ctx,
        captureStream: fps => {
          assert.equal(fps, 24);
          const track = makeTrack(), tracks = [];
          cameraTracks.push(track);
          return {getVideoTracks: () => [track], addTrack: t => tracks.push(t), tracks};
        },
      };
    },
    body: {appendChild() {}},
  },
  MediaRecorder: class {
    static isTypeSupported(type) { return type === browser.supported; }
    constructor(media, options) {
      Object.assign(this, {media, mimeType: options.mimeType, state: 'inactive', stops: 0});
      browser.recorder = this;
    }
    start(timeslice) {
      assert.equal(timeslice, 1000);
      this.state = 'recording';
    }
    stop() {
      this.stops++;
      this.state = 'inactive';
    }
    finish(body = 'video') {
      this.ondataavailable({data: new Blob([body], {type: this.mimeType})});
      this.onstop();
    }
  },
  supported: 'video/mp4',
  URL: {
    createObjectURL(blob) {
      const url = 'blob:' + ++next;
      urls.set(url, blob);
      return url;
    },
    revokeObjectURL(url) { revoked.push(url); },
  },
  Blob, TextEncoder,
  setInterval: (fn, ms) => timers.set(++next, {fn, ms}) && next,
  setTimeout: (fn, ms) => timers.set(++next, {fn, ms}) && next,
  clearInterval: id => timers.delete(id),
  clearTimeout: id => timers.delete(id),
  performance: {now: () => clock},
  $: element,
  t: key => key,
  T: key => key,
  toast: key => notices.push(key),
  wsSend: msg => controls.push({type: msg.type, on: msg.on}),
  v: {readyState: 2, videoWidth: 640, videoHeight: 480},
  stream: {getAudioTracks: () => [mic]},
  ac: {createMediaStreamDestination: () => ({stream: {getAudioTracks: () => [output]}})},
  player: {connect: node => calls.push(['connect', node]), disconnect: node => calls.push(['disconnect', node])},
  aMeta: null,
};
vm.createContext(browser);
const run = js => vm.runInContext(js, browser);
const app = source('app');
run(`let running=false, last={tracks:[{id:7, visible:true, participant:true, bbox:[.1,.25,.3,.5], similarity:.8}]};
const DEF={mirror_view:true, audio_pipeline:'energy_vad'};
const cfg={...DEF, credential:'not-in-export'};
const visBoxes=()=>[], SK=[], ZC={};`);
run(app.slice(app.indexOf('const col='), app.indexOf('// ---------- overlay')));
run(app.slice(app.indexOf('const MIR='), app.indexOf('function draw()')));
run(app.slice(app.indexOf('function drawOverlay('), app.indexOf('function pos(')));
run(source('recording'));

element('recBtn').onclick();
assert.equal(notices.at(-1), 'toast.rec_wait');
assert.equal(browser.recorder, undefined);
run('running=true');
element('recBtn').onclick();
const raw = browser.recorder;
assert.deepEqual(raw.media.tracks, [mic]);
assert.deepEqual(calls.slice(0, 3), [['save'], ['translate', 640, 0], ['scale', -1, 1]]);
const box = calls.find(call => call[0] === 'strokeRect');
assert.equal(box[1], 448);
assert.equal(box[2], 120);
assert.ok(calls.findIndex(call => call[0] === 'restore') < calls.indexOf(box));
assert.ok(calls.some(call => call[0] === 'fillText' && call[1].includes('#7')));

calls.length = 0;
run('last.tracks[0].bbox=[.25,.5,.5,.75]; cfg.mirror_view=false');
[...timers.values()].find(({ms}) => ms === 1000 / 24).fn();
assert.ok(!calls.some(call => call[0] === 'scale'));
assert.equal(calls.find(call => call[0] === 'strokeRect')[1], 160);
clock = 1100;
element('recBtn').onclick();
element('recBtn').onclick();
assert.equal(raw.stops, 1);
assert.equal(element('recBtn').disabled, true);
assert.equal(timers.size, 0);
assert.equal(element('recResult').hidden, true);
raw.finish();
const first = element('recVideo').src;
assert.equal(element('recDownload').href, first);
assert.match(element('recDownload').download, /\.mp4$/);
assert.equal(element('recResult').hidden, false);
assert.equal(urls.get(first).size, 5);
assert.equal(cameraTracks[0].stopped, true);
assert.equal(mic.stopped, false);
assert.equal(element('recDownloadZip').hidden, true);
assert.deepEqual(controls, []);

browser.supported = 'video/webm;codecs=vp8,opus';
run("cfg.audio_pipeline='puresound'");
element('recBtn').onclick();
const clean = browser.recorder;
assert.deepEqual(clean.media.tracks, [output]);
const destination = calls.find(call => call[0] === 'connect')[1];
const pcm = browser.pcm = new DataView(new ArrayBuffer(25));
pcm.setUint8(0, 3);
pcm.setUint32(9, 2, true);
for (const [i, sample] of [11, 12, 21, 22, 31, 32].entries()) pcm.setInt16(13+2*i, sample, true);
run("requestRecTracks(); onRecTracks(new DataView(new ArrayBuffer(13))); onRecTracks(pcm)");
assert.equal(run('rec.tracks.samples'), 0);  // Late packet from before the current clip.
pcm.setFloat64(1, 1150, true);
run('onRecTracks(pcm)');
run("recTrackMeta({type:'audio',decision:'agree'}); recTrackMeta({type:'state',session:2,state:'ACTIVE',tracks:[]})");
assert.equal(run('rec.tracks.samples'), 2);
run("cfg.audio_pipeline='energy_vad'");
[...timers.values()].find(({ms}) => ms === 1000 / 24).fn();
assert.equal(clean.stops, 1);
assert.equal(notices.at(-1), 'toast.rec_pipeline');
run('onRecTracks(pcm)');
assert.equal(run('rec.tracks.samples'), 2);
clean.finish();
assert.deepEqual(controls, [{type:'record',on:true},{type:'record',on:true},{type:'record',on:false}]);
assert.deepEqual(calls.find(call => call[0] === 'disconnect'), ['disconnect', destination]);
assert.equal(output.stopped, true);
assert.deepEqual(revoked, [first]);
assert.match(element('recDownload').download, /\.webm$/);
assert.equal(element('recDownloadZip').hidden, false);

async function checkArchive(){
  await element('recDownloadZip').onclick();
  assert.match(downloads[0].download, /\.zip$/);
  const zip = urls.get(downloads[0].href);
  const bytes = Buffer.from(await zip.arrayBuffer()), files = {};
  let offset = 0;
  while (bytes.readUInt32LE(offset) === 0x04034b50) {
    const nameSize = bytes.readUInt16LE(offset+26), size = bytes.readUInt32LE(offset+18);
    const name = bytes.subarray(offset+30, offset+30+nameSize).toString();
    offset += 30+nameSize;
    files[name.slice(downloads[0].download.length-4)] = bytes.subarray(offset, offset+size);
    offset += size;
  }
  assert.deepEqual(Object.keys(files).sort(), ['.json','.webm','_out.wav','_raw.wav','_vi.wav']);
  assert.equal(files['.webm'].toString(), 'video');
  for (const [name, values] of Object.entries({'_out.wav':[11,12], '_vi.wav':[21,22], '_raw.wav':[31,32]})) {
    const wav = files[name];
    assert.equal(wav.subarray(0,4).toString(), 'RIFF');
    assert.equal(wav.readUInt32LE(24), 16000);
    assert.equal(wav.readUInt32LE(40), 4);
    assert.deepEqual([wav.readInt16LE(44),wav.readInt16LE(46)], values);
  }
  const side = JSON.parse(files['.json'].toString());
  assert.equal(side.samples, 2);
  assert.equal(side.blocks[0].decision, 'agree');
  assert.equal(side.states[0].state, 'ACTIVE');
  assert.equal(side.settings.credential, undefined);
  await element('recDownloadZip').onclick();
  assert.equal(downloads[1].href, downloads[0].href);
}

checkArchive().then(() => {
  element('recBtn').onclick();
  const oldPreview = element('recVideo').src;
  [...timers.values()].find(({ms}) => ms === 600000).fn();
  assert.equal(notices.at(-1), 'toast.rec_max');
  browser.recorder.finish('');
  assert.equal(notices.at(-1), 'toast.rec_empty');
  assert.equal(oldPreview, element('recVideo').src);
  assert.equal(timers.size, 0);
  assert.equal(mic.stopped, false);
  console.log('video recording, aligned WAV archive, preview and cleanup passed');
}).catch(err => { console.error(err); process.exitCode = 1; });
