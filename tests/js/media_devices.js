const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const code = fs.readFileSync(path.join(__dirname, '../../presencekit/static/js/media-devices.js'), 'utf8');
const camera = (deviceId, label = '') => ({kind: 'videoinput', deviceId, label});
const mic = (deviceId, label = '') => ({kind: 'audioinput', deviceId, label});
const stored = new Map();
const probeCalls = [];
const stopped = [];
const visible = [camera('cam-a', 'Built-in'), camera('cam-b', 'USB camera'),
  mic('default', 'Default microphone'), mic('mic-a', 'Internal'), mic('mic-b', 'USB microphone'),
  {kind: 'audiooutput', deviceId: 'speaker', label: 'Speaker'}];
let devices = [camera(''), mic('')];
let enumerateChanged;
const mediaDevices = {
  enumerateDevices: async () => devices,
  addEventListener(name, listener) {
    assert.equal(name, 'devicechange');
    enumerateChanged = listener;
  },
  async getUserMedia(constraints) {
    probeCalls.push(JSON.parse(JSON.stringify(constraints)));
    devices = visible;
    return {getTracks: () => ['video', 'audio'].map(kind => ({stop: () => stopped.push(kind)}))};
  },
};

function openPage() {
  const elements = Object.fromEntries(['cameraDevice', 'microphoneDevice', 'deviceStatus', 'start', 'loadDevices']
    .map(id => [id, {replaceChildren(...options) {this.options = options;}}]));
  const context = vm.createContext({
    Option: class {constructor(text, value) {Object.assign(this, {text, value});}},
    navigator: {mediaDevices},
    localStorage: {getItem: key => stored.get(key), setItem: (key, value) => stored.set(key, value),
      removeItem: key => stored.delete(key)},
    $: id => elements[id],
    t: key => key,
    running: false,
    autoPaused: false,
  });
  const run = expression => vm.runInContext(expression, context);
  run(code + '\nfunction setUI(){syncInputControls()} initInputDevices()');
  return {elements, run};
}

(async () => {
  let {elements, run} = openPage();
  await enumerateChanged();
  assert.deepEqual(probeCalls, []);
  assert.equal(elements.loadDevices.hidden, false);
  assert.deepEqual(elements.cameraDevice.options.map(option => option.value), ['']);
  await elements.loadDevices.onclick();
  assert.deepEqual(probeCalls, [{video: true, audio: true}]);
  assert.deepEqual(stopped, ['video', 'audio']);
  assert.deepEqual(elements.cameraDevice.options.map(option => option.value), ['', 'cam-a', 'cam-b']);
  assert.deepEqual(elements.microphoneDevice.options.map(option => option.value), ['', 'mic-a', 'mic-b']);
  assert.equal(elements.loadDevices.hidden, true);

  elements.cameraDevice.value = 'cam-b';
  elements.cameraDevice.onchange();
  elements.microphoneDevice.value = 'mic-b';
  elements.microphoneDevice.onchange();
  const constraints = JSON.parse(run('JSON.stringify(captureConstraints())'));
  assert.deepEqual(constraints.video.deviceId, {exact: 'cam-b'});
  assert.deepEqual(constraints.audio.deviceId, {exact: 'mic-b'});
  run('running=true;syncInputControls()');
  assert.equal(elements.cameraDevice.disabled, true);
  assert.equal(elements.microphoneDevice.disabled, true);
  assert.equal(elements.start.disabled, false);

  devices = [camera(''), mic('')];
  await enumerateChanged();
  assert.equal(elements.microphoneDevice.options.at(-1).text, 'USB microphone');
  ({elements, run} = openPage());
  await enumerateChanged();
  assert.equal(elements.cameraDevice.value, 'cam-b');
  assert.equal(elements.cameraDevice.options.at(-1).text, 'device.saved');

  devices = visible.filter(device => device.deviceId !== 'cam-b');
  await enumerateChanged();
  assert.equal(elements.cameraDevice.options.at(-1).text, 'device.disconnected');
  assert.equal(elements.cameraDevice.value, 'cam-b');
  assert.equal(run("inputStartError({name:'NotFoundError'},'toast.cam_fail')"), 'device.selection_failed');
  assert.equal(elements.deviceStatus.textContent, 'device.selection_failed');
  elements.cameraDevice.value = '';
  elements.cameraDevice.onchange();
  assert.equal(run('captureConstraints().video.facingMode'), 'user');
  assert.equal(stored.has('presence-camera-device'), false);
})().catch(error => {console.error(error); process.exitCode = 1;});
