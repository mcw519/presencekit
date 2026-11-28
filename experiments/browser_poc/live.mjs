import {FRAME_SECONDS, LocalRuntime, wav} from './runtime.mjs';
import {BLOCK, RATE, defaults} from './audio-core.mjs';
import {MonitorView} from './monitor-view.mjs';
import {initLanguage, localize} from './i18n.mjs';

const start = document.querySelector('#start');
const stop = document.querySelector('#stop');
const camera = document.querySelector('#camera');
const status = document.querySelector('#status');
const playback = document.querySelector('#playback');
const download = document.querySelector('#download');
const placeholder = document.querySelector('#placeholder');
const cameraMessage = document.querySelector('#camera-message');
const cameraLegend = document.querySelector('#camera-legend');
const captureChip = document.querySelector('#cam');
const runtimeDot = document.querySelector('#runtime-dot');
const recordingNote = document.querySelector('#recording-note');
const monitor = new MonitorView(camera, defaults.near_range);
const secondsLimit = 120;
let running, audioUrl, monitorChoice = 'off';

function setStatus(key) {
  localize(status, key);
}

function captureError(error) {
  const messages = {
    NotAllowedError: 'permissionDenied',
    NotFoundError: 'devicesMissing',
    NotReadableError: 'devicesBusy',
  };
  console.warn('Capture error', error);
  return messages[error.name] ?? 'captureError';
}

function selectMonitor(mode) {
  monitorChoice = mode;
  running?.monitorAudio?.port.postMessage(null);
  for (const button of document.querySelectorAll('[data-monitor]')) {
    button.setAttribute('aria-pressed', button.dataset.monitor === mode);
    button.disabled = !running?.monitorAudio;
  }
}

function finish(reason = 'stopped') {
  const session = running;
  if (!session) return;
  running = null;
  clearInterval(session.frames);
  session.runtime.close();
  session.media?.getTracks().forEach(track => track.stop());
  session.context.close().catch(() => {});
  camera.srcObject = null;
  start.hidden = false;
  stop.hidden = true;
  placeholder.hidden = false;
  localize(cameraMessage, 'startPrompt');
  cameraLegend.hidden = true;
  runtimeDot.classList.remove('is-ok');
  localize(captureChip, 'captureOff');
  captureChip.classList.remove('is-live');
  selectMonitor('off');
  setStatus(reason);
  if (!session.length) return;
  monitor.drawSignals();
  const output = new Float32Array(session.length);
  let offset = 0;
  for (const block of session.output) {
    output.set(block, offset);
    offset += block.length;
  }
  audioUrl = URL.createObjectURL(wav(output));
  playback.src = audioUrl;
  playback.hidden = false;
  download.href = audioUrl;
  download.download = 'presencekit-out.wav';
  download.hidden = false;
  localize(recordingNote, 'outputLength', {seconds: (session.length / RATE).toFixed(1)});
}

start.onclick = async () => {
  if (running) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(isSecureContext ? 'captureError' : 'secureContext');
    return;
  }
  start.hidden = true;
  stop.hidden = false;
  monitor.reset();
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  playback.removeAttribute('src');
  playback.hidden = download.hidden = true;
  localize(recordingNote, 'playbackHint');
  localize(cameraMessage, 'loadingPermission');
  setStatus('loadingModels');
  let session;
  try {
    const context = new AudioContext({sampleRate: RATE});
    const runtime = new LocalRuntime();
    session = {context, runtime, output: [], length: 0, backlog: 0, rendering: false, audioBlocks: 0};
    running = session;
    await context.resume();
    await runtime.init({});
    if (running !== session) return;
    const media = await navigator.mediaDevices.getUserMedia({
      video: {width: {ideal: 640}, height: {ideal: 480}},
      audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false},
    });
    if (running !== session) {
      media.getTracks().forEach(track => track.stop());
      return;
    }
    session.media = media;
    const begin = performance.now();
    const clock = () => (performance.now() - begin) / 1000;
    camera.srcObject = media;
    await camera.play();
    await context.audioWorklet.addModule(new URL('./capture-worklet.mjs', import.meta.url));
    if (running !== session) return;
    const capture = new AudioWorkletNode(context, 'microphone-blocks');
    context.createMediaStreamSource(media).connect(capture);
    const sink = context.createGain();
    sink.gain.value = 0;
    capture.connect(sink).connect(context.destination);
    session.monitorAudio = new AudioWorkletNode(context, 'monitor-audio', {numberOfInputs: 0, outputChannelCount: [1]});
    session.monitorAudio.connect(context.destination);
    selectMonitor('off');
    placeholder.hidden = true;
    cameraLegend.hidden = false;
    runtimeDot.classList.add('is-ok');
    captureChip.classList.add('is-live');
    localize(captureChip, 'captureOn');
    setStatus('capturingLocally');
    let nextRender = 0;
    capture.port.onmessage = async ({data}) => {
      if (running !== session) return;
      if (++session.backlog > 25) {
        finish('captureOverload');
        return;
      }
      const time = clock();
      try {
        const result = await runtime.block(data, time);
        if (running !== session) return;
        session.backlog--;
        const from = Math.max(0, -result.start);
        const block = result.out.subarray(from);
        session.output.push(block);
        session.length += block.length;
        monitor.collect(result);
        if (monitorChoice !== 'off' && block.length) {
          const samples = result[monitorChoice].slice(from);
          session.monitorAudio.port.postMessage(samples, [samples.buffer]);
        }
        if (session.length && time >= nextRender) {
          monitor.drawSignals();
          nextRender = time + 0.1;
        }
        if (++session.audioBlocks * BLOCK >= RATE * secondsLimit) finish('captureLimit');
      } catch (error) {
        if (running === session) finish(captureError(error));
      }
    };
    const observe = async () => {
      if (session.rendering || running !== session) return;
      session.rendering = true;
      try {
        const time = clock();
        const bitmap = await createImageBitmap(camera);
        if (running !== session) {
          bitmap.close();
          return;
        }
        const {snapshot} = await runtime.frame(bitmap, time);
        if (running === session) monitor.showFaces(snapshot);
      } catch (error) {
        if (running === session) finish(captureError(error));
      } finally {
        session.rendering = false;
      }
    };
    session.frames = setInterval(observe, FRAME_SECONDS * 1000);
    void observe();
  } catch (error) {
    if (session && running === session) finish(captureError(error));
    else if (!session) {
      setStatus(captureError(error));
      start.hidden = false;
      stop.hidden = true;
      localize(cameraMessage, 'startPrompt');
    }
  }
};
stop.onclick = () => finish();
for (const button of document.querySelectorAll('[data-monitor]')) {
  button.onclick = () => selectMonitor(button.dataset.monitor);
}
for (const button of document.querySelectorAll('[data-view-choice]')) {
  button.onclick = () => {
    const choice = button.dataset.viewChoice;
    document.querySelector('#stage').dataset.view = choice;
    document.querySelector('#sigStack').dataset.view = choice;
    for (const tab of document.querySelectorAll('[data-view-choice]')) tab.setAttribute('aria-pressed', tab === button);
    monitor.drawSignals();
  };
}
const themes = [...document.querySelectorAll('[data-theme-choice]')];
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  for (const button of themes) button.setAttribute('aria-pressed', button.dataset.themeChoice === theme);
}
setTheme(localStorage.getItem('presence-theme') || 'auto');
for (const button of themes) {
  button.onclick = () => {
    const theme = button.dataset.themeChoice;
    localStorage.setItem('presence-theme', theme);
    setTheme(theme);
  };
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) finish('backgroundStop');
});
initLanguage();
