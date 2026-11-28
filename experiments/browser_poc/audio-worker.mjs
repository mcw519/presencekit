import * as ort from './vendor/ort.wasm.min.mjs';
import {PureSoundStreamingRuntime} from './vendor/runtime.js';
import {PresenceAudio} from './audio-core.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
const MODEL = new URL('./models/dpcrn_curriculum_v2.onnx', import.meta.url).href;
let engine;

let queue = Promise.resolve();
self.onmessage = ({data}) => {
  const {id, type} = data;
  queue = queue.then(async () => {
    try {
      if (type === 'init') {
        const response = await fetch(new URL('./models/dpcrn_curriculum_v2.json', import.meta.url));
        if (!response.ok) throw Error('PureSound manifest unavailable');
        const manifest = await response.json();
        manifest.recommended_inference.dry_blend = 1;
        delete manifest.onset_guard;
        const runtime = await PureSoundStreamingRuntime.create(MODEL, manifest, {backend: ort});
        engine = new PresenceAudio(runtime, manifest, data.config);
        postMessage({id});
      } else {
        const started = performance.now();
        const result = await engine.step(data.samples, data.snapshot, data.time);
        const {vi, out, raw} = result;
        postMessage({id, ...result, computeMs: performance.now() - started}, [vi.buffer, out.buffer, raw.buffer]);
      }
    } catch (error) {
      postMessage({id, error: error.message});
    }
  });
};
