import * as ort from './vendor/ort.wasm.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
const MODEL = new URL('./models/yunet_s_dynamic.onnx', import.meta.url).href;
const STRIDES = [8, 16, 32];
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext('2d', {willReadFrequently: true, alpha: false});
let model, cfg, tracks = [];

function iou(a, b) {
  const area = ([x, y, xx, yy]) => (xx - x) * (yy - y);
  const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return width * height / Math.max(1e-9, area(a) + area(b) - width * height);
}

function inside(box, roi) {
  const x = (box[0] + box[2]) / 2, y = (box[1] + box[3]) / 2;
  return x >= roi[0] && x <= roi[2] && y >= roi[1] && y <= roi[3];
}

async function detect(bitmap) {
  const {width, height} = bitmap;
  const [rx, ry, rr, rb] = cfg.roi_crop ? cfg.face_roi : [0, 0, 1, 1];
  const margin = cfg.roi_crop ? cfg.crop_margin : 0;
  const x0 = Math.floor(Math.max(0, rx - margin) * width);
  const y0 = Math.floor(Math.max(0, ry - margin) * height);
  const cropWidth = Math.ceil(Math.min(1, rr + margin) * width) - x0;
  const cropHeight = Math.ceil(Math.min(1, rb + margin) * height) - y0;
  const scale = Math.min(1, cfg.max_width / cropWidth);
  const w = Math.max(1, Math.floor(cropWidth * scale)), h = Math.max(1, Math.floor(cropHeight * scale));
  const pw = Math.ceil(w / 32) * 32, ph = Math.ceil(h / 32) * 32;
  canvas.width = w; canvas.height = h;
  context.drawImage(bitmap, x0, y0, cropWidth, cropHeight, 0, 0, w, h);
  const rgba = context.getImageData(0, 0, w, h).data;
  const pixels = new Float32Array(ph * pw * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const from = 4 * (y * w + x), to = y * pw + x;
      pixels[to] = rgba[from + 2];
      pixels[to + ph * pw] = rgba[from + 1];
      pixels[to + ph * pw * 2] = rgba[from];
    }
  }
  const output = await model.run({[model.inputNames[0]]: new ort.Tensor('float32', pixels, [1, 3, ph, pw])});
  const faces = [];
  for (const stride of STRIDES) {
    const cls = output['cls_' + stride].data, obj = output['obj_' + stride].data;
    const boxes = output['bbox_' + stride].data, cols = pw / stride;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.max(0, Math.min(1, cls[i])) * Math.max(0, Math.min(1, obj[i])));
      if (score < 0.5) continue;
      const k = i * 4;
      const bw = Math.exp(boxes[k + 2]) * stride, bh = Math.exp(boxes[k + 3]) * stride;
      const x = (i % cols + boxes[k]) * stride - bw / 2;
      const y = (Math.floor(i / cols) + boxes[k + 1]) * stride - bh / 2;
      const [nx, ny, nw, nh] = [x, y, bw, bh].map(Math.trunc);
      faces.push({score, nms: [nx, ny, nx + nw, ny + nh], box: [
        Math.max(0, (x / scale + x0) / width), Math.max(0, (y / scale + y0) / height),
        Math.min(1, ((x + bw) / scale + x0) / width), Math.min(1, ((y + bh) / scale + y0) / height),
      ]});
    }
  }
  const kept = [];
  for (const face of faces.sort((a, b) => b.score - a.score).slice(0, 50)) {
    if (!kept.some(other => iou(face.nms, other.nms) > 0.3)) kept.push(face);
  }
  return kept.filter(face => face.score >= cfg.min_conf && inside(face.box, cfg.face_roi));
}

// Box association and measured face width are sufficient for this YuNet-only experiment.
function update(faces, time) {
  const recent = tracks.filter(track => time - track.seen <= (cfg.coast_seconds ?? 2));
  const used = new Set();
  tracks = faces.map(({box}) => {
    let prior, best = 0.2;
    for (const track of recent) {
      const overlap = iou(box, track.box);
      if (!used.has(track) && overlap > best) {
        prior = track;
        best = overlap;
      }
    }
    if (prior) used.add(prior);
    const rawDistance = cfg.face_width_m / ((box[2] - box[0]) * 2 * Math.tan(cfg.camera_hfov * Math.PI / 360));
    const measured = rawDistance * cfg.distance_calib;
    const distance = prior ? prior.distance * (1 - cfg.distance_ema) + measured * cfg.distance_ema : measured;
    return {box, distance, seen: time};
  });
  tracks.push(...recent.filter(track => !used.has(track)));
  const maxCoast = (cfg.coast_seconds ?? 2) * (1 - (cfg.coast_present_conf ?? 0.4));
  return {
    time,
    tracks: tracks.filter(track => time - track.seen <= maxCoast).map(track => ({
      ...track, distance: Math.round(track.distance * 100) / 100,
    })),
  };
}

self.onmessage = async ({data}) => {
  const {id, type, bitmap, time} = data;
  try {
    if (type === 'init') {
      cfg = data.config;
      tracks = [];
      model = await ort.InferenceSession.create(MODEL, {executionProviders: ['wasm']});
      postMessage({id});
    } else {
      const start = performance.now();
      const detections = await detect(bitmap);
      postMessage({id, snapshot: update(detections, time), computeMs: performance.now() - start});
    }
  } catch (error) {
    postMessage({id, error: error.message});
  } finally {
    bitmap?.close();
  }
};
