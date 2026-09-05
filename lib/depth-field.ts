/**
 * Dense monocular depth, for the surfaces the landmark rig cannot reach:
 * torso, arms, hair, the room behind. Optional in every sense -- it needs
 * WebGPU, it loads off a CDN inside a worker, and if any of that fails the
 * experience runs exactly as it does on the tier below.
 *
 * It is never the authority for hands or face. A neural depth map is smooth
 * across occlusion boundaries and unstable frame to frame, so using it to
 * decide whether a strand of light is visible would make the field crawl along
 * every silhouette. It fills; it does not cut.
 */

export type RawDepth = {
  /** Affine-invariant inverse depth, larger meaning nearer. */
  data: Float32Array;
  width: number;
  height: number;
  /** Inference cost in ms, for the tier watchdog. */
  cost: number;
};

/** A point where the rig already knows the answer, in frame uv. */
export type DepthSample = { u: number; v: number; near: number };

export type AffineFit = { scale: number; shift: number; residual: number };

export type DepthMap = {
  /** Nearness on the rig's own scale, 0 far, 1 near. */
  data: Uint8Array;
  width: number;
  height: number;
  version: number;
};

export type DepthEstimator = {
  /** Queue a frame if the worker is idle and the interval has elapsed. */
  submit: (video: HTMLVideoElement, now: number) => void;
  /** The newest raw map, handed over once. */
  take: () => RawDepth | null;
  /** Median inference cost so far, or 0 before any frame completes. */
  readonly cost: number;
  destroy: () => void;
};

// Major-version pinned; jsDelivr resolves the range. Loaded by the worker
// only, so nothing here reaches the app bundle on the tiers that skip it.
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';
const MODEL = 'onnx-community/depth-anything-v2-small';
// 252 = 14 x 18: DPT patch embeddings want a multiple of 14, and this is a
// quarter of the model's default 518, which is where the cost comes from.
const INFERENCE_SIZE = 252;

const WORKER_SOURCE = `
let extractor = null;
let RawImage = null;
let canvas = null;
let context = null;

async function boot() {
  const mod = await import(${JSON.stringify(TRANSFORMERS_CDN)});
  mod.env.allowLocalModels = false;
  RawImage = mod.RawImage;
  extractor = await mod.pipeline('depth-estimation', ${JSON.stringify(MODEL)}, {
    device: 'webgpu',
    dtype: 'fp16',
  });
  // Override the processor's resize target. The default is 518 square, which
  // is four times the pixels we need for a light-shaping depth signal.
  try {
    const processor = extractor.processor?.image_processor ?? extractor.processor;
    if (processor) {
      processor.size = { width: ${INFERENCE_SIZE}, height: ${INFERENCE_SIZE} };
      processor.do_resize = true;
    }
  } catch {}
  self.postMessage({ type: 'ready' });
}

boot().catch((error) => self.postMessage({ type: 'failed', message: String(error) }));

self.onmessage = async (event) => {
  const bitmap = event.data?.bitmap;
  if (!bitmap) return;
  if (!extractor) { bitmap.close(); return; }
  const started = performance.now();
  try {
    if (!canvas || canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      context = canvas.getContext('2d', { willReadFrequently: true });
    }
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height);
    bitmap.close();
    const output = await extractor(new RawImage(pixels.data, bitmap.width, bitmap.height, 4));
    const tensor = output.predicted_depth ?? output.depth;
    const dims = tensor.dims ?? [bitmap.height, bitmap.width];
    const height = dims[dims.length - 2];
    const width = dims[dims.length - 1];
    const data = new Float32Array(tensor.data);
    self.postMessage(
      { type: 'depth', data, width, height, cost: performance.now() - started },
      [data.buffer],
    );
  } catch (error) {
    try { bitmap.close(); } catch {}
    self.postMessage({ type: 'error', message: String(error) });
  }
};
`;

/**
 * Least squares for the scale and shift that carry affine-invariant disparity
 * onto the rig's nearness scale. Depth Anything is trained scale- and
 * shift-invariant, so its output preserves ordering and nothing else; the rig
 * supplies the references the general case does not have.
 *
 * One reweighted pass: fit, discard the worst fifth, fit again. A hand crossing
 * in front of a face produces a handful of samples the model puts on the wrong
 * side, and they would otherwise drag the whole room with them.
 */
export function fitAffine(raw: RawDepth, samples: DepthSample[]): AffineFit | null {
  if (samples.length < 12) return null;
  const disparity = new Float64Array(samples.length);
  const near = new Float64Array(samples.length);
  let used = 0;
  for (const sample of samples) {
    const x = Math.round(sample.u * (raw.width - 1));
    const y = Math.round(sample.v * (raw.height - 1));
    if (x < 0 || y < 0 || x >= raw.width || y >= raw.height) continue;
    const value = raw.data[y * raw.width + x];
    if (!Number.isFinite(value)) continue;
    disparity[used] = value;
    near[used] = sample.near;
    used += 1;
  }
  if (used < 12) return null;

  const solve = (weights: Float64Array | null) => {
    let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < used; i += 1) {
      const w = weights ? weights[i] : 1;
      if (w === 0) continue;
      sw += w; sx += w * disparity[i]; sy += w * near[i];
      sxx += w * disparity[i] * disparity[i]; sxy += w * disparity[i] * near[i];
    }
    const determinant = sw * sxx - sx * sx;
    if (Math.abs(determinant) < 1e-12) return null;
    const scale = (sw * sxy - sx * sy) / determinant;
    const shift = (sxx * sy - sx * sxy) / determinant;
    return { scale, shift };
  };

  const first = solve(null);
  if (!first) return null;

  const residuals = new Float64Array(used);
  for (let i = 0; i < used; i += 1) {
    residuals[i] = Math.abs(first.scale * disparity[i] + first.shift - near[i]);
  }
  const sorted = Array.from(residuals.subarray(0, used)).sort((a, b) => a - b);
  const cutoff = sorted[Math.floor(sorted.length * .8)] || Infinity;
  const weights = new Float64Array(used);
  for (let i = 0; i < used; i += 1) weights[i] = residuals[i] <= cutoff ? 1 : 0;

  const second = solve(weights) ?? first;
  let error = 0;
  let counted = 0;
  for (let i = 0; i < used; i += 1) {
    if (!weights[i]) continue;
    const delta = second.scale * disparity[i] + second.shift - near[i];
    error += delta * delta;
    counted += 1;
  }
  const residual = Math.sqrt(error / Math.max(counted, 1));

  // Disparity rises with nearness, so a non-positive scale is a failed fit,
  // not an unusual scene. A large residual means the same.
  if (!(second.scale > 0) || !Number.isFinite(residual) || residual > .3) return null;
  return { ...second, residual };
}

/** Apply a fit into a byte map on the rig's nearness scale. */
export function applyFit(raw: RawDepth, fit: AffineFit, into: Uint8Array) {
  const count = Math.min(into.length, raw.data.length);
  for (let i = 0; i < count; i += 1) {
    const value = fit.scale * raw.data[i] + fit.shift;
    into[i] = value <= 0 ? 0 : value >= 1 ? 255 : (value * 255) | 0;
  }
}

export async function createDepthEstimator(): Promise<DepthEstimator | null> {
  if (typeof navigator === 'undefined' || !(navigator as Navigator & { gpu?: unknown }).gpu) return null;
  if (typeof Worker === 'undefined' || typeof createImageBitmap !== 'function') return null;

  let worker: Worker;
  let url: string;
  try {
    url = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: 'text/javascript' }));
    worker = new Worker(url, { type: 'module' });
  } catch {
    return null;
  }

  let latest: RawDepth | null = null;
  let busy = false;
  let destroyed = false;
  let lastSubmit = -Infinity;
  // Starts at roughly 10 Hz and follows measured cost from there.
  let interval = 100;
  const costs: number[] = [];

  const ready = await new Promise<boolean>((resolve) => {
    // A cold start pulls the runtime and ~50 MB of weights. If that has not
    // happened in half a minute the tier below is the better experience.
    const timer = setTimeout(() => resolve(false), 30000);
    worker.onmessage = (event) => {
      const message = event.data;
      if (message?.type === 'ready') { clearTimeout(timer); resolve(true); }
      else if (message?.type === 'failed') { clearTimeout(timer); resolve(false); }
    };
    worker.onerror = () => { clearTimeout(timer); resolve(false); };
  });

  if (!ready || destroyed) {
    worker.terminate();
    URL.revokeObjectURL(url);
    return null;
  }

  worker.onmessage = (event) => {
    const message = event.data;
    busy = false;
    if (message?.type !== 'depth') return;
    latest = { data: message.data, width: message.width, height: message.height, cost: message.cost };
    costs.push(message.cost);
    if (costs.length > 16) costs.shift();
    // Keep a duty cycle rather than a frame rate: a slower machine runs the
    // model less often instead of falling behind on it.
    interval = Math.min(Math.max(message.cost * 1.8, 90), 500);
  };
  worker.onerror = () => { busy = false; };

  return {
    submit(video, now) {
      if (destroyed || busy || now - lastSubmit < interval) return;
      if (video.readyState < 2 || !video.videoWidth) return;
      lastSubmit = now;
      busy = true;
      createImageBitmap(video, {
        resizeWidth: INFERENCE_SIZE,
        resizeHeight: INFERENCE_SIZE,
        resizeQuality: 'medium',
      }).then((bitmap) => {
        if (destroyed) { bitmap.close(); busy = false; return; }
        worker.postMessage({ bitmap }, [bitmap]);
      }).catch(() => { busy = false; });
    },
    take() {
      const value = latest;
      latest = null;
      return value;
    },
    get cost() {
      if (!costs.length) return 0;
      const sorted = [...costs].sort((a, b) => a - b);
      return sorted[sorted.length >> 1];
    },
    destroy() {
      destroyed = true;
      worker.terminate();
      URL.revokeObjectURL(url);
    },
  };
}
