import type { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

type Vision = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

/**
 * A soft person silhouette, in video pixel coordinates before mirroring.
 * Coverage only: segmentation says where a body is, never how far away it is.
 * The depth it is composited at comes from the head proxy, which is measured.
 */
export type PersonMask = {
  data: Uint8Array;
  width: number;
  height: number;
  /** Increments on every new mask, so the renderer can skip stale uploads. */
  version: number;
};

export type PersonSegmenter = {
  /** Runs at most once per interval; returns the mask if it changed. */
  update: (source: CanvasImageSource, videoWidth: number, videoHeight: number, now: number) => PersonMask | null;
  destroy: () => void;
};

const MODEL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite';

// The model resamples to 256 square internally, so segmenting a small copy of
// the frame costs nothing in mask quality and makes the readback cheap.
const MASK_WIDTH = 256;
// Soft edges come from a blur here rather than from bilinear magnification,
// which would leave the silhouette visibly stepped at display resolution.
const FEATHER = 2;

/** Separable box blur, in place across `scratch`. Radius is small and fixed. */
function feather(data: Uint8Array, scratch: Uint8Array, width: number, height: number) {
  const window = FEATHER * 2 + 1;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    let sum = data[row] * (FEATHER + 1);
    for (let x = 1; x <= FEATHER; x += 1) sum += data[row + Math.min(x, width - 1)];
    for (let x = 0; x < width; x += 1) {
      scratch[row + x] = sum / window;
      sum += data[row + Math.min(x + FEATHER + 1, width - 1)] - data[row + Math.max(x - FEATHER, 0)];
    }
  }
  for (let x = 0; x < width; x += 1) {
    let sum = scratch[x] * (FEATHER + 1);
    for (let y = 1; y <= FEATHER; y += 1) sum += scratch[Math.min(y, height - 1) * width + x];
    for (let y = 0; y < height; y += 1) {
      data[y * width + x] = sum / window;
      sum += scratch[Math.min(y + FEATHER + 1, height - 1) * width + x] - scratch[Math.max(y - FEATHER, 0) * width + x];
    }
  }
}

/**
 * Optional whole-person coverage for the compositor. The hand rig stays the
 * authority wherever it reaches, because it carries per-finger depth; this
 * fills in everything the rig cannot know about -- torso, arms, shoulders,
 * hair -- so the field can pass behind a body and in front of it.
 * Returns null if the model will not load; the experience runs without it.
 */
export async function createPersonSegmenter(vision: Vision): Promise<PersonSegmenter | null> {
  const { ImageSegmenter: Segmenter } = await import('@mediapipe/tasks-vision');
  const options = {
    baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' as const },
    runningMode: 'VIDEO' as const,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  };

  let segmenter: ImageSegmenter;
  try {
    segmenter = await Segmenter.createFromOptions(vision, options);
  } catch {
    try {
      segmenter = await Segmenter.createFromOptions(vision, {
        ...options,
        baseOptions: { ...options.baseOptions, delegate: 'CPU' },
      });
    } catch {
      return null;
    }
  }

  const scratchCanvas = document.createElement('canvas');
  const scratchContext = scratchCanvas.getContext('2d', { willReadFrequently: false });
  if (!scratchContext) {
    segmenter.close();
    return null;
  }

  let mask: PersonMask | null = null;
  let blurScratch = new Uint8Array(0);
  let lastRun = -Infinity;
  // Segmentation is the slowest of the three models and the silhouette is the
  // slowest-changing signal, so it runs on its own clock and backs off further
  // if a call turns out to be expensive on this machine.
  let interval = 66;
  let failures = 0;

  return {
    update(source, videoWidth, videoHeight, now) {
      if (failures > 3 || now - lastRun < interval || videoWidth < 2 || videoHeight < 2) return null;
      lastRun = now;

      const height = Math.max(2, Math.round(MASK_WIDTH * videoHeight / videoWidth));
      if (scratchCanvas.width !== MASK_WIDTH || scratchCanvas.height !== height) {
        scratchCanvas.width = MASK_WIDTH;
        scratchCanvas.height = height;
      }
      const started = performance.now();
      let result;
      try {
        scratchContext.drawImage(source, 0, 0, MASK_WIDTH, height);
        result = segmenter.segmentForVideo(scratchCanvas, now);
      } catch {
        failures += 1;
        return null;
      }

      const category = result.categoryMask;
      if (!category) {
        result.close();
        failures += 1;
        return null;
      }
      const raw = category.getAsUint8Array();
      const width = category.width;
      const maskHeight = category.height;
      const pixels = width * maskHeight;
      if (!mask || mask.width !== width || mask.height !== maskHeight) {
        mask = { data: new Uint8Array(pixels), width, height: maskHeight, version: 0 };
        blurScratch = new Uint8Array(pixels);
      }
      // Class 0 is background for the selfie segmenters; every other class is
      // some part of a person. The mask is binary before feathering.
      for (let i = 0; i < pixels; i += 1) mask.data[i] = raw[i] > 0 ? 255 : 0;
      result.close();
      feather(mask.data, blurScratch, width, maskHeight);
      mask.version += 1;

      const cost = performance.now() - started;
      if (cost > 14) interval = Math.min(interval * 1.35, 400);
      else if (cost < 7) interval = Math.max(interval * .94, 60);
      return mask;
    },
    destroy() {
      segmenter.close();
    },
  };
}
