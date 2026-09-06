import type { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

type Vision = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

/**
 * A soft person silhouette, in video pixel coordinates before mirroring.
 * Coverage only: segmentation says where a body is, never how far away it is.
 * The depth it is composited at comes from the rig and, when it is running,
 * from the dense depth map.
 */
export type PersonMask = {
  /** Any part of a person, feathered. */
  data: Uint8Array;
  /**
   * Skin specifically -- face and body -- when the multiclass model is in use.
   * Clothes and hair reflect very differently from skin, and the composite
   * uses this to keep a synthetic highlight off a jumper. All zero on the
   * binary model.
   */
  skin: Uint8Array;
  width: number;
  height: number;
  /** Increments on every new mask, so the renderer can skip stale uploads. */
  version: number;
  multiclass: boolean;
};

export type PersonSegmenter = {
  /** Runs at most once per interval; returns the mask if it changed. */
  update: (source: CanvasImageSource, videoWidth: number, videoHeight: number, now: number) => PersonMask | null;
  readonly multiclass: boolean;
  /** Kept explicit because sharing the compositor's GPU caused context loss. */
  readonly delegate: 'CPU';
  destroy: () => void;
};

const BINARY_MODEL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite';
/** 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other. */
const MULTICLASS_MODEL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';
const SKIN_CLASSES = new Set([2, 3]);

// Both models resample to 256 square internally, so segmenting a small copy of
// the frame costs nothing in mask quality and makes the readback cheap.
const MASK_WIDTH = 256;
// Soft edges come from a blur here rather than from bilinear magnification,
// which would leave the silhouette visibly stepped at display resolution.
const FEATHER = 2;
// A mask is a slow-moving cue. Blending it across frames stops a dim scene
// from making the lighting pop at every classification change.
const TEMPORAL_BLEND = .3;

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
 * Optional whole-person coverage for the compositor. The hand rig and the face
 * mesh stay the authority wherever they reach, because they carry real
 * per-vertex depth; this fills in the outline of everything they cannot know
 * about -- torso, arms, shoulders, hair.
 * Returns null if no model will load; the experience runs without it.
 */
export async function createPersonSegmenter(
  vision: Vision,
  preferMulticlass: boolean,
): Promise<PersonSegmenter | null> {
  const { ImageSegmenter: Segmenter } = await import('@mediapipe/tasks-vision');

  const attempt = async (modelAssetPath: string) =>
    Segmenter.createFromOptions(vision, {
      // The visual renderer, hand tracker, and a GPU segmenter can otherwise
      // contend for the browser's video/GPU resources. That manifested as a
      // periodic black composite and eventually a lost WebGL context. The
      // silhouette is deliberately slow-moving, so CPU is the stable trade.
      baseOptions: { modelAssetPath, delegate: 'CPU' },
      runningMode: 'VIDEO' as const,
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });

  // A CPU segmenter cannot interrupt a WebGL render target or steal the
  // camera texture. Prefer the richer model where the quality tier allows it,
  // then fall back to the binary silhouette.
  const plans: [string, boolean][] = preferMulticlass
    ? [[MULTICLASS_MODEL, true], [BINARY_MODEL, false]]
    : [[BINARY_MODEL, false]];

  let segmenter: ImageSegmenter | null = null;
  let multiclass = false;
  for (const [model, isMulticlass] of plans) {
    try {
      segmenter = await attempt(model);
      multiclass = isMulticlass;
      break;
    } catch {
      segmenter = null;
    }
  }
  if (!segmenter) return null;
  const active: ImageSegmenter = segmenter;

  const scratchCanvas = document.createElement('canvas');
  const scratchContext = scratchCanvas.getContext('2d', { willReadFrequently: false });
  if (!scratchContext) {
    active.close();
    return null;
  }

  let mask: PersonMask | null = null;
  let coverageScratch = new Uint8Array(0);
  let skinScratch = new Uint8Array(0);
  let coverageHistory = new Uint8Array(0);
  let skinHistory = new Uint8Array(0);
  let hasHistory = false;
  let lastRun = -Infinity;
  let stableInputHeight = 0;
  // Segmentation is the slowest of the MediaPipe models and the silhouette is
  // the slowest-changing signal. A fixed input shape also means the Three
  // texture is allocated once and never swaps underneath a live render pass.
  let interval = 250;
  let failures = 0;

  return {
    get multiclass() { return multiclass; },
    get delegate() { return 'CPU' as const; },
    update(source, videoWidth, videoHeight, now) {
      if (failures > 3 || now - lastRun < interval || videoWidth < 2 || videoHeight < 2) return null;
      lastRun = now;

      // Camera constraints are stable for a stream. Pin the segmenter's input
      // shape on the first decoded frame rather than reallocating a canvas and
      // mask texture if a browser reports a transient size during warm-up.
      stableInputHeight ||= Math.max(2, Math.round(MASK_WIDTH * videoHeight / videoWidth));
      if (scratchCanvas.width !== MASK_WIDTH || scratchCanvas.height !== stableInputHeight) {
        scratchCanvas.width = MASK_WIDTH;
        scratchCanvas.height = stableInputHeight;
      }
      const started = performance.now();
      let result;
      try {
        scratchContext.drawImage(source, 0, 0, MASK_WIDTH, stableInputHeight);
        result = active.segmentForVideo(scratchCanvas, now);
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
        mask = {
          data: new Uint8Array(pixels),
          skin: new Uint8Array(pixels),
          width,
          height: maskHeight,
          version: 0,
          multiclass,
        };
        coverageScratch = new Uint8Array(pixels);
        skinScratch = new Uint8Array(pixels);
        coverageHistory = new Uint8Array(pixels);
        skinHistory = new Uint8Array(pixels);
        hasHistory = false;
      }
      // Class 0 is background in both models; every other class is some part
      // of a person. Both masks are binary before feathering.
      for (let i = 0; i < pixels; i += 1) {
        const label = raw[i];
        const coverage = label > 0 ? 255 : 0;
        const skin = multiclass && SKIN_CLASSES.has(label) ? 255 : 0;
        coverageHistory[i] = hasHistory
          ? coverageHistory[i] + (coverage - coverageHistory[i]) * TEMPORAL_BLEND
          : coverage;
        skinHistory[i] = hasHistory
          ? skinHistory[i] + (skin - skinHistory[i]) * TEMPORAL_BLEND
          : skin;
        mask.data[i] = coverageHistory[i];
        mask.skin[i] = skinHistory[i];
      }
      hasHistory = true;
      result.close();
      feather(mask.data, coverageScratch, width, maskHeight);
      if (multiclass) feather(mask.skin, skinScratch, width, maskHeight);
      mask.version += 1;

      const cost = performance.now() - started;
      if (cost > 18) interval = Math.min(interval * 1.35, 800);
      else if (cost < 9) interval = Math.max(interval * .94, 220);
      return mask;
    },
    destroy() {
      active.close();
    },
  };
}
