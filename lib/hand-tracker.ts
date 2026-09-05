import { Vector2 } from 'three';

type Point = { x: number; y: number; z?: number };

import { createFieldStateMachine, type FieldState } from './field-state';
import {
  buildPalmFrame,
  containsPoint,
  fieldDepthFromEndpoints,
  nearnessFromDistance,
  NEARNESS_SPAN_MM,
  type PalmFrame,
  type RigPoint,
} from './palm-geometry';
import { buildFaceFrame, trianglesFromTesselation, type FaceFrame } from './face-geometry';
import { createPersonSegmenter, type PersonMask, type PersonSegmenter } from './segmentation';
import {
  applyFit,
  createDepthEstimator,
  fitAffine,
  type AffineFit,
  type DepthEstimator,
  type DepthMap,
  type DepthSample,
} from './depth-field';
import { createGuideLine } from './guide-line';
import type { QualityController } from './quality';
export type { FaceFrame, FieldState, PalmFrame, PersonMask, RigPoint, DepthMap };

export type TrackingUpdate = {
  confinement: number;
  presence: number;
  hands: number;
  label: string;
  mode: 'palms' | 'pinch' | 'none';
  state: FieldState;
  /** 0 open, 1 fully sealed between the palms. */
  seal: number;
  /** Decaying impulse spent on the moment a clasp opens. */
  pulse: number;
  /** How squarely the palms face each other; 0 is edge-on. */
  facing: number;
  left: { x: number; y: number };
  right: { x: number; y: number };
  /** Depth of the plane the field hangs in, 0 far, 1 near. */
  fieldDepth: number;
  palms: PalmFrame[];
  leftDepth: number;
  rightDepth: number;
  /** 21 landmarks per hand, screen space with depth, for the occlusion rig. */
  rig: RigPoint[][];
  /** Millimetres per unit of nearness at each hand, for metric-space normals. */
  handScale: number[];
  /** The head as a real surface, when a face is visible. */
  face: FaceFrame | null;
  /** Shared triangle list for the face mesh; stable for the session. */
  faceTriangles: Uint16Array | null;
  /** Whole-person coverage, when the segmenter is available. */
  person: PersonMask | null;
  /** Dense depth on the rig's nearness scale, when the depth model is running. */
  depth: DepthMap | null;
  /**
   * Nearness the person silhouette is composited at when there is no dense
   * depth. Segmentation has no depth of its own, so the measured head distance
   * stands in for the whole body.
   */
  bodyDepth: number;
  /** How certain the tracker is, 0 to 1. Drives the guide line's marker. */
  confidence: number;
  /**
   * Where the text block that describes the state should sit, and which side of
   * the hand it ended up on. The tracker places it because it is also drawing
   * the leader that connects the two: one owner, so the line always meets the
   * text.
   *
   * Never null. The block does not come and go with the tracking -- when the
   * hands leave it simply stays where it was, and glides back when they return.
   */
  callout: { x: number; y: number; side: 'left' | 'right' };
};

/** Reserved size of the callout, in CSS pixels. Keep in step with the CSS. */
const CALLOUT_WIDTH = 280;
const CALLOUT_HEIGHT = 232;
/** Gap between the tracked dot and the near edge of the text. */
const CALLOUT_GAP = 54;
const CALLOUT_PAD = 38;
/**
 * How slowly the block follows the hand, in seconds. Deliberately far slower
 * than the tracking: the landmarks are already filtered, but a block of text
 * pinned rigidly to a moving hand reads as panic however smooth the underlying
 * signal is. This lags on purpose.
 */
const CALLOUT_TAU = .21;
/**
 * How far past the point of no room the hand has to travel before the block
 * changes sides, in pixels. Without it, a hand hovering at the edge of the
 * frame flips the text back and forth.
 */
const CALLOUT_SWITCH_HYSTERESIS = 110;

export type HandTracker = {
  setDebug: (enabled: boolean) => void;
  setGuide: (enabled: boolean) => void;
  destroy: () => void;
};

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const PALM_IDS = [0, 1, 5, 9, 13, 17];
const FINGERTIPS = [4, 8, 12, 16, 20];

/**
 * Depth from apparent size. A hand and a face are seen through the same lens,
 * so comparing each one's known physical width against how many pixels it
 * covers puts both on the same distance scale -- which the raw landmark z
 * values, being per-model and relative to their own origin, cannot do.
 */
const HAND_SPAN_MM = 85;    // index knuckle to pinky knuckle
// Approximate focal length; cover mapping uses the full drawn video width.
const FOCAL_FACTOR = .866;
function distanceMm(realMm: number, apparentPx: number, frameWidth: number) {
  return FOCAL_FACTOR * frameWidth * realMm / Math.max(apparentPx, 1);
}

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

/**
 * One Euro filter. A fixed lerp has to choose between jitter and lag; this
 * adapts its cutoff to how fast the value is actually moving, so a still hand
 * is steady and a moving one does not drag behind.
 */
class OneEuro {
  private value: number | null = null;
  private derivative = 0;
  private last = 0;

  constructor(private minCutoff = 1.4, private beta = .012, private dCutoff = 1.2) {}

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  reset() { this.value = null; this.derivative = 0; }

  filter(x: number, now: number) {
    if (this.value === null || now <= this.last) {
      this.value = x;
      this.last = now;
      return x;
    }
    const dt = Math.max((now - this.last) / 1000, 1e-3);
    this.last = now;
    const dx = (x - this.value) / dt;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    this.derivative = ad * dx + (1 - ad) * this.derivative;
    const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
    const a = OneEuro.alpha(cutoff, dt);
    this.value = a * x + (1 - a) * this.value;
    return this.value;
  }
}

/** A filter per coordinate, addressed by a stable slot index. */
class PointFilters {
  private filters: OneEuro[] = [];
  constructor(private minCutoff: number, private beta: number) {}
  at(slot: number) {
    let filter = this.filters[slot];
    if (!filter) { filter = new OneEuro(this.minCutoff, this.beta); this.filters[slot] = filter; }
    return filter;
  }
  resetFrom(slot: number) { for (let i = slot; i < this.filters.length; i += 1) this.filters[i]?.reset(); }
}

export async function createHandTracker(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  onUpdate: (update: TrackingUpdate) => void,
  quality: QualityController,
): Promise<HandTracker> {
  const { FaceLandmarker, FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm',
  );

  const options = {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU' as const,
    },
    runningMode: 'VIDEO' as const,
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };

  let handLandmarker: Awaited<ReturnType<typeof HandLandmarker.createFromOptions>>;
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, options);
  } catch {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: 'CPU' },
    });
  }

  // The face model is optional: if it will not load, hands still work and the
  // head simply stays out of the depth buffer, which is the old behaviour.
  let faceLandmarker: Awaited<ReturnType<typeof FaceLandmarker.createFromOptions>> | null = null;
  let faceTriangles: Uint16Array | null = null;
  if (quality.profile.faceMesh) {
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU' as const,
        },
        runningMode: 'VIDEO' as const,
        numFaces: 1,
        // Procrustes fit of the canonical face model, in centimetres. It gives
        // a head distance that head yaw does not corrupt, unlike apparent
        // cheek width, which shortens as soon as you turn.
        outputFacialTransformationMatrixes: true,
      });
      faceTriangles = trianglesFromTesselation(FaceLandmarker.FACE_LANDMARKS_TESSELATION);
      if (faceTriangles.length < 300) faceTriangles = null;
    } catch {
      faceLandmarker = null;
      faceTriangles = null;
    }
  }
  if (!faceLandmarker) quality.disable('faceMesh');

  const drawingContext = canvas.getContext('2d');
  if (!drawingContext) throw new Error('2D canvas is unavailable');
  const ctx: CanvasRenderingContext2D = drawingContext;

  let destroyed = false;
  let animationFrame = 0;
  let lastVideoTime = -1;
  let faceFrameCounter = 0;
  let lastFaceLandmarks: Point[] | null = null;
  let lastFaceMatrix: { data: number[] } | undefined;
  let smoothedConfinement = 0.08;
  const control = { left: new Vector2(), right: new Vector2() };
  let controlsReady = false;
  let debug = false;
  let guideVisible = true;
  const guide = createGuideLine();

  // Person segmentation is optional, and it loads without blocking: hands and
  // the field start immediately, and the silhouette joins the compositor
  // whenever the model arrives.
  let segmenter: PersonSegmenter | null = null;
  let personMask: PersonMask | null = null;
  if (quality.profile.segmentation) {
    void createPersonSegmenter(vision, quality.profile.multiclassSegmentation)
      .then((created) => {
        if (destroyed) created?.destroy();
        else if (created) segmenter = created;
        else quality.disable('segmentation');
      })
      .catch(() => { segmenter = null; quality.disable('segmentation'); });
  }

  // Dense depth is the last thing to arrive and the first thing to go. It only
  // ever fills surfaces the rig cannot reach.
  let depthEstimator: DepthEstimator | null = null;
  let depthMap: DepthMap | null = null;
  let depthFit: AffineFit | null = null;
  const depthSamples: DepthSample[] = [];
  if (quality.profile.denseDepth) {
    void createDepthEstimator()
      .then((created) => {
        if (destroyed) created?.destroy();
        else if (created) depthEstimator = created;
        else quality.disable('denseDepth');
      })
      .catch(() => quality.disable('denseDepth'));
  }
  quality.subscribe((profile) => {
    if (!profile.denseDepth && depthEstimator) {
      depthEstimator.destroy();
      depthEstimator = null;
      depthMap = null;
    }
    if (!profile.segmentation && segmenter) {
      segmenter.destroy();
      segmenter = null;
      personMask = null;
    }
  });

  // State machine memory.
  const machine = createFieldStateMachine();
  let state: FieldState = 'dormant';
  let lastFrameTime = 0;
  const claspCenter = new Vector2();
  let claspRadius = 1;
  let heldDepths = [.5, .5];
  let previousCenters: Vector2[] = [];
  let previousRig: RigPoint[][] = [];
  let previousPalms: PalmFrame[] = [];
  let previousScales: number[] = [];
  let seal = 0;
  let pulse = 0;
  let smoothedFacing = 0;
  let smoothedDepth = .5;
  let lastGapRatio = 99;
  let smoothedConfidence = 0;
  // The block is placed once and then followed. It belongs to one hand at a
  // time and holds its position when that hand is gone.
  const calloutPos = new Vector2();
  const calloutTarget = new Vector2();
  let calloutSide: 'left' | 'right' = 'right';
  let calloutPlaced = false;
  /** Which palm the block is locked to; -1 when nothing is tracked. */
  let anchorSlot = -1;
  // Held between tracked frames so the overlay can repaint at display rate.
  let guideAnchor: { x: number; y: number } | null = null;
  let guideTarget: { x: number; y: number } | null = null;
  let overlayHands: Point[][] = [];
  let overlayMode: 'palms' | 'pinch' | 'none' = 'none';
  // Control points can be smoothed harder than the rig, which has to stay
  // glued to the fingers it is masking.
  const controlFilters = new PointFilters(1.1, .009);
  const rigFilters = new PointFilters(2.4, .022);
  let faceVertexScratch: Float32Array | undefined;

  function resize() {
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    return { width, height };
  }

  function toScreen(point: Point, width: number, height: number) {
    const videoWidth = video.videoWidth || 1280;
    const videoHeight = video.videoHeight || 720;
    const scale = Math.max(width / videoWidth, height / videoHeight);
    const drawnWidth = videoWidth * scale;
    const drawnHeight = videoHeight * scale;
    const offsetX = (width - drawnWidth) / 2;
    const offsetY = (height - drawnHeight) / 2;
    return new Vector2((1 - point.x) * drawnWidth + offsetX, point.y * drawnHeight + offsetY);
  }

  /**
   * Redrawn every animation frame rather than only on frames that carried new
   * landmarks. The models deliver at about half the display rate, so a mark
   * drawn only when they do steps visibly however well the underlying signal is
   * filtered; advancing the follower and repainting at display rate is what
   * makes it read as smooth.
   */
  function renderOverlay(now: number) {
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;
    guide.update(guideAnchor, guideTarget, smoothedConfidence, now);
    drawOverlay(overlayHands, width, height, overlayMode);
  }

  function drawOverlay(hands: Point[][], width: number, height: number, mode: 'palms' | 'pinch' | 'none') {
    ctx.clearRect(0, 0, width, height);
    if (guideVisible) guide.draw(ctx, 1);
    if (!debug) return;
    ctx.lineCap = 'round';

    hands.forEach((hand) => {
      ctx.strokeStyle = 'rgba(215, 232, 244, .2)';
      ctx.lineWidth = 1;
      connections.forEach(([start, end]) => {
        const a = toScreen(hand[start], width, height);
        const b = toScreen(hand[end], width, height);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });

      ctx.fillStyle = 'rgba(226, 238, 247, .62)';
      [0, 4, 8, 9].forEach((id) => {
        const p = toScreen(hand[id], width, height);
        ctx.beginPath();
        ctx.arc(p.x, p.y, id === 9 ? 3.2 : 2.1, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    if (mode !== 'none') {
      ctx.save();
      ctx.strokeStyle = 'rgba(239, 244, 247, .62)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 7]);
      ctx.beginPath();
      ctx.moveTo(control.left.x, control.left.y);
      ctx.lineTo(control.right.x, control.right.y);
      ctx.stroke();
      ctx.setLineDash([]);
      [control.left, control.right].forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
        ctx.stroke();
      });
      ctx.restore();
    }
  }

  /**
   * Fold the newest depth map onto the rig's own scale. Depth Anything is
   * trained scale- and shift-invariant, so its output has the right ordering
   * and arbitrary units; the hands and face supply the references that turn it
   * into a distance. A fit that fails leaves the previous transform in place.
   */
  function alignDepth(rawHands: Point[][], rig: RigPoint[][], face: FaceFrame | null, faceLandmarks: Point[] | null) {
    const estimator = depthEstimator;
    if (!estimator) return;
    const raw = estimator.take();
    if (!raw) return;

    depthSamples.length = 0;
    for (let handIndex = 0; handIndex < rig.length; handIndex += 1) {
      const source = rawHands[handIndex];
      const filtered = rig[handIndex];
      if (!source || !filtered) continue;
      for (let i = 0; i < filtered.length; i += 1) {
        depthSamples.push({ u: source[i].x, v: source[i].y, near: filtered[i].z });
      }
    }
    if (face && faceLandmarks) {
      // Every twelfth landmark: enough of the head to pin the fit without
      // letting the face outvote the hands.
      for (let i = 0; i < face.count; i += 12) {
        depthSamples.push({ u: faceLandmarks[i].x, v: faceLandmarks[i].y, near: face.vertices[i * 3 + 2] });
      }
    }

    const fit = fitAffine(raw, depthSamples);
    if (fit) depthFit = fit;
    if (!depthFit) return;

    const pixels = raw.width * raw.height;
    if (!depthMap || depthMap.width !== raw.width || depthMap.height !== raw.height) {
      depthMap = { data: new Uint8Array(pixels), width: raw.width, height: raw.height, version: 0 };
    }
    applyFit(raw, depthFit, depthMap.data);
    depthMap.version += 1;
  }

  function updateControl(rawHands: Point[][], faceLandmarks: Point[] | null, width: number, height: number) {
    const frameTime = performance.now();
    const dt = Math.min((frameTime - lastFrameTime) / 1000 || 1 / 30, .1);
    lastFrameTime = frameTime;
    const ease = (tau: number) => 1 - Math.exp(-dt / tau);

    // Assign detections to previous screen positions before filtering. Detector
    // output order can change when hands cross or occlude one another.
    let hands = rawHands;
    const centers = hands.map(hand => toScreen(hand[9], width, height));
    if (hands.length === 2 && previousCenters.length === 2) {
      const direct = centers[0].distanceTo(previousCenters[0]) + centers[1].distanceTo(previousCenters[1]);
      const swapped = centers[1].distanceTo(previousCenters[0]) + centers[0].distanceTo(previousCenters[1]);
      if (swapped < direct) { hands = [hands[1], hands[0]]; centers.reverse(); }
    }
    if (hands.length !== previousCenters.length) rigFilters.resetFrom(0);
    previousCenters = centers;

    const videoWidth = video.videoWidth || 1280;
    const videoHeight = video.videoHeight || 720;
    const drawnWidth = videoWidth * Math.max(width / videoWidth, height / videoHeight);
    const handDistances = hands.map(hand => {
      // Include relative landmark z: an edge-on palm must not appear to recede.
      const span = Math.hypot(hand[5].x - hand[17].x,
        (hand[5].y - hand[17].y) * videoHeight / videoWidth,
        (hand[5].z ?? 0) - (hand[17].z ?? 0));
      return distanceMm(HAND_SPAN_MM, span * drawnWidth, drawnWidth);
    });

    // --- Rig first, so every gesture measure below reads the same filtered
    // geometry the renderer will mask and light with.
    let slot = 0;
    let rig = hands.map((hand, handIndex) => {
      const handMm = handDistances[handIndex];
      const width3d = Math.max(Math.hypot(hand[5].x - hand[17].x,
        (hand[5].y - hand[17].y) * videoHeight / videoWidth,
        (hand[5].z ?? 0) - (hand[17].z ?? 0)), .01);
      const localOrigin = hand.reduce((sum, p, i) => PALM_IDS.includes(i) ? sum + (p.z ?? 0) / PALM_IDS.length : sum, 0);
      return hand.map((point) => {
        const screen = toScreen(point, width, height);
        // Landmark z is relative within the hand; spread it over a plausible
        // depth for a hand rather than over the whole scene.
        const localMm = ((point.z ?? 0) - localOrigin) / width3d * HAND_SPAN_MM;
        return {
          x: rigFilters.at(slot++).filter(screen.x, frameTime),
          y: rigFilters.at(slot++).filter(screen.y, frameTime),
          z: rigFilters.at(slot++).filter(nearnessFromDistance(handMm + localMm), frameTime),
        };
      });
    });
    // Stale filters belong to a hand that has left; drop their history so a
    // returning hand does not slide in from where the last one was.
    rigFilters.resetFrom(slot);

    // Screen pixels per unit of nearness, per hand: the factor that puts the
    // palm's z into the same units as its x and y so the normal is real.
    const handScale = handDistances.map(mm => NEARNESS_SPAN_MM * FOCAL_FACTOR * drawnWidth / Math.max(mm, 1));
    let palms = rig.map((hand, index) => buildPalmFrame(hand, handScale[index] ?? 0));

    let targetConfinement = 0.08;
    let presence = 0;
    let label = 'Show one or two hands';
    let mode: 'palms' | 'pinch' | 'none' = 'none';
    let targetLeft = new Vector2(width * 0.42, height * 0.5);
    let targetRight = new Vector2(width * 0.58, height * 0.5);
    let facing = 0;
    let evidence = 0;
    let confidence = 0;

    if (palms.length >= 2) {
      const [a, b] = palms;
      const screenA = new Vector2(a.center.x, a.center.y);
      const screenB = new Vector2(b.center.x, b.center.y);
      const gap = screenA.distanceTo(screenB);

      // Palm size sets the scale for everything: a clasp is a gap comparable to
      // the palms themselves, not a fixed number of pixels.
      const palmSize = Math.max((a.radius + b.radius) / 2, 1e-3);
      lastGapRatio = gap / palmSize;

      // Facing, from the real palm normals against the palm-to-palm axis in the
      // same metric space. Absolute alignment, so edge-on reads 0 whichever way
      // a normal happens to point.
      const scale = (handScale[0] + handScale[1]) / 2;
      const axis = { x: b.center.x - a.center.x, y: b.center.y - a.center.y, z: (b.center.z - a.center.z) * scale };
      const axisLength = Math.hypot(axis.x, axis.y, axis.z) || 1;
      const unit = { x: axis.x / axisLength, y: axis.y / axisLength, z: axis.z / axisLength };
      facing = (Math.abs(a.normal.x * unit.x + a.normal.y * unit.y + a.normal.z * unit.z)
        + Math.abs(b.normal.x * unit.x + b.normal.y * unit.y + b.normal.z * unit.z)) / 2;

      // Extra clasp evidence. Overlapping in projection is not interlocking:
      // the palms have to be at a comparable distance, their silhouettes have
      // to be collapsing, and fingertips crossing into the other outline are
      // the strongest single sign that the hands have actually closed.
      const depthGapMm = Math.abs(a.center.z - b.center.z) * NEARNESS_SPAN_MM;
      const depthAgreement = 1 - clamp01((depthGapMm - 40) / 90);
      const collapse = 1 - clamp01(((a.openness + b.openness) / 2 - .34) / .34);
      let crossings = 0;
      for (const tip of FINGERTIPS) {
        if (containsPoint(b, rig[0][tip].x, rig[0][tip].y)) crossings += 1;
        if (containsPoint(a, rig[1][tip].x, rig[1][tip].y)) crossings += 1;
      }
      const containment = crossings / FINGERTIPS.length / 2;
      evidence = (containment * .55 + collapse * .3) * depthAgreement - (1 - depthAgreement) * .8;

      // The dial spans the whole gesture, from palms at their widest down to
      // a clasp, so it starts responding the instant the hands begin closing
      // rather than waiting for them to get near each other. Two palms held
      // out are five or six palm-widths apart; the range below covers that.
      //
      // It is deliberately linear in the gap. The acceleration the eye reads
      // comes from E/E0 = (L0/L)^2, which is the physics, not from bending
      // this curve.
      targetConfinement = 1 - clamp01((lastGapRatio - 1.0) / 4.6);
      presence = 1;
      [targetLeft, targetRight] = [screenA, screenB];
      label = 'Two palms';
      mode = 'palms';
      confidence = clamp01(.45 + (a.openness + b.openness) * .35);
    } else if (palms.length === 1) {
      const hand = rig[0];
      const palm = palms[0];
      const pinchDistance = Math.hypot(hand[4].x - hand[8].x, hand[4].y - hand[8].y);
      const palmWidth = Math.max(Math.hypot(hand[5].x - hand[17].x, hand[5].y - hand[17].y), 1);
      const pinchRatio = pinchDistance / palmWidth;
      targetConfinement = 1 - clamp01((pinchRatio - 0.14) / 0.62);
      presence = 1;
      targetLeft = new Vector2(hand[4].x, hand[4].y);
      targetRight = new Vector2(hand[8].x, hand[8].y);
      if (targetLeft.x > targetRight.x) [targetLeft, targetRight] = [targetRight, targetLeft];
      facing = clamp01(1 - Math.abs(palm.normal.z) * .6);
      label = 'One hand · pinch';
      mode = 'pinch';
      confidence = clamp01(.4 + palm.openness * .5);
      lastGapRatio = 99;
    } else {
      lastGapRatio = 99;
    }

    const nearClasp = palms.length === 1
      && new Vector2(palms[0].center.x, palms[0].center.y).distanceTo(claspCenter) < claspRadius * 1.8;
    const transition = machine.update({
      now: frameTime,
      hands: hands.length,
      gapRatio: lastGapRatio,
      confinement: targetConfinement,
      nearClasp,
      evidence,
    });
    state = transition.state;
    seal = transition.seal;
    pulse = transition.pulse;
    if (state === 'clasped' && hands.length === 2) {
      claspCenter.copy(targetLeft).add(targetRight).multiplyScalar(.5);
      claspRadius = Math.max(targetLeft.distanceTo(targetRight), 35);
    }
    if (transition.holdAnchors) {
      // Move the compact knot with the remaining palm, never switch its ends
      // to thumb and index while a clasp is temporarily hidden.
      const nextCenter = palms.length ? new Vector2(palms[0].center.x, palms[0].center.y) : claspCenter;
      const shift = nextCenter.clone().sub(claspCenter).multiplyScalar(.25);
      targetLeft.copy(control.left).add(shift);
      targetRight.copy(control.right).add(shift);
      claspCenter.add(shift);
      targetConfinement = 1;
      mode = 'palms';
      presence = hands.length ? 1 : .55;
      confidence = Math.min(confidence, .45);
    }

    // A clasp briefly losing both hands keeps its last geometry rather than
    // dropping the surfaces it was lighting. The palms have to be held with the
    // rig: restoring one without the other leaves the renderer masking fingers
    // that have no surface to emit from.
    if (transition.holdAnchors && hands.length === 0) {
      rig = previousRig;
      palms = previousPalms;
    } else if (rig.length) {
      previousRig = rig;
      previousPalms = palms;
      previousScales = handScale;
    }

    if (mode === 'none') {
      targetLeft.copy(control.left); targetRight.copy(control.right);
      controlsReady = false;
    }

    if (!controlsReady && mode !== 'none') {
      controlFilters.resetFrom(0);
      control.left.copy(targetLeft);
      control.right.copy(targetRight);
      controlsReady = true;
    } else {
      // One Euro rather than a fixed lerp: steady when the hands hold still,
      // still responsive when they move. minCutoff trades jitter for lag.
      control.left.set(
        controlFilters.at(0).filter(targetLeft.x, frameTime),
        controlFilters.at(1).filter(targetLeft.y, frameTime),
      );
      control.right.set(
        controlFilters.at(2).filter(targetRight.x, frameTime),
        controlFilters.at(3).filter(targetRight.y, frameTime),
      );
    }

    smoothedConfinement += (targetConfinement - smoothedConfinement) * ease(.055);
    smoothedFacing += (facing - smoothedFacing) * ease(.12);
    smoothedConfidence += (confidence - smoothedConfidence) * ease(.2);

    const labels: Record<FieldState, string> = {
      dormant: 'Show one or two hands',
      open: `${label} · open`,
      compressing: `${label} · compressing`,
      critical: `${label} · near collapse`,
      clasped: 'Sealed',
      release: `${label} · releasing`,
    };

    // The head, as a surface rather than an ellipsoid.
    let face: FaceFrame | null = null;
    if (faceLandmarks && quality.profile.faceMesh) {
      faceVertexScratch = faceVertexScratch ?? new Float32Array(478 * 3);
      face = buildFaceFrame(
        faceLandmarks,
        lastFaceMatrix,
        (point) => toScreen(point, width, height),
        (realMm, apparentPx) => distanceMm(realMm, apparentPx, drawnWidth),
        faceVertexScratch,
      );
    }

    let leftDepth = palms[0]?.center.z ?? heldDepths[0];
    let rightDepth = palms[1]?.center.z ?? leftDepth;
    if (mode === 'pinch' && rig[0]) {
      const ordered = [rig[0][4], rig[0][8]].sort((a, b) => a.x - b.x);
      [leftDepth, rightDepth] = ordered.map(p => p.z);
    }
    if (transition.holdAnchors) [leftDepth, rightDepth] = heldDepths;
    else heldDepths = [leftDepth, rightDepth];
    const normalisedPlane = fieldDepthFromEndpoints({ x: 0, y: 0, z: leftDepth }, { x: 0, y: 0, z: rightDepth });
    smoothedDepth += (normalisedPlane - smoothedDepth) * ease(.1);

    // One hand, and it stays that hand. The block picks the more open palm when
    // it first acquires and then holds that slot for as long as the slot
    // exists, so two visible hands never cause it to hop between them. The
    // detector's own ordering is already stabilised against the previous frame
    // above, so the slot keeps its identity through crossings.
    if (anchorSlot < 0 || anchorSlot >= palms.length) {
      anchorSlot = palms.length
        ? (palms.length === 2 && palms[1].openness > palms[0].openness ? 1 : 0)
        : -1;
    }
    const anchorPalm = anchorSlot >= 0 ? palms[anchorSlot] : undefined;
    const anchor = anchorPalm ? { x: anchorPalm.center.x, y: anchorPalm.center.y } : null;

    if (anchor) {
      // Which side has room, with a wide band before it will change its mind.
      const limit = width - CALLOUT_WIDTH - CALLOUT_PAD;
      if (calloutSide === 'right' && anchor.x + CALLOUT_GAP > limit) calloutSide = 'left';
      else if (calloutSide === 'left' && anchor.x + CALLOUT_GAP < limit - CALLOUT_SWITCH_HYSTERESIS) {
        calloutSide = 'right';
      }
      const preferred = calloutSide === 'right'
        ? anchor.x + CALLOUT_GAP
        : anchor.x - CALLOUT_GAP - CALLOUT_WIDTH;
      calloutTarget.set(
        Math.max(CALLOUT_PAD, Math.min(limit, preferred)),
        Math.max(CALLOUT_PAD, Math.min(height - CALLOUT_HEIGHT - CALLOUT_PAD, anchor.y - CALLOUT_HEIGHT * .3)),
      );
    } else if (!calloutPlaced) {
      // Nothing has been tracked yet: rest at the left of the frame, and glide
      // out to the hand when one arrives.
      calloutTarget.set(CALLOUT_PAD + 18, Math.max(CALLOUT_PAD, height * .5 - CALLOUT_HEIGHT * .5));
    } else {
      // The hands have gone. Hold position rather than snapping anywhere.
      calloutTarget.copy(calloutPos);
    }

    if (!calloutPlaced) {
      calloutPos.copy(calloutTarget);
      calloutPlaced = true;
    } else {
      calloutPos.lerp(calloutTarget, ease(CALLOUT_TAU));
    }
    const callout = { x: calloutPos.x, y: calloutPos.y, side: calloutSide };

    // The leader is only drawn when there is a hand to point at; the text
    // itself never leaves. Both ends are handed to the overlay, which advances
    // and repaints them at display rate.
    guideAnchor = anchor;
    guideTarget = anchor
      ? { x: calloutSide === 'right' ? calloutPos.x - 2 : calloutPos.x + CALLOUT_WIDTH + 2, y: calloutPos.y + 30 }
      : null;

    alignDepth(hands, rig, face, faceLandmarks);
    overlayHands = hands;
    overlayMode = mode;

    onUpdate({
      confinement: smoothedConfinement,
      presence,
      hands: hands.length,
      label: labels[state],
      mode,
      state,
      seal,
      pulse,
      facing: smoothedFacing,
      left: { x: control.left.x, y: control.left.y },
      right: { x: control.right.x, y: control.right.y },
      fieldDepth: smoothedDepth,
      leftDepth, rightDepth, palms,
      rig,
      handScale: rig === previousRig && !hands.length ? previousScales : handScale,
      face,
      faceTriangles,
      person: personMask,
      depth: depthMap,
      // Without a head there is no measured distance for a body, so the
      // silhouette is parked behind the scene: it can still catch rim light,
      // but it never wrongly swallows the field.
      bodyDepth: face ? Math.max(face.depth - .035, 0) : .1,
      confidence: smoothedConfidence,
      callout,
    });
  }

  function predict() {
    if (destroyed) return;
    const { width, height } = resize();
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const stamp = performance.now();
      const result = handLandmarker.detectForVideo(video, stamp);
      let face: Point[] | null = null;
      if (faceLandmarker && quality.profile.faceMesh) {
        try {
          // Face geometry changes far slower than hands, so it is sampled at
          // half rate to keep two models inside the frame budget.
          faceFrameCounter = (faceFrameCounter + 1) % 2;
          if (faceFrameCounter === 0) {
            const faceResult = faceLandmarker.detectForVideo(video, stamp);
            lastFaceLandmarks = (faceResult.faceLandmarks?.[0] as Point[] | undefined) ?? null;
            lastFaceMatrix = faceResult.facialTransformationMatrixes?.[0];
          }
          face = lastFaceLandmarks;
        } catch {
          face = null;
        }
      }
      // The silhouette is sampled from the same frame the landmarks came from.
      if (segmenter) personMask = segmenter.update(video, video.videoWidth, video.videoHeight, stamp) ?? personMask;
      depthEstimator?.submit(video, stamp);
      updateControl(result.landmarks as Point[][], face, width, height);
    }
    renderOverlay(performance.now());
    animationFrame = requestAnimationFrame(predict);
  }

  window.addEventListener('resize', resize);
  predict();

  return {
    setDebug(enabled) {
      debug = enabled;
      if (!debug) ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    },
    setGuide(enabled) {
      guideVisible = enabled;
      if (!enabled) guide.reset();
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      handLandmarker.close();
      faceLandmarker?.close();
      segmenter?.destroy();
      segmenter = null;
      depthEstimator?.destroy();
      depthEstimator = null;
    },
  };
}
