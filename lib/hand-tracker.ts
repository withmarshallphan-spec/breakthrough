import { Vector2 } from 'three';

type Point = { x: number; y: number; z?: number };

import { createFieldStateMachine, type FieldState } from './field-state';
import { buildPalmFrame, fieldDepthFromEndpoints, nearnessFromDistance, type PalmFrame, type RigPoint } from './palm-geometry';
export type { FieldState, PalmFrame, RigPoint };

export type TrackingUpdate = {
  confinement: number;
  presence: number;
  hands: number;
  label: string;
  mode: 'palms' | 'pinch' | 'none';
  state: FieldState;
  /** 0 open, 1 fully sealed between the palms. */
  seal: number;
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
  /** Head proxy, when a face is visible. */
  face: FaceProxy | null;
  fingertips: { x: number; y: number }[];
};

export type HandTracker = {
  setDebug: (enabled: boolean) => void;
  destroy: () => void;
};


/**
 * A head proxy for lighting and occlusion. Its depth comes from apparent size
 * against a known face width, the same measure used for the hands, so the two
 * sit on one distance scale and the light falls off between them correctly.
 */
export type FaceProxy = {
  center: { x: number; y: number };
  radiusX: number;
  radiusY: number;
  radiusZ: number;
  /** Column-major 3x3 basis: right, up, forward. */
  basis: number[];
  depth: number;
  nose: { x: number; y: number; z: number; radius: number };
};

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const PALM_IDS = [0, 1, 5, 9, 13, 17];

/**
 * Depth from apparent size. A hand and a face are seen through the same lens,
 * so comparing each one's known physical width against how many pixels it
 * covers puts both on the same distance scale -- which the raw landmark z
 * values, being per-model and relative to their own origin, cannot do.
 */
const HAND_SPAN_MM = 85;    // index knuckle to pinky knuckle
const FACE_WIDTH_MM = 145;  // cheekbone to cheekbone
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
const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

function palmCenter(hand: Point[]) {
  const ids = [0, 5, 9, 13, 17];
  return ids.reduce((center, id) => {
    center.x += hand[id].x / ids.length;
    center.y += hand[id].y / ids.length;
    return center;
  }, { x: 0, y: 0 });
}

/**
 * Palm plane from the wrist and the two outer knuckles. The normal is only ever
 * used through its absolute alignment with the palm-to-palm axis, so the
 * handedness sign convention cannot invert the result: a palm turned edge-on
 * reads zero either way.
 */
function palmNormal(hand: Point[]) {
  const w = hand[0];
  const a = { x: hand[5].x - w.x, y: hand[5].y - w.y, z: (hand[5].z ?? 0) - (w.z ?? 0) };
  const b = { x: hand[17].x - w.x, y: hand[17].y - w.y, z: (hand[17].z ?? 0) - (w.z ?? 0) };
  const n = {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
  const length = Math.hypot(n.x, n.y, n.z) || 1;
  return { x: n.x / length, y: n.y / length, z: n.z / length };
}

function palmDepth(hand: Point[]) {
  return PALM_IDS.reduce((sum, id) => sum + (hand[id].z ?? 0), 0) / PALM_IDS.length;
}

// Landmarks that bound the head and give it an orientation.
const FACE_TOP = 10;
const FACE_CHIN = 152;
const FACE_LEFT = 234;
const FACE_RIGHT = 454;
const FACE_NOSE = 1;

function normalise(v: { x: number; y: number; z: number }) {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function cross(a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) {
  return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
}

export async function createHandTracker(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  onUpdate: (update: TrackingUpdate) => void,
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
  // head simply falls back to relief inferred from the camera image.
  let faceLandmarker: Awaited<ReturnType<typeof FaceLandmarker.createFromOptions>> | null = null;
  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU' as const,
      },
      runningMode: 'VIDEO' as const,
      numFaces: 1,
    });
  } catch {
    faceLandmarker = null;
  }

  const drawingContext = canvas.getContext('2d');
  if (!drawingContext) throw new Error('2D canvas is unavailable');
  const ctx: CanvasRenderingContext2D = drawingContext;

  let destroyed = false;
  let animationFrame = 0;
  let lastVideoTime = -1;
  let faceFrame = 0;
  let lastFace: Point[] | null = null;
  let smoothedConfinement = 0.08;
  const control = { left: new Vector2(), right: new Vector2() };
  let controlsReady = false;
  let debug = false;

  // State machine memory.
  const machine = createFieldStateMachine();
  let state: FieldState = 'dormant';
  let lastFrameTime = 0;
  let claspCenter = new Vector2();
  let claspRadius = 1;
  let heldDepths = [.5, .5];
  let previousCenters: Vector2[] = [];
  let previousRig: RigPoint[][] = [];
  let seal = 0;
  let smoothedFacing = 0;
  let smoothedDepth = .5;
  let lastGapRatio = 99;
  // Control points can be smoothed harder than the rig, which has to stay
  // glued to the fingers it is masking.
  const controlFilters = new PointFilters(1.1, .009);
  const rigFilters = new PointFilters(2.4, .022);

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

  function drawHands(hands: Point[][], width: number, height: number, mode: 'palms' | 'pinch' | 'none') {
    ctx.clearRect(0, 0, width, height);
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
   * Fit an oriented ellipsoid to the face. Orientation comes straight from the
   * landmarks -- cheek to cheek across, chin to brow up -- the same trick used
   * for the palm plane, so no pose matrix has to be decoded.
   */
  function buildFace(landmarks: Point[] | null, width: number, height: number): FaceProxy | null {
    if (!landmarks || landmarks.length < 468) return null;
    const top = toScreen(landmarks[FACE_TOP], width, height);
    const chin = toScreen(landmarks[FACE_CHIN], width, height);
    const left = toScreen(landmarks[FACE_LEFT], width, height);
    const rightSide = toScreen(landmarks[FACE_RIGHT], width, height);
    const nose = toScreen(landmarks[FACE_NOSE], width, height);

    const center = {
      x: (top.x + chin.x + left.x + rightSide.x) / 4,
      y: (top.y + chin.y + left.y + rightSide.y) / 4,
    };
    const radiusY = Math.max(Math.hypot(top.x - chin.x, top.y - chin.y) / 2, 8) * 1.04;
    const radiusX = Math.max(Math.hypot(rightSide.x - left.x, rightSide.y - left.y) / 2, 8) * 1.08;

    // Basis in screen space, with the model's own z for the depth axis.
    const across = normalise({
      x: rightSide.x - left.x,
      y: rightSide.y - left.y,
      z: ((landmarks[FACE_RIGHT].z ?? 0) - (landmarks[FACE_LEFT].z ?? 0)) * width,
    });
    let up = normalise({
      x: top.x - chin.x,
      y: top.y - chin.y,
      z: ((landmarks[FACE_TOP].z ?? 0) - (landmarks[FACE_CHIN].z ?? 0)) * width,
    });
    const forward = normalise(cross(across, up));
    // Re-orthogonalise so the basis is clean even when the landmarks are not.
    up = normalise(cross(forward, across));

    // Same apparent-size measure the hands use, so the head lands at a distance
    // that is directly comparable with them: lean back and it dims.
    const frameWidth = (video.videoWidth || 1280) * Math.max(width / (video.videoWidth || 1280), height / (video.videoHeight || 720));
    const faceMm = distanceMm(FACE_WIDTH_MM, radiusX * 2, frameWidth);
    const depth = nearnessFromDistance(faceMm);

    return {
      center,
      radiusX,
      radiusY,
      radiusZ: radiusX * .92,
      basis: [across.x, across.y, across.z, up.x, up.y, up.z, forward.x, forward.y, forward.z],
      depth,
      // The nose reaches toward the camera by roughly its own radius.
      nose: { x: nose.x, y: nose.y, z: nearnessFromDistance(faceMm - 55), radius: radiusX * .3 },
    };
  }

  function updateControl(hands: Point[][], face: Point[] | null, width: number, height: number) {
    const frameTime = performance.now();
    const dt = Math.min((frameTime - lastFrameTime) / 1000 || 1 / 30, .1);
    lastFrameTime = frameTime;
    const ease = (tau: number) => 1 - Math.exp(-dt / tau);
    // Assign detections to previous screen positions before filtering. Detector
    // output order can change when hands cross or occlude one another.
    const centers = hands.map(hand => toScreen(palmCenter(hand), width, height));
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
    let targetConfinement = 0.08;
    let presence = 0;
    let label = 'Show one or two hands';
    let mode: 'palms' | 'pinch' | 'none' = 'none';
    let targetLeft = new Vector2(width * 0.42, height * 0.5);
    let targetRight = new Vector2(width * 0.58, height * 0.5);
    let facing = 0;

    const depths = hands.map(palmDepth);
    if (hands.length >= 2) {
      const centerA = palmCenter(hands[0]);
      const centerB = palmCenter(hands[1]);
      const screenA = toScreen(centerA, width, height);
      const screenB = toScreen(centerB, width, height);
      const gap = screenA.distanceTo(screenB);

      // Palm size sets the scale for everything: a clasp is a gap comparable to
      // the palms themselves, not a fixed number of pixels.
      const sizeA = Math.max(toScreen(hands[0][0], width, height).distanceTo(toScreen(hands[0][9], width, height)), toScreen(hands[0][5], width, height).distanceTo(toScreen(hands[0][17], width, height)));
      const sizeB = Math.max(toScreen(hands[1][0], width, height).distanceTo(toScreen(hands[1][9], width, height)), toScreen(hands[1][5], width, height).distanceTo(toScreen(hands[1][17], width, height)));
      const palmSize = Math.max((sizeA + sizeB) / 2, 1e-3);
      lastGapRatio = gap / palmSize;

      const axis = { x: centerB.x - centerA.x, y: centerB.y - centerA.y, z: (depths[1] - depths[0]) };
      const axisLength = Math.hypot(axis.x, axis.y, axis.z) || 1;
      const unit = { x: axis.x / axisLength, y: axis.y / axisLength, z: axis.z / axisLength };
      const nA = palmNormal(hands[0]);
      const nB = palmNormal(hands[1]);
      // Absolute alignment: edge-on palms read 0 whichever way the normal points.
      facing = (Math.abs(nA.x * unit.x + nA.y * unit.y + nA.z * unit.z)
        + Math.abs(nB.x * unit.x + nB.y * unit.y + nB.z * unit.z)) / 2;

      targetConfinement = 1 - clamp01((lastGapRatio - 1.05) / 5.5);
      presence = 1;
      // Keep identities through crossings; never exchange endpoint histories.
      [targetLeft, targetRight] = [screenA, screenB];
      label = 'Two palms';
      mode = 'palms';
    } else if (hands.length === 1) {
      const hand = hands[0];
      const pinchDistance = distance(hand[4], hand[8]);
      const palmWidth = Math.max(distance(hand[5], hand[17]), 0.035);
      const pinchRatio = pinchDistance / palmWidth;
      targetConfinement = 1 - clamp01((pinchRatio - 0.16) / 0.92);
      presence = 1;
      targetLeft = toScreen(hand[4], width, height);
      targetRight = toScreen(hand[8], width, height);
      if (targetLeft.x > targetRight.x) [targetLeft, targetRight] = [targetRight, targetLeft];
      const n = palmNormal(hand);
      facing = clamp01(1 - Math.abs(n.z) * .6);
      label = 'One hand · pinch';
      mode = 'pinch';

      lastGapRatio = 99;
    } else {
      lastGapRatio = 99;
    }

    const nearClasp = hands.length === 1 && toScreen(palmCenter(hands[0]), width, height).distanceTo(claspCenter) < claspRadius * 1.8;
    const transition = machine.update({ now: frameTime, hands: hands.length,
      gapRatio: lastGapRatio, confinement: targetConfinement, nearClasp });
    state = transition.state;
    seal = transition.seal;
    if (state === 'clasped' && hands.length === 2) {
      claspCenter.copy(targetLeft).add(targetRight).multiplyScalar(.5);
      claspRadius = Math.max(targetLeft.distanceTo(targetRight), 35);
    }
    if (transition.holdAnchors) {
      // Move the compact knot with the remaining palm, never switch its ends
      // to thumb and index while a clasp is temporarily hidden.
      const nextCenter = hands.length ? toScreen(palmCenter(hands[0]), width, height) : claspCenter;
      const shift = nextCenter.clone().sub(claspCenter).multiplyScalar(.25);
      targetLeft.copy(control.left).add(shift);
      targetRight.copy(control.right).add(shift);
      claspCenter.add(shift);
      targetConfinement = 1;
      mode = 'palms';
      presence = hands.length ? 1 : .55;
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

    smoothedConfinement += (targetConfinement - smoothedConfinement) * ease(.1);
    smoothedFacing += (facing - smoothedFacing) * ease(.12);

    const labels: Record<FieldState, string> = {
      dormant: 'Show one or two hands',
      open: `${label} · open`,
      compressing: `${label} · compressing`,
      clasped: 'Sealed',
      release: `${label} · releasing`,
    };

    drawHands(hands, width, height, mode);

    // Hands live in the near part of the depth range so the head proxy can sit
    // behind them; within that band their own z spread is preserved.
    let slot = 0;
    let rig = hands.map((hand, handIndex) => {
      const handMm = handDistances[handIndex];
      const width3d = Math.max(Math.hypot(hand[5].x - hand[17].x,
        (hand[5].y - hand[17].y) * videoHeight / videoWidth,
        (hand[5].z ?? 0) - (hand[17].z ?? 0)), .01);
      const localOrigin = palmDepth(hand);
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

    if (transition.holdAnchors && hands.length === 0) rig = previousRig;
    else previousRig = rig;
    const palms = rig.map(buildPalmFrame);
    // Use the very same filtered landmarks for the surface and its anchor.
    if (!transition.holdAnchors && palms.length === 2) {
      control.left.set(palms[0].center.x, palms[0].center.y);
      control.right.set(palms[1].center.x, palms[1].center.y);
    } else if (mode === 'pinch' && rig[0]) {
      const ordered = [rig[0][4], rig[0][8]].sort((a, b) => a.x - b.x);
      control.left.set(ordered[0].x, ordered[0].y);
      control.right.set(ordered[1].x, ordered[1].y);
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

    const fingertips = hands.flatMap((hand) => [4, 8, 12, 16, 20].map((id) => {
      const point = toScreen(hand[id], width, height);
      return { x: point.x, y: point.y };
    }));

    onUpdate({
      confinement: smoothedConfinement,
      presence,
      hands: hands.length,
      label: labels[state],
      mode,
      state,
      seal,
      facing: smoothedFacing,
      left: { x: control.left.x, y: control.left.y },
      right: { x: control.right.x, y: control.right.y },
      fieldDepth: smoothedDepth,
      leftDepth, rightDepth, palms,
      rig,
      face: buildFace(face, width, height),
      fingertips,
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
      if (faceLandmarker) {
        try {
          // Face geometry changes far slower than hands, so it is sampled at
          // half rate to keep two models inside the frame budget.
          faceFrame = (faceFrame + 1) % 2;
          if (faceFrame === 0) {
            const faceResult = faceLandmarker.detectForVideo(video, stamp);
            lastFace = (faceResult.faceLandmarks?.[0] as Point[] | undefined) ?? null;
          }
          face = lastFace;
        } catch {
          face = null;
        }
      }
      updateControl(result.landmarks as Point[][], face, width, height);
    }
    animationFrame = requestAnimationFrame(predict);
  }

  window.addEventListener('resize', resize);
  predict();

  return {
    setDebug(enabled) {
      debug = enabled;
      if (!debug) ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      handLandmarker.close();
      faceLandmarker?.close();
    },
  };
}
