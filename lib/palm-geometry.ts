export type RigPoint = { x: number; y: number; z: number };

/** Wrist, thumb base and the four MCP knuckles, in winding order. */
export const PALM_BOUNDARY = [0, 1, 5, 9, 13, 17] as const;

/** Catmull-Rom samples emitted per control edge of the palm silhouette. */
const CONTOUR_SEGMENTS = 4;
/** Control points below; the closed contour is this many samples long. */
export const PALM_CONTOUR_POINTS = 9 * CONTOUR_SEGMENTS;

export type PalmFrame = {
  center: RigPoint;
  /** The six landmarks themselves, unmodified. */
  boundary: RigPoint[];
  /**
   * Closed, smoothed palm silhouette. Built only from landmark positions, so
   * it rotates, foreshortens and deforms with the hand rather than being a
   * disc pinned to a centre point.
   */
  contour: RigPoint[];
  /**
   * Unit surface normal of the palm, in a space where z has been scaled into
   * the same units as x and y. Without that scaling the cross product is
   * dominated by the screen-space term and every palm reports as facing the
   * camera, which is exactly what makes landmark-driven lighting look like a
   * disc stuck to the hand.
   */
  normal: RigPoint;
  /** Wrist to middle knuckle: the palm's own long axis. */
  up: RigPoint;
  radius: number;
  /**
   * Twice the area the silhouette encloses over the area of the disc that
   * would fit its longest reach. A palm turned edge-on, or curled into a
   * fist, collapses its own outline, and this reads that off the landmarks
   * with no normal and no mixed units.
   */
  openness: number;
};

const lerp = (a: RigPoint, b: RigPoint, t: number): RigPoint => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
  z: a.z + (b.z - a.z) * t,
});

/** Push a point away from the palm centre, where the hand is actually widest. */
const widen = (p: RigPoint, center: RigPoint, k: number): RigPoint => ({
  x: p.x + (p.x - center.x) * k,
  y: p.y + (p.y - center.y) * k,
  z: p.z + (p.z - center.z) * k,
});

function catmullRom(p0: RigPoint, p1: RigPoint, p2: RigPoint, p3: RigPoint, t: number): RigPoint {
  const t2 = t * t;
  const t3 = t2 * t;
  const at = (a: number, b: number, c: number, d: number) =>
    .5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3);
  return {
    x: at(p0.x, p1.x, p2.x, p3.x),
    y: at(p0.y, p1.y, p2.y, p3.y),
    z: at(p0.z, p1.z, p2.z, p3.z),
  };
}

/** Resample a closed control ring into an evenly spaced, smooth contour. */
function smoothClosed(controls: RigPoint[]): RigPoint[] {
  const n = controls.length;
  const contour: RigPoint[] = [];
  for (let i = 0; i < n; i += 1) {
    const p0 = controls[(i - 1 + n) % n];
    const p1 = controls[i];
    const p2 = controls[(i + 1) % n];
    const p3 = controls[(i + 2) % n];
    for (let step = 0; step < CONTOUR_SEGMENTS; step += 1) {
      contour.push(catmullRom(p0, p1, p2, p3, step / CONTOUR_SEGMENTS));
    }
  }
  return contour;
}

/** Signed area doubled, in screen space. */
function signedArea(contour: RigPoint[]) {
  let area = 0;
  for (let i = 0; i < contour.length; i += 1) {
    const a = contour[i];
    const b = contour[(i + 1) % contour.length];
    area += a.x * b.y - b.x * a.y;
  }
  return area;
}

/**
 * The palm surface, from the landmarks alone. The MCP knuckles are carried a
 * short way toward their own PIP joints so the surface reaches into the roots
 * of the fingers, and the thumb and little-finger edges are widened over the
 * thenar and hypothenar muscle, which is where a palm is broadest. Every term
 * is a linear combination of landmark positions, so the shape stays correct
 * under rotation, tilt and perspective foreshortening for free.
 *
 * `zScale` converts the points' z -- carried as nearness, 0 to 1 -- into the
 * same units x and y are in, so the surface normal is a real one. Pass 0 and
 * the normal degenerates to facing the camera, which is the old behaviour.
 */
export function buildPalmFrame(hand: RigPoint[], zScale = 0): PalmFrame {
  const boundary = PALM_BOUNDARY.map(i => ({ ...hand[i] }));
  const center = [0, 5, 9, 13, 17].reduce((p, i) => ({
    x: p.x + hand[i].x / 5, y: p.y + hand[i].y / 5, z: p.z + hand[i].z / 5,
  }), { x: 0, y: 0, z: 0 });

  const controls = [
    hand[0],
    widen(lerp(hand[0], hand[1], .55), center, .18),
    widen(lerp(hand[1], hand[2], .38), center, .1),
    lerp(hand[5], hand[6], .3),
    lerp(hand[9], hand[10], .3),
    lerp(hand[13], hand[14], .28),
    lerp(hand[17], hand[18], .24),
    widen(hand[17], center, .16),
    widen(lerp(hand[17], hand[0], .5), center, .2),
  ];
  const contour = smoothClosed(controls);

  // Normal in a metric-consistent space: x and y already share units, and z is
  // lifted into them by zScale before the cross product.
  const a = {
    x: hand[5].x - hand[0].x,
    y: hand[5].y - hand[0].y,
    z: (hand[5].z - hand[0].z) * zScale,
  };
  const b = {
    x: hand[17].x - hand[0].x,
    y: hand[17].y - hand[0].y,
    z: (hand[17].z - hand[0].z) * zScale,
  };
  const normal = { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
  const length = Math.hypot(normal.x, normal.y, normal.z) || 1;
  normal.x /= length; normal.y /= length; normal.z /= length;
  // Screen y runs down and nearness runs toward the viewer, so a palm facing
  // the camera has to report a positive z whichever way the landmarks wound.
  if (normal.z < 0) { normal.x = -normal.x; normal.y = -normal.y; normal.z = -normal.z; }

  const up = { x: hand[9].x - hand[0].x, y: hand[9].y - hand[0].y, z: hand[9].z - hand[0].z };
  const upLength = Math.hypot(up.x, up.y, up.z) || 1;
  up.x /= upLength; up.y /= upLength; up.z /= upLength;

  const radius = Math.max(...contour.map(p => Math.hypot(p.x - center.x, p.y - center.y)));
  const disc = Math.PI * radius * radius;
  const openness = Math.min(Math.abs(signedArea(contour)) / 2 / Math.max(disc, 1e-6) * 1.9, 1);

  return { center, boundary, contour, normal, up, radius, openness };
}

/**
 * A ring inside the silhouette, offset along the contour's own inward normal
 * rather than scaled toward the centre. Scaling toward a point drives any
 * outline to a disc as it shrinks, which is why a contour-built palm can still
 * read as a circular sprite at its bright core; offsetting keeps the palm's
 * shape all the way in.
 *
 * Where the inset would carry a point past the centre -- the narrow neck by
 * the wrist -- it falls back to a blend toward the centre, so the ring stays
 * a simple closed loop instead of folding through itself.
 */
export function insetContour(palm: PalmFrame, inset: number): RigPoint[] {
  const contour = palm.contour;
  const n = contour.length;
  const winding = Math.sign(signedArea(contour)) || 1;
  const ring: RigPoint[] = Array.from({ length: n });
  for (let i = 0; i < n; i += 1) {
    const point = contour[i];
    const previous = contour[(i - 1 + n) % n];
    const next = contour[(i + 1) % n];
    const tx = next.x - previous.x;
    const ty = next.y - previous.y;
    const tangent = Math.hypot(tx, ty) || 1;
    // Inward normal. For a ring wound so that the signed area is positive, the
    // outward normal is (T.y, -T.x); the inward one is its negation, and the
    // winding term makes that hold whichever way the contour came out.
    const nx = (-ty / tangent) * winding;
    const ny = (tx / tangent) * winding;

    const reachX = point.x - palm.center.x;
    const reachY = point.y - palm.center.y;
    const reach = Math.hypot(reachX, reachY) || 1;
    const limit = reach * .82;
    const applied = Math.min(inset, limit);
    let x = point.x + nx * applied;
    let y = point.y + ny * applied;
    // If the offset overshot what this part of the outline could give, take
    // the remainder toward the centre instead of letting the loop cross over.
    if (inset > limit) {
      const remainder = Math.min((inset - limit) / Math.max(reach, 1e-3), 1);
      x += (palm.center.x - x) * remainder;
      y += (palm.center.y - y) * remainder;
    }
    ring[i] = { x, y, z: point.z + (palm.center.z - point.z) * Math.min(inset / Math.max(reach, 1e-3), 1) };
  }
  return ring;
}

/** Even-odd test against the palm silhouette, in screen space. */
export function containsPoint(palm: PalmFrame, x: number, y: number): boolean {
  const contour = palm.contour;
  let inside = false;
  for (let i = 0, j = contour.length - 1; i < contour.length; j = i, i += 1) {
    const a = contour[i];
    const b = contour[j];
    if ((a.y > y) === (b.y > y)) continue;
    if (x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** Same model for hands and head; approximate webcam intrinsics, not calibration. */
export function nearnessFromDistance(mm: number) {
  return Math.max(0, Math.min(1, 1 - (mm - 280) / 1320));
}

/** How many millimetres one unit of nearness spans; the inverse of the above. */
export const NEARNESS_SPAN_MM = 1320;

export function fieldDepthFromEndpoints(left: RigPoint, right: RigPoint) {
  return (left.z + right.z) * .5;
}
