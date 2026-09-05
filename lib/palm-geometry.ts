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
  normal: RigPoint;
  /** Wrist to middle knuckle: the palm's own long axis. */
  up: RigPoint;
  radius: number;
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

/**
 * The palm surface, from the landmarks alone. The MCP knuckles are carried a
 * short way toward their own PIP joints so the surface reaches into the roots
 * of the fingers, and the thumb and little-finger edges are widened over the
 * thenar and hypothenar muscle, which is where a palm is broadest. Every term
 * is a linear combination of landmark positions, so the shape stays correct
 * under rotation, tilt and perspective foreshortening for free.
 */
export function buildPalmFrame(hand: RigPoint[]): PalmFrame {
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

  const a = { x: hand[5].x - hand[0].x, y: hand[5].y - hand[0].y, z: hand[5].z - hand[0].z };
  const b = { x: hand[17].x - hand[0].x, y: hand[17].y - hand[0].y, z: hand[17].z - hand[0].z };
  const normal = { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
  const length = Math.hypot(normal.x, normal.y, normal.z) || 1;
  normal.x /= length; normal.y /= length; normal.z /= length;

  const up = { x: hand[9].x - hand[0].x, y: hand[9].y - hand[0].y, z: hand[9].z - hand[0].z };
  const upLength = Math.hypot(up.x, up.y, up.z) || 1;
  up.x /= upLength; up.y /= upLength; up.z /= upLength;

  const radius = Math.max(...contour.map(p => Math.hypot(p.x - center.x, p.y - center.y)));
  return { center, boundary, contour, normal, up, radius };
}

/** Same model for hands and head; approximate webcam intrinsics, not calibration. */
export function nearnessFromDistance(mm: number) {
  return Math.max(0, Math.min(1, 1 - (mm - 280) / 1320));
}

export function fieldDepthFromEndpoints(left: RigPoint, right: RigPoint) {
  return (left.z + right.z) * .5;
}
