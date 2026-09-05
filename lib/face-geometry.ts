import { nearnessFromDistance, type RigPoint } from './palm-geometry';

export type Landmark = { x: number; y: number; z?: number };

/** Cheekbone to cheekbone on the canonical face, used only as a fallback. */
export const FACE_WIDTH_MM = 145;

const FACE_TOP = 10;
const FACE_CHIN = 152;
const FACE_LEFT = 234;
const FACE_RIGHT = 454;
export const FACE_NOSE_TIP = 1;
export const FACE_FOREHEAD = 151;

/**
 * The head as actual geometry rather than an ellipsoid: every landmark with a
 * screen position and its own nearness, plus the triangle list that turns them
 * into a surface. Triangle normals are what let a cheekbone catch light that
 * the plane beside it does not.
 */
export type FaceFrame = {
  /** 3 floats per landmark: screen x, screen y, nearness (0 far, 1 near). */
  vertices: Float32Array;
  count: number;
  center: { x: number; y: number };
  radiusX: number;
  radiusY: number;
  /** Nearness of the head as a whole. */
  depth: number;
  /** Distance in millimetres, before the nearness curve. */
  distanceMm: number;
  /** Which estimator produced `distanceMm`. */
  source: 'matrix' | 'apparent';
  /** Unit head-forward in screen convention (x right, y down, z toward view). */
  forward: RigPoint;
  nose: RigPoint;
};

/**
 * MediaPipe documents FACE_LANDMARKS_TESSELATION as a list of connections, but
 * it is stored as consecutive edge triples, each triple being one triangle:
 * [a,b], [b,c], [c,a]. Verified against the shipped bundle -- 2556 edges group
 * into exactly 852 triangles with no leftovers, covering all 468 vertices.
 * Groups that do not close are skipped rather than trusted, so a change to the
 * constant upstream degrades the mesh instead of corrupting it.
 */
export function trianglesFromTesselation(connections: readonly { start: number; end: number }[]): Uint16Array {
  const indices: number[] = [];
  for (let i = 0; i + 2 < connections.length; i += 3) {
    const [ab, bc, ca] = [connections[i], connections[i + 1], connections[i + 2]];
    if (ab.end !== bc.start || bc.end !== ca.start || ca.end !== ab.start) continue;
    indices.push(ab.start, bc.start, ca.start);
  }
  return new Uint16Array(indices);
}

/**
 * Head distance from the facial transformation matrix. MediaPipe fits it by
 * Procrustes analysis from the canonical face model, whose metric unit is the
 * centimetre, so the translation column is a real distance -- and unlike
 * apparent cheek width it is not shortened by head yaw.
 * Returns null when the matrix is missing or implausible.
 */
export function distanceFromMatrix(matrix: { data: number[] } | undefined): number | null {
  const m = matrix?.data;
  if (!m || m.length < 16) return null;
  // Column-major: the translation is the fourth column.
  const cm = Math.hypot(m[12], m[13], m[14]);
  if (!Number.isFinite(cm)) return null;
  const mm = cm * 10;
  // A head closer than 15 cm or further than 3 m is a failed fit, not a pose.
  return mm > 150 && mm < 3000 ? mm : null;
}

type ToScreen = (point: Landmark) => { x: number; y: number };

/**
 * Screen positions plus a per-vertex nearness. Landmark z is relative to the
 * head's own origin and scaled like x, so it is converted to millimetres
 * against the face's own apparent width before being added to the head
 * distance. The result is one depth scale shared with the hands.
 */
export function buildFaceFrame(
  landmarks: Landmark[],
  matrix: { data: number[] } | undefined,
  toScreen: ToScreen,
  focalDistanceMm: (realMm: number, apparentPx: number) => number,
  into?: Float32Array,
): FaceFrame | null {
  if (!landmarks || landmarks.length < 468) return null;

  const top = toScreen(landmarks[FACE_TOP]);
  const chin = toScreen(landmarks[FACE_CHIN]);
  const left = toScreen(landmarks[FACE_LEFT]);
  const right = toScreen(landmarks[FACE_RIGHT]);

  const center = { x: (top.x + chin.x + left.x + right.x) / 4, y: (top.y + chin.y + left.y + right.y) / 4 };
  const radiusX = Math.max(Math.hypot(right.x - left.x, right.y - left.y) / 2, 8);
  const radiusY = Math.max(Math.hypot(top.x - chin.x, top.y - chin.y) / 2, 8);

  const matrixMm = distanceFromMatrix(matrix);
  const distanceMm = matrixMm ?? focalDistanceMm(FACE_WIDTH_MM, radiusX * 2);
  const source: FaceFrame['source'] = matrixMm === null ? 'apparent' : 'matrix';

  // Normalised width across the cheeks, in the landmarks' own units, so the
  // per-vertex z spread can be given a physical size.
  const spanNorm = Math.max(Math.hypot(
    landmarks[FACE_RIGHT].x - landmarks[FACE_LEFT].x,
    landmarks[FACE_RIGHT].y - landmarks[FACE_LEFT].y,
  ), 1e-4);
  let originZ = 0;
  for (const id of [FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_CHIN]) originZ += (landmarks[id].z ?? 0) / 4;

  const count = landmarks.length;
  const vertices = into && into.length >= count * 3 ? into : new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const point = landmarks[i];
    const screen = toScreen(point);
    const localMm = ((point.z ?? 0) - originZ) / spanNorm * FACE_WIDTH_MM;
    vertices[i * 3] = screen.x;
    vertices[i * 3 + 1] = screen.y;
    vertices[i * 3 + 2] = nearnessFromDistance(distanceMm + localMm);
  }

  // Head forward from the matrix when it exists: the third column, with y and
  // z flipped into the screen convention. Otherwise from the landmark frame.
  let forward: RigPoint;
  const m = matrix?.data;
  if (m && m.length >= 16) {
    forward = { x: m[8], y: -m[9], z: m[10] };
  } else {
    const across = { x: right.x - left.x, y: right.y - left.y, z: 0 };
    const up = { x: top.x - chin.x, y: top.y - chin.y, z: 0 };
    forward = {
      x: across.y * up.z - across.z * up.y,
      y: across.z * up.x - across.x * up.z,
      z: across.x * up.y - across.y * up.x,
    };
  }
  const length = Math.hypot(forward.x, forward.y, forward.z) || 1;
  forward = { x: forward.x / length, y: forward.y / length, z: forward.z / length };

  const noseIndex = FACE_NOSE_TIP * 3;
  return {
    vertices,
    count,
    center,
    radiusX,
    radiusY,
    depth: nearnessFromDistance(distanceMm),
    distanceMm,
    source,
    forward,
    nose: { x: vertices[noseIndex], y: vertices[noseIndex + 1], z: vertices[noseIndex + 2] },
  };
}
