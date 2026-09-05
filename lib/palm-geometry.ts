export type RigPoint = { x: number; y: number; z: number };
export const PALM_BOUNDARY = [0, 1, 5, 9, 13, 17] as const;
export type PalmFrame = {
  center: RigPoint;
  boundary: RigPoint[];
  normal: RigPoint;
  radius: number;
};

/** The silhouette follows the wrist, thumb base and MCP arc, in winding order. */
export function buildPalmFrame(hand: RigPoint[]): PalmFrame {
  const boundary = PALM_BOUNDARY.map(i => ({ ...hand[i] }));
  const center = [0, 5, 9, 13, 17].reduce((p, i) => ({
    x: p.x + hand[i].x / 5, y: p.y + hand[i].y / 5, z: p.z + hand[i].z / 5,
  }), { x: 0, y: 0, z: 0 });
  const a = { x: hand[5].x - hand[0].x, y: hand[5].y - hand[0].y, z: hand[5].z - hand[0].z };
  const b = { x: hand[17].x - hand[0].x, y: hand[17].y - hand[0].y, z: hand[17].z - hand[0].z };
  const normal = { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
  const length = Math.hypot(normal.x, normal.y, normal.z) || 1;
  normal.x /= length; normal.y /= length; normal.z /= length;
  const radius = Math.max(...boundary.map(p => Math.hypot(p.x - center.x, p.y - center.y)));
  return { center, boundary, normal, radius };
}

/** Same model for hands and head; approximate webcam intrinsics, not calibration. */
export function nearnessFromDistance(mm: number) {
  return Math.max(0, Math.min(1, 1 - (mm - 280) / 1320));
}

export function fieldDepthFromEndpoints(left: RigPoint, right: RigPoint) {
  return (left.z + right.z) * .5;
}
