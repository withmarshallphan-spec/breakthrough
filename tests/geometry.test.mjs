import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPalmFrame, insetContour, containsPoint, NEARNESS_SPAN_MM } from '../lib/palm-geometry.ts';
import { trianglesFromTesselation, distanceFromMatrix, buildFaceFrame } from '../lib/face-geometry.ts';
import { fitAffine, applyFit } from '../lib/depth-field.ts';
import { createFieldStateMachine } from '../lib/field-state.ts';
import { momentumRatio, energyRatio, relativeWellWidth } from '../lib/quantum.ts';

/** A flat right hand facing the camera, in screen pixels with nearness. */
function flatHand({ tiltZ = 0, x = 400, y = 300, size = 120 } = {}) {
  const grid = [
    [0, .5, 1], [1, .18, .82], [2, .06, .66], [3, 0, .54], [4, -.04, .44],
    [5, .18, .3], [6, .16, .16], [7, .15, .07], [8, .14, 0],
    [9, .42, .27], [10, .42, .12], [11, .42, .03], [12, .42, -.05],
    [13, .66, .3], [14, .68, .16], [15, .69, .07], [16, .7, -.01],
    [17, .86, .36], [18, .92, .24], [19, .95, .16], [20, .98, .09],
  ];
  return grid.map(([, u, v]) => ({
    x: x + (u - .5) * size,
    y: y + (v - .5) * size,
    // Tilting about the vertical axis makes one edge of the palm nearer.
    z: .5 + (u - .5) * tiltZ,
  }));
}

test('the palm normal responds to tilt only when z is given real units', () => {
  const tilted = flatHand({ tiltZ: .04 });
  // Nearness spans over a metre, so without a scale the cross product is all
  // screen-space and every palm reports as square to the camera.
  const unscaled = buildPalmFrame(tilted, 0);
  assert.ok(Math.abs(unscaled.normal.z) > .999, `flat without scale, got ${unscaled.normal.z}`);

  // With pixels-per-unit-nearness the same landmarks describe a tilted plane.
  const scale = NEARNESS_SPAN_MM * 2.2;
  const scaled = buildPalmFrame(tilted, scale);
  assert.ok(Math.abs(scaled.normal.x) > .2, `tilt is visible, got x=${scaled.normal.x}`);
  assert.ok(scaled.normal.z > 0, 'a palm toward the camera reports positive z');
  const length = Math.hypot(scaled.normal.x, scaled.normal.y, scaled.normal.z);
  assert.ok(Math.abs(length - 1) < 1e-9, 'the normal stays a unit vector');

  // An untilted palm of the same hand is square to the camera at any scale.
  const flat = buildPalmFrame(flatHand({ tiltZ: 0 }), scale);
  assert.ok(Math.abs(flat.normal.z) > .99, `no tilt, no lean: ${flat.normal.z}`);
});

test('the palm normal flips with the tilt direction', () => {
  const scale = NEARNESS_SPAN_MM * 2.2;
  const left = buildPalmFrame(flatHand({ tiltZ: .04 }), scale).normal;
  const right = buildPalmFrame(flatHand({ tiltZ: -.04 }), scale).normal;
  assert.ok(left.x * right.x < 0, `opposite tilts lean opposite ways: ${left.x} vs ${right.x}`);
});

test('inset rings keep the palm shape instead of collapsing to a disc', () => {
  const palm = buildPalmFrame(flatHand(), NEARNESS_SPAN_MM);
  const ring = insetContour(palm, palm.radius * .5);
  assert.equal(ring.length, palm.contour.length, 'the ring keeps its winding');

  const radii = ring.map(p => Math.hypot(p.x - palm.center.x, p.y - palm.center.y));
  const min = Math.min(...radii);
  const max = Math.max(...radii);
  // A disc would have every sample at the same radius. A palm does not.
  assert.ok(max / Math.max(min, 1e-6) > 1.35, `the inner ring is not round: ${max / min}`);
  // And it stays inside the outline it came from.
  const outer = palm.contour.map(p => Math.hypot(p.x - palm.center.x, p.y - palm.center.y));
  assert.ok(max <= Math.max(...outer) + 1e-6, 'the ring never leaves the silhouette');

  // Scaling toward the centre, which is what this replaces, is a similarity
  // transform: it preserves the ratio exactly and so keeps no extra shape.
  const scaled = palm.contour.map(p => Math.hypot(p.x - palm.center.x, p.y - palm.center.y) * .5);
  const scaledRatio = Math.max(...scaled) / Math.min(...scaled);
  const outerRatio = Math.max(...outer) / Math.min(...outer);
  assert.ok(Math.abs(scaledRatio - outerRatio) < 1e-9, 'scaling cannot change the outline');
});

test('point containment finds fingertips inside the other palm', () => {
  const palm = buildPalmFrame(flatHand(), NEARNESS_SPAN_MM);
  assert.equal(containsPoint(palm, palm.center.x, palm.center.y), true);
  assert.equal(containsPoint(palm, palm.center.x + palm.radius * 4, palm.center.y), false);
});

test('a curled palm reads as less open than a flat one', () => {
  const flat = buildPalmFrame(flatHand(), NEARNESS_SPAN_MM);
  // Collapse the hand toward its own centre: a fist, in projection.
  const curled = flatHand().map((p, i, all) => {
    const cx = all.reduce((sum, q) => sum + q.x / all.length, 0);
    return { x: cx + (p.x - cx) * .12, y: p.y, z: p.z };
  });
  assert.ok(buildPalmFrame(curled, NEARNESS_SPAN_MM).openness < flat.openness * .6);
});

test('the face tesselation decomposes into closed triangles', () => {
  // Three triangles as MediaPipe stores them: consecutive edge triples.
  const connections = [
    { start: 0, end: 1 }, { start: 1, end: 2 }, { start: 2, end: 0 },
    { start: 2, end: 1 }, { start: 1, end: 3 }, { start: 3, end: 2 },
    // A group that does not close is dropped rather than trusted.
    { start: 4, end: 5 }, { start: 6, end: 7 }, { start: 7, end: 4 },
  ];
  const triangles = trianglesFromTesselation(connections);
  assert.deepEqual(Array.from(triangles), [0, 1, 2, 2, 1, 3]);
});

test('head distance comes from the transformation matrix, and refuses nonsense', () => {
  // Column-major, translation in the fourth column, canonical units are cm.
  const at = (x, y, z) => ({ data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1] });
  assert.ok(Math.abs(distanceFromMatrix(at(0, 0, -60)) - 600) < 1e-9, '60 cm reads as 600 mm');
  assert.equal(distanceFromMatrix(at(0, 0, -2)), null, 'a head 2 cm away is a failed fit');
  assert.equal(distanceFromMatrix(at(0, 0, -900)), null, 'nor is it 9 m away');
  assert.equal(distanceFromMatrix(undefined), null);
  assert.equal(distanceFromMatrix({ data: [1, 2, 3] }), null);
});

test('the face frame prefers the matrix and falls back to apparent width', () => {
  const landmarks = Array.from({ length: 468 }, (_, i) => ({
    x: .5 + Math.cos(i) * .08,
    y: .5 + Math.sin(i) * .1,
    z: Math.cos(i * .7) * .02,
  }));
  const toScreen = (p) => ({ x: p.x * 1280, y: p.y * 720 });
  const apparent = () => 700;

  const withMatrix = buildFaceFrame(
    landmarks,
    { data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -55, 1] },
    toScreen, apparent,
  );
  assert.equal(withMatrix.source, 'matrix');
  assert.ok(Math.abs(withMatrix.distanceMm - 550) < 1e-6);

  const without = buildFaceFrame(landmarks, undefined, toScreen, apparent);
  assert.equal(without.source, 'apparent');
  assert.equal(without.distanceMm, 700);

  // Per-vertex nearness varies with the landmarks' own relief.
  const depths = Array.from({ length: withMatrix.count }, (_, i) => withMatrix.vertices[i * 3 + 2]);
  assert.ok(Math.max(...depths) - Math.min(...depths) > 1e-4, 'the head is not a plane');
  assert.equal(buildFaceFrame(landmarks.slice(0, 100), undefined, toScreen, apparent), null);
});

test('affine alignment recovers a known scale and shift, and rejects a bad fit', () => {
  const width = 32;
  const height = 32;
  const data = new Float32Array(width * height);
  // A horizontal disparity ramp. Samples land on exact pixel centres, so the
  // test measures the fit rather than the sampler's rounding.
  for (let i = 0; i < data.length; i += 1) data[i] = (i % width) / (width - 1);
  const raw = { data, width, height, cost: 0 };

  const scale = .4;
  const shift = .25;
  const samples = [];
  for (let x = 0; x < width; x += 1) {
    const u = x / (width - 1);
    samples.push({ u, v: .5, near: scale * u + shift });
  }
  const fit = fitAffine(raw, samples);
  assert.ok(fit, 'a clean fit succeeds');
  assert.ok(Math.abs(fit.scale - scale) < 1e-6, `scale ${fit.scale}`);
  assert.ok(Math.abs(fit.shift - shift) < 1e-6, `shift ${fit.shift}`);
  assert.ok(fit.residual < 1e-6);

  const into = new Uint8Array(width * height);
  applyFit(raw, fit, into);
  // The byte map truncates, so agreement is to within one least-significant
  // step -- about 5 mm of the nearness range it encodes.
  assert.ok(Math.abs(into[0] - shift * 255) <= 1, `first sample ${into[0]}`);
  // At the far end of the ramp the disparity is 1, so the mapped nearness is
  // scale + shift.
  assert.ok(Math.abs(into[width - 1] - (scale + shift) * 255) <= 1, `last sample ${into[width - 1]}`);

  // Disparity rises with nearness, so a negative slope is a failure, not a scene.
  const inverted = samples.map(s => ({ ...s, near: 1 - s.near }));
  assert.equal(fitAffine(raw, inverted), null, 'a negative scale is rejected');
  assert.equal(fitAffine(raw, samples.slice(0, 6)), null, 'too few samples is rejected');
});

test('alignment survives a minority of badly placed samples', () => {
  const width = 64;
  const data = new Float32Array(width * 4);
  for (let i = 0; i < data.length; i += 1) data[i] = (i % width) / (width - 1);
  const raw = { data, width, height: 4, cost: 0 };
  const samples = [];
  for (let x = 0; x < width; x += 1) {
    const u = x / (width - 1);
    // A tenth of the samples are wrong by a long way, as a hand crossing a
    // face produces. The reweighted pass should discard them.
    const outlier = x % 10 === 0;
    samples.push({ u, v: .5, near: outlier ? .95 : .5 * u + .2 });
  }
  const fit = fitAffine(raw, samples);
  assert.ok(fit, 'still fits');
  assert.ok(Math.abs(fit.scale - .5) < .06, `scale held near .5, got ${fit.scale}`);
});

test('clasp evidence widens the gate without replacing it', () => {
  const sample = (now, gapRatio, evidence) =>
    ({ now, hands: 2, gapRatio, confinement: .9, nearClasp: true, evidence });

  // A gap that would not clasp on its own does when the hands are interlocked.
  const withEvidence = createFieldStateMachine();
  withEvidence.update(sample(0, 6, 0));
  assert.equal(withEvidence.update(sample(100, 1.8, .8)).state, 'clasped');

  // The same gap with no supporting evidence does not.
  const without = createFieldStateMachine();
  without.update(sample(0, 6, 0));
  assert.notEqual(without.update(sample(100, 1.8, 0)).state, 'clasped');

  // And evidence against -- palms far apart in depth -- narrows the gate,
  // below even a gap that would otherwise have clasped comfortably.
  const against = createFieldStateMachine();
  against.update(sample(0, 6, 0));
  assert.notEqual(against.update(sample(100, 1.0, -.8)).state, 'clasped');
});

test('momentum spread is the square root of the energy rise, exactly', () => {
  for (const confinement of [0, .25, .5, .75, 1]) {
    const dp = momentumRatio(confinement);
    assert.ok(Math.abs(dp - 1 / relativeWellWidth(confinement)) < 1e-12, 'dp/dp0 is L0/L');
    assert.ok(Math.abs(dp * dp - energyRatio(confinement)) < 1e-12, 'E/E0 is (dp/dp0)^2');
  }
  assert.equal(momentumRatio(0), 1);
  assert.ok(Math.abs(momentumRatio(1) - 2) < 1e-12, 'halving the width doubles the spread');
});

test('the phase wheel is a full circle at constant luminance', async () => {
  const { phaseColour, luminance } = await import('../lib/field-palette.ts');

  const samples = Array.from({ length: 64 }, (_, i) => phaseColour(i / 64 * Math.PI * 2));

  // Constant luminance all the way round. This is the property that keeps the
  // colour from silently restating |psi|^2: brightness carries the density, and
  // a raw hue wheel would swing by roughly a factor of two between yellow and
  // blue on top of it.
  for (const colour of samples) {
    assert.ok(Math.abs(luminance(colour) - 1) < 1e-9, `luma ${luminance(colour)}`);
  }

  // A full circle, not a chord: the dominant channel has to visit all three.
  const dominant = new Set(samples.map((c) => {
    if (c.r >= c.g && c.r >= c.b) return 'r';
    return c.g >= c.b ? 'g' : 'b';
  }));
  assert.equal(dominant.size, 3, 'every hue region is reached');

  // And it is one-to-one: no two phases half a turn apart share a colour, which
  // is exactly what the old two-pole mix got wrong.
  const distance = (a, b) => Math.hypot(a.r - b.r, a.g - b.g, a.b - b.b);
  for (let i = 0; i < 32; i += 1) {
    const here = phaseColour(i / 64 * Math.PI * 2);
    const opposite = phaseColour(i / 64 * Math.PI * 2 + Math.PI);
    assert.ok(distance(here, opposite) > .35, `opposed phases differ, got ${distance(here, opposite)}`);
  }

  // Chroma pulls the whole wheel toward white without breaking either property.
  const washed = phaseColour(1.2, .2);
  const full = phaseColour(1.2, 1);
  assert.ok(Math.abs(luminance(washed) - 1) < 1e-9);
  const spread = (c) => Math.max(c.r, c.g, c.b) - Math.min(c.r, c.g, c.b);
  assert.ok(spread(washed) < spread(full) * .4);
});

test('emission level rises with confinement while its hue stays free', async () => {
  const { emissionFor, luminance } = await import('../lib/field-palette.ts');
  const open = emissionFor(0, 0);
  const confined = emissionFor(1, 0);
  // Level is decided in one place, so every gain downstream stays calibrated.
  assert.ok(Math.abs(luminance(open) - .28) < 1e-9);
  assert.ok(Math.abs(luminance(confined) - .42) < 1e-9);
  // A broad state spills blue; a confined one is close to neutral.
  assert.ok(open.b / open.r > 1.4, 'open leans ice-blue');
  assert.ok(Math.abs(confined.b / confined.r - 1) < .25, 'confined is near neutral');
});
