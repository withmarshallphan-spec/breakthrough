import test from 'node:test';
import assert from 'node:assert/strict';
import { createFieldStateMachine } from '../lib/field-state.ts';
import { buildPalmFrame, nearnessFromDistance, fieldDepthFromEndpoints, PALM_CONTOUR_POINTS } from '../lib/palm-geometry.ts';
import { levelEnergy, positionSpread, momentumSpread, uncertaintyProduct, GROUND_UNCERTAINTY, energyRatio, normalisedEnergy, relativeWellWidth } from '../lib/quantum.ts';

const sample = (now, hands, gapRatio, confinement = .9, nearClasp = true) => ({ now, hands, gapRatio, confinement, nearClasp });

test('clasp hysteresis, release and bounded hidden-hand memory', () => {
  const machine = createFieldStateMachine();
  assert.equal(machine.update(sample(0, 0, 99)).state, 'dormant');
  assert.equal(machine.update(sample(100, 2, 6, 0)).state, 'open');
  assert.equal(machine.update(sample(200, 2, 3, .6)).state, 'compressing');
  assert.equal(machine.update(sample(300, 2, 1)).state, 'clasped');
  assert.equal(machine.update(sample(400, 2, 1.4)).state, 'clasped');
  const merged = machine.update(sample(500, 1, 99));
  assert.equal(merged.state, 'clasped');
  assert.equal(merged.holdAnchors, true);
  assert.equal(machine.update(sample(1400, 1, 99)).state, 'clasped');
  assert.equal(machine.update(sample(1600, 1, 99)).state, 'release');
  assert.equal(machine.update(sample(2000, 1, 99, .1)).state, 'release');
  assert.equal(machine.update(sample(2600, 1, 99, .1)).state, 'open');
});

test('a lone pinch cannot acquire clasp; moving away breaks merged-hand hold', () => {
  const machine = createFieldStateMachine();
  assert.notEqual(machine.update(sample(100, 1, .1)).state, 'clasped');
  machine.update(sample(200, 2, 1));
  assert.equal(machine.update(sample(250, 1, 99, .8, false)).state, 'release');
});

test('opening a clasp fires one decaying release impulse', () => {
  const machine = createFieldStateMachine();
  assert.equal(machine.update(sample(0, 2, 6, 0)).pulse, 0);
  assert.equal(machine.update(sample(100, 2, 1)).pulse, 0, 'entering a clasp is not the event');
  const opened = machine.update(sample(116, 2, 1.7));
  assert.ok(opened.pulse > .9 && opened.pulse <= 1, `release fires, got ${opened.pulse}`);
  // Stepped at a real frame interval, because the impulse decays per frame.
  let last = opened.pulse;
  let now = 132;
  while (last > 0 && now < 6000) {
    const next = machine.update(sample(now, 2, 6, 0)).pulse;
    assert.ok(next < last, 'decays every frame rather than latching');
    last = next;
    now += 16;
  }
  assert.equal(last, 0);
  assert.ok(now - 116 < 3000, `settles in a couple of seconds, took ${now - 116}ms`);
});

test('brief total loss holds a clasp then becomes dormant', () => {
  const machine = createFieldStateMachine();
  machine.update(sample(100, 2, 1));
  assert.equal(machine.update(sample(200, 0, 99)).holdAnchors, true);
  assert.equal(machine.update(sample(350, 0, 99)).state, 'dormant');
});

test('opening after a clasp consistently enters release even without opening velocity', () => {
  const machine = createFieldStateMachine();
  machine.update(sample(100, 2, 1));
  assert.equal(machine.update(sample(200, 2, 1.6)).state, 'release');
  assert.equal(machine.update(sample(300, 2, 1.7)).state, 'release');
});

test('palm boundary follows wrist and MCPs under rotation and translation', () => {
  const hand = Array.from({ length: 21 }, (_, i) => ({ x: Math.cos(i) * 40, y: Math.sin(i) * 60, z: .55 + i * .001 }));
  const palm = buildPalmFrame(hand);
  assert.deepEqual(palm.boundary, [0, 1, 5, 9, 13, 17].map(i => hand[i]));
  const rotated = buildPalmFrame(hand.map(p => ({ x: -p.y + 700, y: p.x - 400, z: p.z })));
  assert.ok(Math.abs(rotated.center.x - (-palm.center.y + 700)) < 1e-10);
  assert.ok(Math.abs(rotated.center.y - (palm.center.x - 400)) < 1e-10);
  assert.ok(Math.abs(rotated.radius - palm.radius) < 1e-10);
  assert.ok(rotated.center.y < 0, 'palm coordinates are not clamped to a horizontal band');

  // The emission surface is the contour, so it has to rotate with the hand too.
  assert.equal(palm.contour.length, PALM_CONTOUR_POINTS);
  palm.contour.forEach((point, i) => {
    assert.ok(Math.abs(rotated.contour[i].x - (-point.y + 700)) < 1e-9);
    assert.ok(Math.abs(rotated.contour[i].y - (point.x - 400)) < 1e-9);
  });
});

test('the palm contour is a closed silhouette, not a disc around the centre', () => {
  // A plausible flat right palm: wrist at the base, knuckles in an arc above.
  const hand = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: .6 }));
  const put = (i, x, y) => { hand[i] = { x, y, z: .6 }; };
  put(0, 100, 200); put(1, 70, 175); put(2, 52, 148);
  put(5, 66, 110); put(6, 62, 82);
  put(9, 96, 100); put(10, 94, 70);
  put(13, 124, 106); put(14, 128, 78);
  put(17, 148, 124); put(18, 156, 98);
  const palm = buildPalmFrame(hand);

  let area = 0;
  let perimeter = 0;
  palm.contour.forEach((a, i) => {
    const b = palm.contour[(i + 1) % palm.contour.length];
    area += a.x * b.y - b.x * a.y;
    perimeter += Math.hypot(b.x - a.x, b.y - a.y);
  });
  area = Math.abs(area) / 2;
  assert.ok(area > 1500, `palm encloses real area, got ${area}`);
  // Isoperimetric ratio: 1 is a circle. A palm has to be measurably longer
  // than it is wide, or the surface is a disc again.
  const circularity = 4 * Math.PI * area / (perimeter * perimeter);
  assert.ok(circularity < .92, `contour is not a disc, got ${circularity}`);

  // Every sample sits at a different distance from the centre; a circular
  // fallback would make these all equal.
  const reach = palm.contour.map(p => Math.hypot(p.x - palm.center.x, p.y - palm.center.y));
  assert.ok(Math.max(...reach) / Math.min(...reach) > 1.5);

  // The wrist-to-knuckle axis points up the hand, away from the wrist.
  assert.ok(palm.up.y < -.5 && Math.abs(palm.up.x) < .5);
});

test('field depth follows both palms and can sort in front of or behind the head', () => {
  const p = mm => ({ x: 0, y: 0, z: nearnessFromDistance(mm) });
  const head = p(850).z;
  assert.ok(fieldDepthFromEndpoints(p(400), p(650)) > head);
  assert.ok(fieldDepthFromEndpoints(p(1100), p(1350)) < head);
  assert.notEqual(fieldDepthFromEndpoints(p(400), p(650)), .5);
});

test('fixed-mode box energy quadruples on halving width without increasing n', () => {
  for (const n of [1, 2, 6]) {
    assert.ok(Math.abs(levelEnergy(n, .8) / levelEnergy(n, 1.6) - 4) < 1e-12);
    assert.ok(Math.abs(positionSpread(.8, n) / positionSpread(1.6, n) - .5) < 1e-12);
    assert.ok(Math.abs(momentumSpread(.8, n) / momentumSpread(1.6, n) - 2) < 1e-12);
  }
  assert.equal(relativeWellWidth(0), 1);
  assert.equal(relativeWellWidth(1), .5);
  assert.equal(normalisedEnergy(0), 0);
  assert.equal(normalisedEnergy(1), 1);
});

test('internal energy ratio agrees with the ideal box', () => {
  // E/E0 = (L0/L)^2 internally; the UI shows only qualitative labels.
  for (const c of [0, .25, .5, .75, 1]) {
    const width = relativeWellWidth(c);
    assert.ok(Math.abs(energyRatio(c) - 1 / (width * width)) < 1e-12);
    assert.ok(Math.abs(energyRatio(c) - levelEnergy(1, width) / levelEnergy(1, 1)) < 1e-9);
  }
  assert.equal(energyRatio(0), 1);
  assert.ok(Math.abs(energyRatio(1) - 4) < 1e-12);
});

test('the uncertainty product is independent of the squeeze and stays above hbar/2', () => {
  for (const n of [1, 2, 6]) {
    // dx dp has no L in it: the gesture cannot push it toward the bound.
    for (const widthNm of [.4, 1, 3.7]) {
      const product = positionSpread(widthNm, n) * 1e-9 * momentumSpread(widthNm, n);
      assert.ok(Math.abs(product / 1.054571817e-34 - uncertaintyProduct(n)) < 1e-6);
    }
    assert.ok(uncertaintyProduct(n) > .5, 'never dips below the uncertainty bound');
  }
  assert.ok(Math.abs(GROUND_UNCERTAINTY - .5678) < 5e-4);
});
