import test from 'node:test';
import assert from 'node:assert/strict';
import { createFieldStateMachine } from '../lib/field-state.ts';
import { buildPalmFrame, nearnessFromDistance, fieldDepthFromEndpoints } from '../lib/palm-geometry.ts';
import { levelEnergy, positionSpread, momentumSpread, uncertaintyProduct, normalisedEnergy, relativeWellWidth } from '../lib/quantum.ts';

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
    assert.ok(uncertaintyProduct(.8, n) >= .5);
  }
  assert.equal(relativeWellWidth(0), 1);
  assert.equal(relativeWellWidth(1), .5);
  assert.equal(normalisedEnergy(0), 0);
  assert.equal(normalisedEnergy(1), 1);
});
