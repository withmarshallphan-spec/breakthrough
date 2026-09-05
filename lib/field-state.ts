/** Gesture states are presentation direction, not quantum phase transitions. */
/**
 * `critical` is the state just short of a clasp: palms very close, the well
 * narrow, nothing sealed yet. It exists as its own state because it is where
 * the image is most legible -- narrow, intricate and bright -- and because
 * lumping it in with `compressing` gave the renderer no way to know it had
 * arrived there.
 */
export type FieldState = 'dormant' | 'open' | 'compressing' | 'critical' | 'clasped' | 'release';
export type GestureSample = {
  now: number;
  hands: number;
  gapRatio: number;
  confinement: number;
  /** One visible hand remains close to the last two-hand clasp. */
  nearClasp: boolean;
  /**
   * Evidence beyond the gap, in -1..1, that the hands are actually interlocked
   * rather than merely overlapping in projection: silhouettes collapsing,
   * fingertips inside the other palm, both palms at a comparable distance.
   * It widens or narrows the gap threshold rather than replacing it, so the
   * gesture still reads the same when no extra evidence is available.
   */
  evidence?: number;
};

export function createFieldStateMachine() {
  let state: FieldState = 'dormant';
  let lastPair = -Infinity;
  let lastClaspEvidence = -Infinity;
  let releaseUntil = 0;
  let lastNow = 0;
  let lastConfinement = 0;
  let seal = 0;
  let pulse = 0;
  return {
    update(sample: GestureSample) {
      const { now, hands, gapRatio, confinement, nearClasp, evidence = 0 } = sample;
      const dt = Math.min(Math.max((now - lastNow) / 1000, 1 / 120), .1);
      const speed = (confinement - lastConfinement) / dt;
      const wasClasped = state === 'clasped';
      if (hands === 2) lastPair = now;
      // A clasp is easier to reach than it was, and just as sticky once held.
      // Palms roughly one and a half palm-widths apart already read as closed;
      // the hysteresis band above keeps it from chattering back out.
      const threshold = (wasClasped ? 1.95 : 1.45)
        * (1 + Math.max(-1, Math.min(1, evidence)) * .45);
      const pairClasp = hands === 2 && gapRatio < threshold;
      if (pairClasp) lastClaspEvidence = now;
      // Bounded memory prevents one dropped detection from drawing a pinch
      // bridge through a clasp, without locking a lone hand in this state.
      const merged = hands === 1 && nearClasp && wasClasped && now - lastPair < 1100;
      const briefLoss = hands === 0 && wasClasped && now - lastClaspEvidence < 220;
      if (pairClasp || merged || briefLoss) state = 'clasped';
      else if (hands === 0) state = 'dormant';
      else if (wasClasped) { state = 'release'; releaseUntil = now + 750; pulse = 1; }
      else if (now < releaseUntil) state = 'release';
      // Hysteresis on the way in and out, so a hand hovering at the boundary
      // does not flicker between the two treatments.
      else if (confinement > (state === 'critical' ? .62 : .74)) state = 'critical';
      else if (speed > .035 || (confinement > .18 && speed > -.045)) state = 'compressing';
      else state = 'open';
      // The approach envelope starts compressing the image well before the
      // hands actually meet, so closing reads as a continuous squeeze rather
      // than a switch that flips at the end.
      const approach = hands === 2 ? .78 * Math.max(0, Math.min(1, (2.4 - gapRatio) / 1.0)) : 0;
      const targetSeal = state === 'clasped' ? 1 : approach;
      seal += (targetSeal - seal) * (1 - Math.exp(-dt / (targetSeal ? .055 : .2)));
      // One decaying impulse at the instant a clasp opens. The renderer spends
      // it on a brief expansion and bloom, so release is an event rather than a
      // fade; it carries no physical meaning of its own.
      pulse *= Math.exp(-dt / .32);
      if (pulse < 1e-3) pulse = 0;
      lastNow = now;
      lastConfinement = confinement;
      return { state, seal, pulse, holdAnchors: merged || briefLoss };
    },
  };
}
