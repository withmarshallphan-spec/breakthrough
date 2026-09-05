/** Gesture states are presentation direction, not quantum phase transitions. */
export type FieldState = 'dormant' | 'open' | 'compressing' | 'clasped' | 'release';
export type GestureSample = {
  now: number;
  hands: number;
  gapRatio: number;
  confinement: number;
  /** One visible hand remains close to the last two-hand clasp. */
  nearClasp: boolean;
};

export function createFieldStateMachine() {
  let state: FieldState = 'dormant';
  let lastPair = -Infinity;
  let lastClaspEvidence = -Infinity;
  let releaseUntil = 0;
  let lastNow = 0;
  let lastConfinement = 0;
  let seal = 0;
  return {
    update(sample: GestureSample) {
      const { now, hands, gapRatio, confinement, nearClasp } = sample;
      const dt = Math.min(Math.max((now - lastNow) / 1000, 1 / 120), .1);
      const speed = (confinement - lastConfinement) / dt;
      const wasClasped = state === 'clasped';
      if (hands === 2) lastPair = now;
      const pairClasp = hands === 2 && gapRatio < (wasClasped ? 1.55 : 1.12);
      if (pairClasp) lastClaspEvidence = now;
      // Bounded memory prevents one dropped detection from drawing a pinch
      // bridge through a clasp, without locking a lone hand in this state.
      const merged = hands === 1 && nearClasp && wasClasped && now - lastPair < 1100;
      const briefLoss = hands === 0 && wasClasped && now - lastClaspEvidence < 220;
      if (pairClasp || merged || briefLoss) state = 'clasped';
      else if (hands === 0) state = 'dormant';
      else if (wasClasped) { state = 'release'; releaseUntil = now + 750; }
      else if (now < releaseUntil) state = 'release';
      else if (speed > .055 || (confinement > .36 && speed > -.055)) state = 'compressing';
      else state = 'open';
      const approach = hands === 2 ? .7 * Math.max(0, Math.min(1, (1.6 - gapRatio) / .48)) : 0;
      const targetSeal = state === 'clasped' ? 1 : approach;
      seal += (targetSeal - seal) * (1 - Math.exp(-dt / (targetSeal ? .09 : .24)));
      lastNow = now;
      lastConfinement = confinement;
      return { state, seal, holdAnchors: merged || briefLoss };
    },
  };
}
