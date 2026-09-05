/**
 * The tracker's one visible mark: a dot on the tracked palm and a hairline
 * leader running from it to the text block that describes the state. It is
 * instrumentation in the sense a plate caption is -- it says where the reading
 * was taken -- not a debug skeleton. No joints, no bones, no vertical rules.
 */
export type GuidePoint = { x: number; y: number };

export type GuideLine = {
  /** Called once per tracked frame with the palm dot and where the text sits. */
  update: (anchor: GuidePoint | null, target: GuidePoint | null, confidence: number, now: number) => void;
  /** Draws into an already-cleared 2D context, in CSS pixels. */
  draw: (ctx: CanvasRenderingContext2D, opacity: number) => void;
  reset: () => void;
};

/**
 * Time constant of the follower, in seconds. Long enough that the dot reads as
 * drawn rather than as a cursor chasing the hand -- the landmarks underneath
 * are already filtered, and the remaining impression of jitter comes from the
 * mark tracking them too faithfully.
 */
const FOLLOW_TAU = .115;
/** Length of the horizontal run into the text, in CSS pixels. */
const TAIL = 14;

export function createGuideLine(): GuideLine {
  let dotX = 0;
  let dotY = 0;
  let endX = 0;
  let endY = 0;
  let live = false;
  let presence = 0;
  let smoothConfidence = 1;
  let last = 0;

  return {
    update(anchor, target, confidence, now) {
      const dt = Math.min(Math.max((now - last) / 1000, 1 / 240), .1);
      last = now;
      const ease = (tau: number) => 1 - Math.exp(-dt / tau);

      if (anchor && target) {
        if (!live) {
          dotX = anchor.x; dotY = anchor.y;
          endX = target.x; endY = target.y;
          live = true;
        }
        dotX += (anchor.x - dotX) * ease(FOLLOW_TAU);
        dotY += (anchor.y - dotY) * ease(FOLLOW_TAU);
        // The text end is smoothed harder: the block itself is placed in whole
        // pixels and steps as it clamps to the viewport, and the line should
        // not inherit that step.
        endX += (target.x - endX) * ease(.16);
        endY += (target.y - endY) * ease(.16);
      } else {
        live = false;
      }
      presence += ((anchor && target ? 1 : 0) - presence) * ease(anchor ? .12 : .2);
      smoothConfidence += (confidence - smoothConfidence) * ease(.25);
    },

    draw(ctx, opacity) {
      const alpha = opacity * presence;
      if (alpha < .01 || !live) return;

      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      // Hairline: one device pixel at any ratio, because the context is
      // already scaled by devicePixelRatio.
      ctx.lineWidth = Math.max(1 / (ctx.getTransform().a || 1), .5);

      // The leader leaves the dot, sags gently, and arrives horizontally at the
      // text. The horizontal run is what makes it read as a caption rule rather
      // than a pointer, and the curve is what keeps it from being a vertical.
      const elbowX = endX + (endX >= dotX ? -TAIL : TAIL);
      ctx.strokeStyle = `rgba(238, 244, 248, ${(alpha * .3).toFixed(4)})`;
      ctx.beginPath();
      ctx.moveTo(dotX, dotY);
      ctx.quadraticCurveTo((dotX + elbowX) / 2, dotY + (endY - dotY) * .16, elbowX, endY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Terminal dot on the hand. Confident tracking closes it to a point; as
      // confidence falls it opens into a faint ring, which is the only reaction
      // the mark makes to the tracker's own certainty.
      const uncertainty = 1 - Math.min(Math.max(smoothConfidence, 0), 1);
      ctx.strokeStyle = `rgba(244, 248, 251, ${(alpha * (.42 - uncertainty * .18)).toFixed(4)})`;
      ctx.beginPath();
      ctx.arc(dotX, dotY, 1.4 + uncertainty * 4.6, 0, Math.PI * 2);
      ctx.stroke();

      if (uncertainty < .5) {
        ctx.fillStyle = `rgba(255, 255, 255, ${(alpha * (.5 - uncertainty) * 1.1).toFixed(4)})`;
        ctx.beginPath();
        ctx.arc(dotX, dotY, 1.3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    },

    reset() {
      live = false;
      presence = 0;
    },
  };
}
