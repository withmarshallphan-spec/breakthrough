/**
 * The field's colour, kept out of the renderer so its properties can be
 * checked rather than eyeballed.
 *
 * Two rules, and they are the reason the colour survives to the screen:
 *
 *  - The phase poles are normalised to **equal luminance**. Brightness already
 *    carries |psi|^2; if the poles differed in luminance, turning through the
 *    phase would also change the apparent density, and the two encodings would
 *    be reading each other's channel.
 *  - The emission level is applied **separately** from the hue. The gains
 *    downstream are calibrated against the emissive buffer, so the state ramp
 *    must be free to change colour without changing how brightly the source
 *    burns.
 */
export type Rgb = { r: number; g: number; b: number };

export const luminance = ({ r, g, b }: Rgb) => .299 * r + .587 * g + .114 * b;

/** Scale a colour to a stated luminance, preserving its channel ratios. */
export function atLuminance(colour: Rgb, level: number): Rgb {
  const scale = level / Math.max(luminance(colour), 1e-4);
  return { r: colour.r * scale, g: colour.g * scale, b: colour.b * scale };
}

/**
 * How far round the hue circle the phase is allowed to travel. 1 is the full
 * wheel; lower pulls every hue toward white together.
 */
export const PHASE_CHROMA = .92;

/**
 * Hue from the phase of the state, over the **whole** hue circle.
 *
 * This is domain colouring proper. An earlier version mixed between two poles,
 * which is a chord across the wheel rather than the wheel itself: it passes
 * through near-white halfway round, so half of every cycle carried almost no
 * colour, and two phases a full pi apart in opposite directions came out the
 * same. Three cosines at 120 degrees give the complete circle, so arg(psi) maps
 * one-to-one onto hue and every part of the cycle is distinguishable.
 *
 * The result is then normalised to a fixed luminance. That is the load-bearing
 * step: a raw hue wheel swings by a factor of two in brightness between yellow
 * and blue, and brightness here is already carrying |psi|^2. Without the
 * normalisation the colour would be silently restating the density, and getting
 * it wrong.
 *
 * The GLSL in the palette chunk is the same expression; this one exists so the
 * properties can be tested rather than eyeballed.
 */
export function phaseColour(phase: number, chroma = PHASE_CHROMA): Rgb {
  const wheel = (offset: number) => .5 + .5 * Math.cos(phase + offset);
  const raw = {
    r: wheel(0),
    g: wheel(2 * Math.PI / 3),
    b: wheel(4 * Math.PI / 3),
  };
  const mixed = {
    r: 1 + (raw.r - 1) * chroma,
    g: 1 + (raw.g - 1) * chroma,
    b: 1 + (raw.b - 1) * chroma,
  };
  return atLuminance(mixed, 1);
}

/**
 * The source's own colour by state: pale ice-blue when the state is broad,
 * through silver to a neutral white core under confinement, with a little warmth
 * only in the falloff at the top end. Returned at a level that rises with
 * confinement, hue and level having been decided separately.
 */
export function emissionFor(energy: number, seal: number): Rgb {
  const toWhite = Math.min(Math.max(energy / .5, 0), 1);
  const warmth = Math.min(Math.max((energy - .5) / .5, 0), 1) * .5 + seal * .22;
  const hue = {
    r: .62 + .38 * toWhite + .02 * warmth,
    g: .80 + .20 * toWhite - .03 * warmth,
    b: 1.0 - .12 * warmth,
  };
  return atLuminance(hue, .28 + .14 * energy);
}
