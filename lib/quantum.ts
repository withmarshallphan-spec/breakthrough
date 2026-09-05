/** Reference formulae for the ideal 1D infinite square well, not a hydrogen solver.
 * Everything here is either a physical constant or a relation that is exact for
 * that well. Nothing converts a hand gesture into an atomic length: the width
 * dial is dimensionless, and the only quantities the experience displays are
 * ratios against the open state and one L-independent invariant.
 */

export const PLANCK = 6.62607015e-34; // J s
export const HBAR = PLANCK / (2 * Math.PI);
export const ELECTRON_MASS = 9.1093837015e-31; // kg
export const ELECTRON_VOLT = 1.602176634e-19; // J

/** Standing modes superposed in the well. */
export const MODES = 6;

/** E_n = n^2 h^2 / (8 m L^2), in electronvolts, for a stated width in nm. */
export function levelEnergy(n: number, widthNm: number): number {
  const length = widthNm * 1e-9;
  return (n * n * PLANCK * PLANCK) / (8 * ELECTRON_MASS * length * length) / ELECTRON_VOLT;
}

/**
 * Position spread of mode n in the well:
 * dx = L sqrt(1/12 - 1/(2 pi^2 n^2)).
 */
export function positionSpread(widthNm: number, n: number): number {
  return widthNm * Math.sqrt(1 / 12 - 1 / (2 * Math.PI * Math.PI * n * n));
}

/** Momentum spread of mode n, dp = n pi hbar / L. */
export function momentumSpread(widthNm: number, n: number): number {
  return (n * Math.PI * HBAR) / (widthNm * 1e-9);
}

/**
 * dx dp for mode n, in units of hbar. The width cancels: squeezing the well
 * narrows dx and widens dp by exactly reciprocal factors, so the product is a
 * property of the mode alone. It is ~0.568 for the ground state, comfortably
 * above the 0.5 the uncertainty principle allows, and the gesture cannot move
 * it. This is the honest reason confinement costs energy.
 */
export function uncertaintyProduct(n: number): number {
  return n * Math.PI * Math.sqrt(1 / 12 - 1 / (2 * Math.PI * Math.PI * n * n));
}

/** dx dp of the ground state, in units of hbar. */
export const GROUND_UNCERTAINTY = uncertaintyProduct(1);

/**
 * The width dial, as a fraction of the open width: L/L0, from 1 down to 1/2.
 * Dimensionless on purpose. No atomic length is inferred from a gesture.
 */
export function relativeWellWidth(confinement: number): number {
  return Math.pow(2, -Math.min(Math.max(confinement, 0), 1));
}

/**
 * E/E0 for a fixed mode, which is exactly (L0/L)^2 in an ideal box: 1 when the
 * hands are open, 4 when the width has halved. Mode populations do not change,
 * so this is the cost of confinement alone and not of any excitation.
 */
export function energyRatio(confinement: number): number {
  return 1 / relativeWellWidth(confinement) ** 2;
}

/** The same rise mapped onto 0..1, for driving the look and the meters. */
export function normalisedEnergy(confinement: number): number {
  return (energyRatio(confinement) - 1) / 3;
}

/**
 * Wavelength to display colour, after Bruton's piecewise approximation of the
 * visible spectrum. This is a perceptual approximation for showing a spectrum,
 * not a colorimetric transform, and the wavelength it is fed is a chosen visual
 * ramp rather than any light the illustrated state would emit.
 */
function brutonRgb(nm: number): [number, number, number] {
  if (nm < 440) return [-(nm - 440) / 60, 0, 1];
  if (nm < 490) return [0, (nm - 440) / 50, 1];
  if (nm < 510) return [0, 1, -(nm - 510) / 20];
  if (nm < 580) return [(nm - 510) / 70, 1, 0];
  if (nm < 645) return [1, -(nm - 645) / 65, 0];
  return [1, 0, 0];
}

const SRGB_GAMMA = (v: number) => (v <= .0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - .055);

/**
 * Display colour of a single wavelength. Outside the visible band there is no
 * colour to show, so the nearest visible edge is used and desaturated toward
 * white by how far out it is.
 */
export function spectralColor(wavelengthNm: number): [number, number, number] {
  const clamped = Math.min(Math.max(wavelengthNm, 385), 740);
  let [r, g, b] = brutonRgb(clamped);

  const peak = Math.max(r, g, b, 1e-6);
  r = Math.max(r, 0) / peak;
  g = Math.max(g, 0) / peak;
  b = Math.max(b, 0) / peak;

  const outside = wavelengthNm > 740
    ? Math.min((wavelengthNm - 740) / 700, 1)
    : wavelengthNm < 385 ? Math.min((385 - wavelengthNm) / 90, 1) : 0;
  const wash = outside * .55;

  return [
    SRGB_GAMMA(r + (1 - r) * wash),
    SRGB_GAMMA(g + (1 - g) * wash),
    SRGB_GAMMA(b + (1 - b) * wash),
  ];
}
