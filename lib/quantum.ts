/** Reference formulae for the ideal 1D infinite square well, not a hydrogen solver.
 * Physical-unit helpers are available for reference/tests only. The experience
 * uses dimensionless width, qualitative energy, and explicitly artistic color.
 */

export const PLANCK = 6.62607015e-34; // J s
export const HBAR = PLANCK / (2 * Math.PI);
export const ELECTRON_MASS = 9.1093837015e-31; // kg
export const ELECTRON_VOLT = 1.602176634e-19; // J
export const SPEED_OF_LIGHT = 299792458; // m/s

// Legacy reference scale for the analytic helpers only; not gesture calibration.
export const WELL_MAX_NM = .88;
export const WELL_MIN_NM = .54;

/** Standing modes superposed in the well. */
export const MODES = 6;

/** Well width in nanometres for a confinement value in [0, 1]. */
export function wellWidth(confinement: number): number {
  const c = Math.min(Math.max(confinement, 0), 1);
  return WELL_MAX_NM * Math.pow(WELL_MIN_NM / WELL_MAX_NM, c);
}

/** E_n = n^2 h^2 / (8 m L^2), in electronvolts. */
export function levelEnergy(n: number, widthNm: number): number {
  const length = widthNm * 1e-9;
  return (n * n * PLANCK * PLANCK) / (8 * ELECTRON_MASS * length * length) / ELECTRON_VOLT;
}

/** Photon energy in eV released by an n -> m transition. */
export function transitionEnergy(widthNm: number, from = 2, to = 1): number {
  return levelEnergy(from, widthNm) - levelEnergy(to, widthNm);
}

/** Wavelength in nm of that transition, lambda = hc / E. */
export function transitionWavelength(widthNm: number, from = 2, to = 1): number {
  const joules = transitionEnergy(widthNm, from, to) * ELECTRON_VOLT;
  return (PLANCK * SPEED_OF_LIGHT) / joules * 1e9;
}

/** de Broglie wavelength of the standing mode, 2L / n, in nm. */
export function deBroglieWavelength(widthNm: number, n: number): number {
  return (2 * widthNm) / n;
}

/** Momentum magnitude of mode n, p = nh / 2L, in kg m/s. */
export function momentum(widthNm: number, n: number): number {
  return (n * PLANCK) / (2 * widthNm * 1e-9);
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
 * dx dp expressed in units of hbar. Constant at ~0.568 for the ground state,
 * and never below the 0.5 the uncertainty principle allows.
 */
export function uncertaintyProduct(widthNm: number, n: number): number {
  return (positionSpread(widthNm, n) * 1e-9 * momentumSpread(widthNm, n)) / HBAR;
}

/**
 * Confinement energy normalised across the reachable range of well widths, for
 * driving the look. A dimensionless width dial from L₀ to L₀/2 uses the
 * fixed-mode inverse-square law. No atomic length is inferred from a gesture.
 */
export function relativeWellWidth(confinement: number): number {
  return Math.pow(2, -Math.min(Math.max(confinement, 0), 1));
}

export function normalisedEnergy(confinement: number): number {
  return (1 / relativeWellWidth(confinement) ** 2 - 1) / 3;
}

export type Band = 'infrared' | 'visible' | 'ultraviolet';

export function band(wavelengthNm: number): Band {
  if (wavelengthNm > 740) return 'infrared';
  if (wavelengthNm < 380) return 'ultraviolet';
  return 'visible';
}

/**
 * Wavelength to display colour, after Bruton's piecewise approximation of the
 * visible spectrum. This is a perceptual approximation for showing a spectrum,
 * not a colorimetric transform; it is used because the trace needs a
 * recognisable spectral hue, not a measurement.
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
 * white by how far out it is; the readout names the band.
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

