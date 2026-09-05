/**
 * Quality tiers. Chosen once from what the device advertises, then demoted by
 * what it actually delivers. Nothing here gates the hand rig or the palm
 * surfaces: every tier renders real geometry, and no tier falls back to a
 * screen-space glow.
 */
export type QualityTier = 'high' | 'medium' | 'fallback';

export type QualityProfile = {
  tier: QualityTier;
  /** 852-triangle face mesh for normals and depth ordering. */
  faceMesh: boolean;
  /** Per-class silhouette (face skin, body skin, hair, clothes). */
  multiclassSegmentation: boolean;
  /** Any person silhouette at all. */
  segmentation: boolean;
  /** Dense monocular depth for surfaces the rig cannot reach. */
  denseDepth: boolean;
  /** Ping-pong pairs in the irradiance blur; more widens the light's reach. */
  blurPasses: number;
  particles: number;
  /** Emissive quads along the finger bones. */
  fingerEmission: boolean;
};

const PROFILES: Record<QualityTier, Omit<QualityProfile, 'tier'>> = {
  high: {
    faceMesh: true,
    multiclassSegmentation: true,
    segmentation: true,
    denseDepth: true,
    blurPasses: 6,
    particles: 8000,
    fingerEmission: true,
  },
  medium: {
    faceMesh: true,
    multiclassSegmentation: false,
    segmentation: true,
    denseDepth: false,
    blurPasses: 4,
    particles: 8000,
    fingerEmission: true,
  },
  fallback: {
    faceMesh: false,
    multiclassSegmentation: false,
    segmentation: false,
    denseDepth: false,
    blurPasses: 3,
    particles: 3600,
    fingerEmission: false,
  },
};

/** Order demotion walks, cheapest loss of fidelity first. */
const LADDER: QualityTier[] = ['high', 'medium', 'fallback'];

function detectTier(): QualityTier {
  if (typeof navigator === 'undefined') return 'medium';
  const nav = navigator as Navigator & { deviceMemory?: number; gpu?: unknown };
  const cores = nav.hardwareConcurrency ?? 4;
  const memory = nav.deviceMemory ?? 8;
  const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
  const small = typeof window !== 'undefined' && window.innerWidth < 900;

  if (coarse || small || cores <= 4 || memory <= 4) return 'fallback';
  // WebGPU is the entry ticket for the depth model; everything else on the
  // high tier is cheap enough that a capable CPU is the real requirement.
  if (nav.gpu && cores >= 8 && memory >= 8) return 'high';
  return 'medium';
}

export type QualityController = {
  readonly profile: QualityProfile;
  /** Called once per second-ish with a measured frame rate. */
  observe: (fps: number) => boolean;
  /** Drop a specific capability without changing tier, e.g. depth failed. */
  disable: (key: 'denseDepth' | 'segmentation' | 'faceMesh') => void;
  subscribe: (listener: (profile: QualityProfile) => void) => () => void;
};

export function createQualityController(initial?: QualityTier): QualityController {
  const tier = initial ?? detectTier();
  let profile: QualityProfile = { tier, ...PROFILES[tier] };
  const listeners = new Set<(profile: QualityProfile) => void>();
  let slowWindows = 0;

  const publish = () => listeners.forEach(listener => listener(profile));

  return {
    get profile() { return profile; },
    observe(fps) {
      // Two consecutive slow windows, so one hitch does not cost a tier.
      if (fps >= 47) { slowWindows = 0; return false; }
      slowWindows += 1;
      if (slowWindows < 2) return false;
      slowWindows = 0;
      const index = LADDER.indexOf(profile.tier);
      if (index >= LADDER.length - 1) return false;
      const next = LADDER[index + 1];
      profile = { tier: next, ...PROFILES[next] };
      publish();
      return true;
    },
    disable(key) {
      if (!profile[key]) return;
      profile = { ...profile, [key]: false };
      if (key === 'segmentation') profile.multiclassSegmentation = false;
      publish();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
