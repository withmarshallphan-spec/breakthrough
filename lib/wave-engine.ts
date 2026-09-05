import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import type { TrackingUpdate } from './hand-tracker';
import { insetContour, NEARNESS_SPAN_MM, PALM_CONTOUR_POINTS, type PalmFrame, type RigPoint } from './palm-geometry';
import { MODES, normalisedEnergy, relativeWellWidth } from './quantum';
import type { QualityProfile } from './quality';
import { emissionFor, PHASE_CHROMA } from './field-palette';

export type FieldPoint = { x: number; y: number };

export type FieldTracking = TrackingUpdate;

export type WaveEngine = {
  setTracking: (tracking: FieldTracking) => void;
  /** Follow a tier change without rebuilding the pipeline. */
  setProfile: (profile: QualityProfile) => void;
  /** Keep the native video visible and render only the field into the canvas. */
  setCameraCompositing: (enabled: boolean) => void;
  setDiagnosticStage: (stage: WaveDiagnosticStage) => void;
  getDiagnostics: () => WaveEngineDiagnostics;
  /** Tests the actual WebGL video texture, rather than only the HTML video. */
  hasCameraTexturePixels: () => boolean;
  destroy: () => void;
};

export type WaveEngineOptions = {
  /** iOS starts in overlay mode so the GPU can never hide the live camera. */
  cameraCompositing?: boolean;
};

export type WaveDiagnosticStage =
  | 'raw'
  | 'transparent'
  | 'tracking'
  | 'wave'
  | 'segmentation'
  | 'depth'
  | 'composite';

export type WaveEngineDiagnostics = {
  webgl2: boolean;
  cameraTextureInitialized: boolean;
  cameraCompositing: boolean;
  diagnosticStage: WaveDiagnosticStage;
};

const DIAGNOSTIC_STAGE_ORDER: Record<WaveDiagnosticStage, number> = {
  raw: 0,
  transparent: 1,
  tracking: 2,
  wave: 3,
  segmentation: 4,
  depth: 5,
  composite: 6,
};

// Standing modes remain pinned to the two boundaries. Fixed populations keep
// the confinement lesson honest: E rises as L shrinks even without exciting n.
// Transverse geometry and light are cinematic illustrations of this 1D state.

const screenVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// Shared wave evaluation. `lag` offsets time per layer/particle so trailing
// copies of the curve smear into a motion trail instead of a tangle.
const waveChunk = /* glsl */ `
  #define MODES ${MODES}
  const float PI = 3.14159265;
  uniform float uWeights[MODES];
  uniform float uPhases[MODES];

  /**
   * The superposition as a complex amplitude. Each mode contributes its own
   * spatial shape times its own rotating phase factor; the real part is what
   * the ribbon traces, and the argument is the phase.
   */
  vec2 psiComplex(float t, float lag) {
    vec2 sum = vec2(0.0);
    for (int i = 0; i < MODES; i++) {
      float n = float(i + 1);
      float amplitude = sqrt(uWeights[i]) * sin(n * PI * t);
      float angle = uPhases[i] + lag * n;
      sum += amplitude * vec2(cos(angle), sin(angle));
    }
    return sum;
  }

  float psi(float t, float lag) { return psiComplex(t, lag).x; }

  /**
   * arg(psi). Probability density throws the phase away; it is nevertheless a
   * real property of the state, and the one that makes a wavefunction a wave
   * rather than a cloud. Showing it in hue while density stays in brightness is
   * the standard domain-colouring convention, and it means the colour on screen
   * is carrying information instead of decorating.
   */
  float psiPhase(float t, float lag) {
    vec2 z = psiComplex(t, lag);
    return atan(z.y, z.x);
  }

  float dpsi(float t, float lag) {
    float sum = 0.0;
    for (int i = 0; i < MODES; i++) {
      float n = float(i + 1);
      sum += sqrt(uWeights[i]) * n * PI * cos(n * PI * t) * cos(uPhases[i] + lag * n);
    }
    return sum;
  }

  // Time-averaged |psi|^2: the probability density, with visible nodes.
  float density(float t) {
    float sum = 0.0;
    for (int i = 0; i < MODES; i++) {
      float n = float(i + 1);
      float s = sin(n * PI * t);
      sum += uWeights[i] * s * s;
    }
    return sum;
  }
`;

// Everything downstream works in "square space": NDC with x scaled by aspect,
// so a perpendicular is actually perpendicular on screen and a length means
// the same thing in every direction.
// How saturated the spectral colour is allowed to be on screen. 1 is the true
// hue for the wavelength; lower keeps the sweep but pulls it toward white,
// which is kinder to skin tones through the green middle of the range.


// The source's own colour, set per state on the CPU: ice-blue when the state
// is broad, silver through white as it is confined, a faint warm cast only in
// the falloff at high confinement. It is a chosen palette, not predicted
// emission -- and the core stays neutral, which is how a genuinely bright
// source lands on skin. Nothing here tints a whole hand.
const paletteChunk = /* glsl */ `
  uniform vec3 uEmission;
  uniform float uPhaseChroma;
  const vec3 SILVER = vec3(1.0, .985, .965);

  vec3 fieldTint(float energy) { return uEmission; }

  /**
   * Hue from the phase of the state, over the whole hue circle -- domain
   * colouring proper, so arg(psi) maps one-to-one onto hue and no two phases
   * share a colour.
   *
   * Normalised to fixed luminance, which is the load-bearing step: a raw hue
   * wheel swings by a factor of two in brightness between yellow and blue, and
   * brightness here is already carrying |psi|^2. Left unnormalised the colour
   * would be restating the density, and restating it wrongly.
   *
   * Mirrored in lib/field-palette.ts, where these properties are tested.
   */
  vec3 phaseTint(float phase) {
    vec3 wheel = .5 + .5 * cos(phase + vec3(0.0, 2.0943951, 4.1887902));
    wheel = mix(vec3(1.0), wheel, uPhaseChroma);
    return wheel / max(dot(wheel, vec3(.299, .587, .114)), 1e-4);
  }
`;

const frameChunk = /* glsl */ `
  uniform vec2 uResolution;
  uniform vec2 uLeft;
  uniform vec2 uRight;
  uniform float uConfinement;
  uniform float uMode;
  uniform float uPulse;
  /**
   * How near the state is to collapse: palms very close, nothing sealed yet.
   * Rises continuously out of the confinement dial and falls away again as a
   * clasp takes over, so the treatment ramps in rather than switching on.
   * Everything it drives narrows or brightens -- nothing it drives shakes.
   */
  uniform float uCritical;

  float aspectOf() { return uResolution.x / max(uResolution.y, 1.0); }
  vec2 toSquare(vec2 ndc, float aspect) { return vec2(ndc.x * aspect, ndc.y); }
  vec2 toNdc(vec2 square, float aspect) { return vec2(square.x / aspect, square.y); }

  // The release impulse breathes the whole volume outward for a moment, so
  // opening a clasp is an event rather than a fade back to the resting size.
  float amplitudeFor(float span) {
    return max(span * mix(.24, .19, uConfinement)
      * (1.0 - uCritical * .34) * (1.0 + uPulse * .24), .008);
  }
`;

// Landmark-constrained perspective volume. Endpoints project exactly to palms;
// offsets occupy a plane and its binormal in camera space, including true z.
const volumeChunk = /* glsl */ `
  uniform vec2 uEndpointDepth;
  uniform float uPalmScale;
  float cameraDistance(float nearValue) { return 4.8 - 3.96 * nearValue; }
  float nearAt(vec3 p) { return clamp((4.8 + p.z) / 3.96, 0.0, 1.0); }
  float focal() { return 1.732 * aspectOf(); }
  vec3 unprojectPalm(vec2 ndc, float nearValue) {
    float distance = cameraDistance(nearValue);
    return vec3(toSquare(ndc, aspectOf()) * distance / focal(), -distance);
  }
  vec3 fieldPoint(float t, float lateral, float depthOffset) {
    vec3 a = unprojectPalm(uLeft, uEndpointDepth.x);
    vec3 b = unprojectPalm(uRight, uEndpointDepth.y);
    vec3 axis = normalize(b - a + vec3(.00001, 0.0, 0.0));
    vec3 side = normalize(cross(vec3(0.0, 0.0, 1.0), axis) + vec3(0.0, .00001, 0.0));
    vec3 binormal = normalize(cross(axis, side));
    float scale = -(a.z + b.z) * .5 / focal();
    return mix(a, b, t) + (side * lateral + binormal * depthOffset) * scale;
  }
  vec4 projectField(vec3 p) {
    float distance = max(-p.z, .25);
    return vec4(toNdc(p.xy * focal(), aspectOf()), (1.0 - 2.0 * nearAt(p)) * distance, distance);
  }
`;

// How much the mask edge is broken up. The occluder is rendered at a fraction
// of display resolution, so its boundary arrives as a visible staircase; a fine
// dot screen dissolves that into a halftone the eye reads as a soft edge
// instead of a cut. It only acts inside the transition band -- solid fingers
// stay solid and clear sky stays clear.
const HALFTONE_DEPTH = .55;

// Shared by the passes that cut the field and by the composite that shades its
// edges, so both agree about where a silhouette ends. Kept separate from the
// occluder chunk because the composite declares its own samplers and cannot
// take that one wholesale.
const halftoneChunk = /* glsl */ `
  /** A rotated dot screen, in device pixels. */
  float halftone(vec2 fragment) {
    vec2 rotated = vec2(fragment.x * .7071 - fragment.y * .7071,
                        fragment.x * .7071 + fragment.y * .7071);
    return sin(rotated.x * 1.05) * sin(rotated.y * 1.05) * .5 + .5;
  }

  /** Dither only the soft shoulder of a coverage value. */
  float featherCoverage(float coverage, vec2 fragment) {
    float shoulder = 1.0 - abs(coverage * 2.0 - 1.0);
    return clamp(coverage + (halftone(fragment) - .5) * shoulder * ${HALFTONE_DEPTH.toFixed(2)}, 0.0, 1.0);
  }
`;

const occlusionChunk = /* glsl */ `
  uniform sampler2D tOccluder;
  uniform vec2 uRenderResolution;
  uniform float uOcclusion;

  ${halftoneChunk}

  float visibilityAt(vec2 uv, float depth) {
    vec4 hand = texture2D(tOccluder, uv);
    // A wider soft depth margin than a hard cutout, then the dot screen across
    // the edge itself, so a finger crossing a strand dissolves it rather than
    // slicing it.
    float ahead = smoothstep(depth - .012, depth + .05, hand.r);
    float covered = featherCoverage(hand.a * ahead, uv * uRenderResolution);
    return 1.0 - uOcclusion * covered;
  }
`;

// The triangulated palm surface: R = depth, G = structured emission, B = the
// smooth body of it, which is what lights skin. The mesh is the landmark
// silhouette itself, so there is no disc anywhere in this pass.
// Low-frequency turbulence gives the light soft wisps, without a repeating
// radial pattern or flashing noise. This is a visual treatment only.
const mistChunk = /* glsl */ `
  float mistHash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float mistNoise(vec2 p) {
    vec2 cell = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mistHash(cell), mistHash(cell + vec2(1.0, 0.0)), f.x),
      mix(mistHash(cell + vec2(0.0, 1.0)), mistHash(cell + vec2(1.0)), f.x), f.y);
  }
  float mist(vec2 p) {
    return .58 * mistNoise(p) + .28 * mistNoise(p * 2.03 + 17.0)
      + .14 * mistNoise(p * 4.07 - 9.0);
  }
`;

const palmVertexShader = /* glsl */ `
  attribute float aGlow;
  attribute vec2 aFlow;
  attribute float aField;
  varying float vGlow;
  varying float vDepth;
  varying vec2 vFlow;
  varying float vField;
  void main() {
    vGlow = aGlow;
    vDepth = position.z;
    // x: angle around the silhouette in radians, y: 0 at the centre of the
    // palm, 1 at its edge. Both travel with the hand, so the structure below
    // rotates and foreshortens with the surface rather than with the screen.
    vFlow = aFlow;
    vField = aField;
    gl_Position = vec4(position.xy, 1.0 - position.z * 2.0, 1.0);
  }
`;
const palmFragmentShader = /* glsl */ `
  precision highp float;
  uniform float uTime;
  uniform float uEnergy;
  uniform float uSeal;
  varying float vGlow;
  varying float vDepth;
  varying vec2 vFlow;
  varying float vField;

  ${mistChunk}

  void main() {
    float body = smoothstep(0.0, 1.0, vGlow);
    vec2 local = vec2(cos(vFlow.x), sin(vFlow.x)) * vFlow.y;
    float flow = mist(local * 3.1 + vec2(uTime * .14, -uTime * .21));
    float structure = .72 + .38 * flow;
    // Anisotropy: the side of the palm the field leaves from burns brighter,
    // which is what stops a symmetric surface from reading as a blob.
    float emission = body * structure * mix(.5, 1.0, vField) * (1.0 + uSeal * .5);
    gl_FragColor = vec4(vDepth, emission, body, 1.0);
  }
`;

// Whole-person coverage, composited into the occluder at the head's measured
// distance. Segmentation contributes a silhouette and nothing else: the depth
// is the head proxy's, and the hand rig overwrites it wherever it reaches,
// because only the rig knows how far away an individual finger is.
const personVertexShader = /* glsl */ `
  uniform float uBodyDepth;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    // Same clip-space depth convention as the rig spheres under the shared
    // orthographic camera: world z = 2 * nearness - 1, clip z = -world z / 2.
    // The plane sits at the head's distance so the rig always wins the depth
    // test; the per-pixel distance it reports in r is the useful part.
    gl_Position = vec4(position.xy, .5 - uBodyDepth, 1.0);
  }
`;
// How hard the silhouette edge rolls the normal over. This is the only shape
// a mask can carry; it exists so a body catches rim light, not to fake relief.
const PERSON_RELIEF = 2.6;
// How hard a depth-map slope rolls the normal over, once dense depth is
// available and the body has real relief rather than only an outline.
const DEPTH_RELIEF = 9.0;
const personFragmentShader = /* glsl */ `
  precision highp float;
  uniform sampler2D tPerson;
  uniform sampler2D tDepth;
  uniform vec2 uResolution;
  uniform vec2 uVideoSize;
  uniform vec2 uPersonTexel;
  uniform vec2 uDepthTexel;
  uniform float uHasDepth;
  uniform float uBodyDepth;
  varying vec2 vUv;

  vec2 coverUv(vec2 uv) {
    float screenAspect = uResolution.x / max(uResolution.y, 1.0);
    float videoAspect = uVideoSize.x / max(uVideoSize.y, 1.0);
    if (screenAspect > videoAspect) uv.y = (uv.y - .5) * (videoAspect / screenAspect) + .5;
    else uv.x = (uv.x - .5) * (screenAspect / videoAspect) + .5;
    uv.x = 1.0 - uv.x;
    return uv;
  }

  float personAt(vec2 uv) { return texture2D(tPerson, coverUv(uv)).r; }
  float depthAt(vec2 uv) { return texture2D(tDepth, coverUv(uv)).r; }

  void main() {
    float coverage = personAt(vUv);
    if (coverage < .06) discard;
    // Sampled in screen space, so the mirrored feed is handled by coverUv and
    // the gradient still points the way it looks like it should.
    vec2 probe = uPersonTexel * 2.0;
    vec2 outline = vec2(
      personAt(vUv + vec2(probe.x, 0.0)) - personAt(vUv - vec2(probe.x, 0.0)),
      personAt(vUv + vec2(0.0, probe.y)) - personAt(vUv - vec2(0.0, probe.y)));
    float outlineSlope = clamp(length(outline) * ${PERSON_RELIEF.toFixed(1)}, 0.0, .94);
    vec2 maskNormal = -outline / max(length(outline), 1e-5) * outlineSlope;

    // With dense depth the body has actual relief, so the normal comes from
    // the depth gradient -- a shoulder turns away from the light because it is
    // turned away, not because it is near the edge of a cut-out.
    float depth = uBodyDepth;
    vec2 normalXy = maskNormal;
    if (uHasDepth > .5) {
      vec2 step = uDepthTexel * 1.5;
      float left = depthAt(vUv - vec2(step.x, 0.0));
      float right = depthAt(vUv + vec2(step.x, 0.0));
      float down = depthAt(vUv - vec2(0.0, step.y));
      float up = depthAt(vUv + vec2(0.0, step.y));
      depth = depthAt(vUv);
      // A height field with z toward the viewer has normal proportional to
      // (-df/dx, -df/dy, 1); the gain folds in the sample spacing.
      vec2 slope = vec2(right - left, up - down) * ${DEPTH_RELIEF.toFixed(1)};
      float magnitude = length(slope);
      normalXy = -slope / max(magnitude, 1e-5) * clamp(magnitude, 0.0, .94);
      // Keep the silhouette's own edge roll where the depth map is smoothest:
      // it is what makes a body catch rim light from behind.
      normalXy = mix(normalXy, maskNormal, clamp(outlineSlope - magnitude, 0.0, 1.0));
    }
    gl_FragColor = vec4(clamp(depth, 0.0, 1.0), normalXy * .5 + .5, coverage);
  }
`;

const cameraFragmentShader = /* glsl */ `
  precision highp float;
  uniform sampler2D tVideo;
  uniform vec2 uResolution;
  uniform vec2 uVideoSize;
  uniform float uEnergy;
  uniform float uCameraCompositing;
  varying vec2 vUv;

  vec2 coverUv(vec2 uv) {
    float screenAspect = uResolution.x / max(uResolution.y, 1.0);
    float videoAspect = uVideoSize.x / max(uVideoSize.y, 1.0);
    if (screenAspect > videoAspect) {
      float scale = videoAspect / screenAspect;
      uv.y = (uv.y - .5) * scale + .5;
    } else {
      float scale = screenAspect / videoAspect;
      uv.x = (uv.x - .5) * scale + .5;
    }
    uv.x = 1.0 - uv.x;
    return uv;
  }

  // The room grade, which now only shapes albedo -- the colour of the surfaces,
  // not how brightly they are lit. For the untouched camera image:
  // DESATURATE 0, CONTRAST 1, EXPOSURE 1, VIGNETTE 0.
  const float ROOM_DESATURATE = .18;
  const float ROOM_CONTRAST = 1.08;
  const float ROOM_EXPOSURE = .95;
  const float ROOM_VIGNETTE = .24;

  void main() {
    if (uCameraCompositing < .5) {
      gl_FragColor = vec4(0.0);
      return;
    }
    vec3 color = texture2D(tVideo, coverUv(vUv)).rgb;
    float luma = dot(color, vec3(.299, .587, .114));
    color = mix(color, vec3(luma), ROOM_DESATURATE);
    color = max((color - .5) * ROOM_CONTRAST + .5, vec3(0.0));
    // This pass now yields albedo only: what the room reflects. How much light
    // reaches it is decided in the field pass, so the wave can be the room's
    // light rather than a layer painted over it.
    color *= ROOM_EXPOSURE;
    float vignette = 1.0 - smoothstep(.32, 1.0, distance(vUv, vec2(.5)));
    color *= (1.0 - ROOM_VIGNETTE) + ROOM_VIGNETTE * vignette;
    gl_FragColor = vec4(color, 1.0);
  }
`;

// The wave's own light, rendered on its own into a buffer. Nothing here knows
// about the room: it is the source, and the composite pass decides what it
// falls on. Blurring this buffer is what lets the particles light the scene.
const emissiveFragmentShader = /* glsl */ `
  precision highp float;
  uniform float uTime;
  uniform float uEnergy;
  uniform float uPresence;
  uniform float uSeal;
  varying vec2 vUv;
  uniform sampler2D tPalm;

  ${waveChunk}
  ${frameChunk}
  ${paletteChunk}
  ${volumeChunk}
  ${occlusionChunk}
  ${mistChunk}

  void main() {
    float aspect = aspectOf();
    vec2 q = toSquare(vUv * 2.0 - 1.0, aspect);
    vec2 left = toSquare(uLeft, aspect);
    vec2 right = toSquare(uRight, aspect);

    vec2 axis = right - left;
    float span = max(length(axis), .05);
    vec2 dir = axis / span;
    vec2 nrm = vec2(-dir.y, dir.x);
    vec2 middle = (left + right) * .5;

    vec2 rel = q - left;
    float t = dot(rel, dir) / span;
    float d = dot(rel, nrm);

    float amp = amplitudeFor(span);
    float y = psi(t, 0.0) * amp;
    float slope = dpsi(t, 0.0) * amp / span;
    float dist = abs(d - y) * inversesqrt(1.0 + slope * slope);

    float bridge = 1.0 - smoothstep(0.0, .7, uSeal);
    float box = smoothstep(-.008, .008, t) * smoothstep(-.008, .008, 1.0 - t) * bridge;
    float coreWidth = (span * .009 + .004) * (1.0 - uCritical * .38);
    float glowWidth = (span * .07 + .018) * (1.0 - uCritical * .3);
    float core = exp(-(dist * dist) / (coreWidth * coreWidth));
    float glow = exp(-(dist * dist) / (glowWidth * glowWidth));

    // The body of the wave: a fuller volume than the old thin filament, so the
    // thing reads as substance rather than a translucent thread.
    // Two widths of body: the cloud that fills the well, and a slightly softer
    // halo around it. The halo has to stay bounded -- at this amplitude a wide
    // one reaches past the top of frame and reads as a column of light.
    float cloudWidth = amp * .95;
    float cloud = density(t) * exp(-(d * d) / (cloudWidth * cloudWidth));
    float haze = density(t) * exp(-(d * d) / (cloudWidth * cloudWidth * 1.6));

    vec2 knotOffset = q - middle;
    float knotDist = length(knotOffset);
    // Sealing compresses the volume rather than just dimming the bridge: the
    // interior tightens as the hands close on it.
    float knotRadius = max(uPalmScale * .48 * (1.0 - uSeal * .34), .012);
    vec2 smokeUv = knotOffset / knotRadius;
    float turbulence = mist(smokeUv * 2.3 + vec2(-uTime * .17, uTime * .23));
    knotRadius *= .78 + .44 * turbulence;
    float knot = exp(-(knotDist * knotDist) / (knotRadius * knotRadius));
    float churn = .68 + .45 * turbulence;
    float knotCore = pow(knot, 3.5);
    float sealed = uSeal * uPresence;

    vec3 tint = fieldTint(uEnergy);
    // The body of the wave takes the phase where it is; the halo and the core
    // stay near neutral, because that is the part that becomes room light and
    // a saturated spill on skin reads as a filter over the shot.
    vec3 body = phaseTint(psiPhase(clamp(t, 0.0, 1.0), 0.0));
    vec3 halo = mix(tint, SILVER, .12);

    vec3 color = vec3(0.0);
    color += SILVER * core * box * (.06 + uEnergy * .09) * uPresence;
    color += halo * glow * box * (.05 + uEnergy * .1) * uPresence;
    color += body * cloud * box * (.055 + uEnergy * .11) * uPresence;
    color += body * haze * box * (.03 + uEnergy * .055) * uPresence;
    // A sealed interior is brighter and smaller than the open bridge it
    // replaces, which is what makes a clasp read as compression rather than as
    // the effect having been switched off. It is only ever seen through the
    // hands or through the gaps between the fingers -- the visibility test
    // below is what puts it behind them.
    color += SILVER * knotCore * sealed * (.19 + uEnergy * .28) * (1.0 + uSeal * .5);
    color += tint * knot * churn * sealed * (.08 + uEnergy * .17);

    float wisps = mist(vec2(t * 6.0 - uTime * .14, d / max(amp, .01) * 2.2 + uTime * .1));
    color += body * haze * box * smoothstep(.34, .73, wisps) * (.04 + uEnergy * .1) * uPresence;
    // Squeezing produces a hotter core and denser surrounding light, rather
    // than just reducing the total number of lit pixels.
    float radiance = 1.0 + uEnergy * 1.9 + uSeal * 1.1 + uCritical * 1.5;
    color *= radiance;
    float fieldDepth = nearAt(fieldPoint(clamp(t, 0.0, 1.0), d, -uSeal * uPalmScale * .18));
    color *= visibilityAt(vUv, fieldDepth);
    vec4 palm = texture2D(tPalm, vUv);
    // Surface emission has its own depth, so a nearer finger can cover it.
    color += mix(tint, SILVER, uEnergy * .35) * palm.g * (.07 + uEnergy * .33) * uPresence
      * visibilityAt(vUv, palm.r + .025);
    gl_FragColor = vec4(color, 1.0);
  }
`;

// Separable blur, run at low resolution. The result is the irradiance the wave
// throws into the room: broad, and shaped by where the particles actually are.
const blurFragmentShader = /* glsl */ `
  precision highp float;
  uniform sampler2D tSource;
  uniform vec2 uDirection;
  varying vec2 vUv;

  void main() {
    vec3 sum = texture2D(tSource, vUv).rgb * .227;
    sum += (texture2D(tSource, vUv + uDirection * 1.385).rgb
          + texture2D(tSource, vUv - uDirection * 1.385).rgb) * .316;
    sum += (texture2D(tSource, vUv + uDirection * 3.231).rgb
          + texture2D(tSource, vUv - uDirection * 3.231).rgb) * .07;
    gl_FragColor = vec4(sum, 1.0);
  }
`;

const compositeFragmentShader = /* glsl */ `
  precision highp float;
  uniform sampler2D tEnvironment;
  uniform sampler2D tEmissive;
  uniform sampler2D tLight;
  uniform sampler2D tOccluder;
  uniform sampler2D tFrontOccluder;
  uniform sampler2D tPalm;
  uniform sampler2D tPerson;
  uniform sampler2D tDepth;
  uniform vec2 uVideoSize;
  uniform vec2 uRenderResolution;
  uniform float uHasDepth;
  uniform float uHasSkin;
  uniform float uTime;
  uniform vec2 uTexel;
  uniform vec2 uOccluderTexel;
  uniform vec2 uLightTexel;
  uniform float uEnergy;
  uniform float uPresence;
  uniform float uAmbient;
  uniform float uSeal;
  uniform float uHasOccluder;
  uniform float uCameraCompositing;
  varying vec2 vUv;

  // How pronounced the inferred relief is. Higher shapes the subject harder
  // but starts turning albedo edges into fake geometry.
  const float RELIEF = .24;
  // How hard the wave's own irradiance lights the room.
  // Calibrated against the measured buffer: a wide blur conserves energy, so
  // spreading the wave over the frame leaves irradiance around 0.01. The gain
  // has to be in the tens for the field to be the room's actual light source.
  const float LIGHT_GAIN = 30.0;
  // How hard light wraps a backlit silhouette.
  const float RIM_GAIN = 2.1;
  // How much of a sealed field escapes through the hands holding it. Same
  // scale as LIGHT_GAIN, because it reads the same irradiance buffer.
  const float SEAL_TRANSMIT = 20.0;
  // Over how much of the depth range the light falls to a quarter strength.
  // Depth is 0..1 nearness, derived from apparent size, so this is a real
  // distance falloff between the hands and the face behind them. Generous on
  // purpose: a face is most of a metre behind the hands holding the field, and
  // a tight falloff leaves it unlit however bright the source is.
  const float DEPTH_RANGE = .38;
  // What is assumed for pixels the rig knows nothing about -- the room behind.
  const float BACKGROUND_DEPTH = .08;
  // How far the field throws light as a source in the room, in camera units.
  // The whole tracked depth range is about .84 units deep, so a radius well
  // past that lights the whole subject and the wall behind them, not just the
  // hands holding it.
  const float NEAR_FIELD_RADIUS = 2.1;
  // How hard that near-field term lights the room, relative to the screen-space
  // irradiance it sits alongside.
  const float MATCH_GAIN = 1.6;

  ${waveChunk}
  ${frameChunk}
  ${paletteChunk}
  ${volumeChunk}
  ${halftoneChunk}

  float luma(vec3 c) { return dot(c, vec3(.299, .587, .114)); }

  // Same cover-and-mirror mapping the camera pass uses, so the silhouette and
  // the depth map line up with the image they were computed from.
  vec2 coverUv(vec2 uv) {
    float screenAspect = uResolution.x / max(uResolution.y, 1.0);
    float videoAspect = uVideoSize.x / max(uVideoSize.y, 1.0);
    if (screenAspect > videoAspect) uv.y = (uv.y - .5) * (videoAspect / screenAspect) + .5;
    else uv.x = (uv.x - .5) * (screenAspect / videoAspect) + .5;
    uv.x = 1.0 - uv.x;
    return uv;
  }

  void main() {
    // Overlay mode never samples, grades, refracts, or relights the camera.
    // It leaves the native video element responsible for the visible feed and
    // composites only the field's own transparent light above it.
    if (uCameraCompositing < .5) {
      vec3 emission = texture2D(tEmissive, vUv).rgb;
      vec3 halo = texture2D(tLight, vUv).rgb;
      vec3 overlay = emission + halo * .46;
      float alpha = clamp(max(max(overlay.r, overlay.g), overlay.b) * .72, 0.0, .92);
      gl_FragColor = vec4(overlay, alpha);
      return;
    }
    float aspect = aspectOf();
    vec2 q = toSquare(vUv * 2.0 - 1.0, aspect);
    vec2 left = toSquare(uLeft, aspect);
    vec2 right = toSquare(uRight, aspect);
    vec2 axis = right - left;
    float span = max(length(axis), .05);
    vec2 dir = axis / span;
    vec2 nrm = vec2(-dir.y, dir.x);
    float t = dot(q - left, dir) / span;
    float d = dot(q - left, nrm);
    float amp = amplitudeFor(span);
    float y = psi(t, 0.0) * amp;

    // --- Occlusion --------------------------------------------------------
    vec4 mask = texture2D(tOccluder, vUv);
    float coverage = mask.a * uHasOccluder;
    float localDepth = nearAt(fieldPoint(clamp(t, 0.0, 1.0), d, -uSeal * uPalmScale * .18));
    vec4 frontMask = texture2D(tFrontOccluder, vUv);
    float frontCoverage = frontMask.a;
    // The same widened band and dot screen the field itself is cut with, so the
    // refraction and the strand agree about where an edge is.
    float inFront = smoothstep(localDepth - .012, localDepth + .05, frontMask.r);
    float hidden = featherCoverage(clamp(frontCoverage * inFront, 0.0, 1.0), vUv * uRenderResolution);
    float visible = 1.0 - hidden;

    float mL = texture2D(tFrontOccluder, vUv - vec2(uOccluderTexel.x, 0.0)).a;
    float mR = texture2D(tFrontOccluder, vUv + vec2(uOccluderTexel.x, 0.0)).a;
    float mD = texture2D(tFrontOccluder, vUv - vec2(0.0, uOccluderTexel.y)).a;
    float mU = texture2D(tFrontOccluder, vUv + vec2(0.0, uOccluderTexel.y)).a;
    float rimEdge = length(vec2(mR - mL, mU - mD)) * uHasOccluder;

    // --- Refraction -------------------------------------------------------
    float glowWidth = (span * .07 + .018) * (1.0 - uCritical * .3);
    float slope = dpsi(t, 0.0) * amp / span;
    float dist = abs(d - y) * inversesqrt(1.0 + slope * slope);
    float wellGate = step(0.0, t) * step(t, 1.0);
    float lens = exp(-(dist * dist) / (glowWidth * glowWidth)) * uPresence * (1.0 - uSeal) * wellGate * visible;
    vec2 knotOffset = q - (left + right) * .5;
    float claspLens = exp(-dot(knotOffset, knotOffset) / max(uPalmScale * uPalmScale * .8, .0001)) * uSeal * uPresence * visible;
    vec2 nUv = vec2(nrm.x / aspect, nrm.y) * .5;
    float strength = (.012 + uEnergy * .026 + uCritical * .022) * lens;
    vec2 offset = nUv * clamp((d - y) / max(glowWidth, .001), -1.5, 1.5) * strength;
    offset += knotOffset * sin(length(knotOffset) * 65.0 - uTime * 2.0) * claspLens * .007;
    float dispersion = strength * .18;
    vec3 refracted = vec3(
      texture2D(tEnvironment, clamp(vUv + offset + nUv * dispersion, .002, .998)).r,
      texture2D(tEnvironment, clamp(vUv + offset, .002, .998)).g,
      texture2D(tEnvironment, clamp(vUv + offset - nUv * dispersion, .002, .998)).b
    );

    // --- Surface ----------------------------------------------------------
    // Hands and the head come with true geometric normals from the rig. Only
    // the rest of the room falls back to relief inferred from image brightness.
    // A wider sample deliberately ignores sensor grain and small exposure
    // changes. Those are not room geometry and should not steer the light.
    vec2 probe = uTexel * 8.0;
    float lx = luma(texture2D(tEnvironment, vUv - vec2(probe.x, 0.0)).rgb);
    float rx = luma(texture2D(tEnvironment, vUv + vec2(probe.x, 0.0)).rgb);
    float dy = luma(texture2D(tEnvironment, vUv - vec2(0.0, probe.y)).rgb);
    float uy = luma(texture2D(tEnvironment, vUv + vec2(0.0, probe.y)).rgb);
    vec3 relief = normalize(vec3((lx - rx) * RELIEF, (dy - uy) * RELIEF, 1.0));

    // Feather surface normals, not the finger depth mask. This broadens the
    // light across skin without smearing the webcam or erasing its texture.
    vec2 normalProbe = uOccluderTexel * 3.0;
    vec4 nL = texture2D(tOccluder, vUv - vec2(normalProbe.x, 0.0));
    vec4 nR = texture2D(tOccluder, vUv + vec2(normalProbe.x, 0.0));
    vec4 nD = texture2D(tOccluder, vUv - vec2(0.0, normalProbe.y));
    vec4 nU = texture2D(tOccluder, vUv + vec2(0.0, normalProbe.y));
    float totalCoverage = mask.a * 4.0 + nL.a + nR.a + nD.a + nU.a;
    vec2 softNormal = (mask.gb * mask.a * 4.0 + nL.gb * nL.a + nR.gb * nR.a + nD.gb * nD.a + nU.gb * nU.a) / max(totalCoverage, .001);
    vec2 rigNxy = mix(vec2(0.0), softNormal * 2.0 - 1.0, step(.001, totalCoverage));
    vec3 rigNormal = vec3(rigNxy, sqrt(max(1.0 - dot(rigNxy, rigNxy), 0.0)));
    // How much shape the covering surface actually has to offer. The hand rig
    // is real geometry and curves everywhere; a silhouette mask only knows its
    // own outline, so across the middle of a torso the image's own relief is
    // the better guess and the mask keeps only its edges.
    float shaped = clamp(length(rigNxy) * 2.4, 0.0, 1.0);
    vec3 normal = normalize(mix(relief, rigNormal, coverage * mix(.25, .9, shaped)));

    // --- Light from the field itself --------------------------------------
    // Irradiance is the blurred emissive, so every grain and filament that is
    // actually on screen contributes. Its gradient gives the direction the
    // light arrives from, which is what shapes the subject.
    vec2 softReach = uLightTexel * 4.0;
    vec3 irradiance = texture2D(tLight, vUv).rgb * .5
      + texture2D(tLight, vUv + vec2(softReach.x, 0.0)).rgb * .125
      + texture2D(tLight, vUv - vec2(softReach.x, 0.0)).rgb * .125
      + texture2D(tLight, vUv + vec2(0.0, softReach.y)).rgb * .125
      + texture2D(tLight, vUv - vec2(0.0, softReach.y)).rgb * .125;

    // --- Distance ----------------------------------------------------------
    // The blurred buffer only knows where light is on screen. Everything the
    // rig covers also has a depth, so the falloff between the field's plane and
    // the surface it lands on is a real one: a face further back is dimmer, and
    // dims further as you lean away.
    // Where the rig covers nothing, the dense depth map knows the distance of
    // the room; without it, one flat assumption stands in for everything.
    vec2 frameUv = coverUv(vUv);
    float roomDepth = mix(BACKGROUND_DEPTH, texture2D(tDepth, frameUv).r, uHasDepth);
    float surfaceDepth = mix(roomDepth, mask.r, coverage);
    float depthGap = surfaceDepth - localDepth;
    float depthFalloff = 1.0 / (1.0 + (depthGap * depthGap) / (DEPTH_RANGE * DEPTH_RANGE));

    // The wave is anchored at the palms, so that is where its light is born.
    // This is a near-field source sitting on the hands themselves, which the
    // screen-space buffer alone renders too evenly.
    vec4 palm = texture2D(tPalm, vUv);
    // The smooth body of the palm surface, not its structure: relighting skin
    // with the animated ridges would make the light itself flicker.
    float palmLight = palm.b * frontCoverage * (1.0 - smoothstep(palm.r + .025, palm.r + .07, frontMask.r));
    palmLight *= (1.0 + uSeal * .3);
    float gL = luma(texture2D(tLight, vUv - vec2(uLightTexel.x, 0.0)).rgb);
    float gR = luma(texture2D(tLight, vUv + vec2(uLightTexel.x, 0.0)).rgb);
    float gD = luma(texture2D(tLight, vUv - vec2(0.0, uLightTexel.y)).rgb);
    float gU = luma(texture2D(tLight, vUv + vec2(0.0, uLightTexel.y)).rgb);
    // Direction and strength have to be separated. Feeding the raw gradient in
    // as a vector makes the diffuse term non-monotonic, which bands the falloff
    // wherever the gradient is steep. Take a unit direction, then let a bounded
    // tilt say how directional the light is here.
    vec2 gradient = vec2((gR - gL) * aspect, gU - gD);
    float gradientMag = length(gradient);
    vec2 lightXY = gradient / max(gradientMag, 1e-5);
    float tilt = clamp(gradientMag * 30.0, 0.0, .92);
    vec3 toLight = normalize(vec3(lightXY * tilt, .78));

    float diffuse = clamp((dot(normal, toLight) + .5) / 1.5, 0.0, 1.0);
    vec3 halfVec = normalize(toLight + vec3(0.0, 0.0, 1.0));
    float specular = pow(max(dot(normal, halfVec), 0.0), 12.0) * .07;

    // --- The field as a source standing in the room ----------------------
    // The blurred buffer above only knows where light is *on screen*, so a face
    // beside the field receives almost nothing from it however bright it is.
    // This term treats the field as what it is: a small bright thing at a known
    // place in the room. Every surface with a depth gets light from it by
    // distance and orientation, which is how a struck match lights a room --
    // not only the part of the wall it happens to overlap.
    vec3 surfacePoint = unprojectPalm(vUv * 2.0 - 1.0, surfaceDepth);
    vec3 sourcePoint = fieldPoint(.5, 0.0, -uSeal * uPalmScale * .18);
    vec3 toSource = sourcePoint - surfacePoint;
    float sourceDistance = max(length(toSource), .05);
    vec3 sourceDir = toSource / sourceDistance;
    // The same wrap the diffuse term uses: a broad near-field source does not
    // leave a hard terminator.
    float sourceLambert = clamp((dot(normal, sourceDir) + .35) / 1.35, 0.0, 1.0);
    float sourceFalloff = 1.0 /
      (1.0 + sourceDistance * sourceDistance / (NEAR_FIELD_RADIUS * NEAR_FIELD_RADIUS));
    float matchLight = sourceLambert * sourceFalloff * uPresence
      * (.4 + uEnergy * 1.2 + uSeal * .35 + uCritical * .75);

    // A brighter wave throws more light, not just a brighter wave.
    float emissionGain = LIGHT_GAIN * (.65 + uEnergy * 1.0 + uCritical * .6);
    vec3 tint = fieldTint(uEnergy);
    vec3 lit = irradiance * emissionGain * depthFalloff * (diffuse + specular);
    lit += tint * matchLight * MATCH_GAIN;
    // Palms get their own contribution, still shaded and still attenuated by
    // distance, so it reads as light landing on skin rather than a decal.
    lit += tint * palmLight * depthFalloff * diffuse * (.09 + uEnergy * .18) * uPresence;
    lit = lit / (1.0 + lit * .62);

    vec3 albedo = mix(texture2D(tEnvironment, vUv).rgb, refracted, clamp(lens + claspLens, 0.0, 1.0));
    vec3 emissive = texture2D(tEmissive, vUv).rgb;
    // A diffuse, additive scattering lobe lets red light lift dark skin and
    // fabric while preserving the original texture beneath it.
    // Skin scatters this light; hair and clothing largely do not. With the
    // multiclass silhouette the difference is known, so the diffuse lobe is
    // held back off fabric instead of washing the whole subject evenly.
    float skin = texture2D(tPerson, frameUv).g;
    float material = mix(1.0, .45 + .8 * skin, uHasSkin);
    vec3 scattering = irradiance * depthFalloff * coverage * material * (1.3 + uEnergy * 2.6);
    // Skin is not opaque; a near source bleeds a little through it wherever it
    // lands, which is what keeps the match light off the surface of a mask.
    scattering += tint * matchLight * material * coverage * .22;
    vec3 color = albedo * (uAmbient + lit) + scattering;
    color += emissive;

    // Backlight. Light wraps a silhouette where the surface turns away from the
    // viewer, so the strength is a grazing term off the real normal rather than
    // a stamp along the mask's edge. The irradiance is sampled a little way
    // outside the surface, along that normal, which is where the light
    // actually is when it is behind the hand.
    float grazing = pow(1.0 - abs(normal.z), 2.4);
    vec2 outward = normalize(rigNxy + vec2(1e-5)) * uLightTexel * 4.0;
    vec3 behindLight = texture2D(tLight, clamp(vUv + outward, .002, .998)).rgb;
    float wrap = frontCoverage * inFront * grazing * luma(behindLight) * RIM_GAIN * depthFalloff;
    // A thin specular catch right on the edge keeps it from reading as a decal.
    // Sealed, the same edges are where light escapes between the fingers, so
    // the catch is where a clasp shows its gaps.
    float edgeCatch = rimEdge * inFront * luma(behindLight) * (1.0 + uSeal * 1.8);
    color += mix(SILVER, tint, .4) * (wrap + edgeCatch) * uPresence;

    // Light sealed between the palms has nowhere to go but through them. Thin
    // parts of a silhouette pass the most, so the grazing term carries this
    // too; the result is hands lit from the inside rather than a bridge drawn
    // across them. It is a compositing cue, not a model of tissue scattering.
    // The falloff below keeps it on the hands doing the holding, and nothing
    // here is gated on visibility: the whole point is light reaching skin that
    // is in front of the knot.
    float held = exp(-dot(knotOffset, knotOffset) / max(uPalmScale * uPalmScale * 2.2, .0001));
    float transmit = uSeal * frontCoverage * held * luma(irradiance) * SEAL_TRANSMIT
      * (.3 + .7 * grazing) * depthFalloff * uPresence * (1.0 + uSeal * .6);
    color += mix(tint, SILVER, .2) * transmit;

    gl_FragColor = vec4(color, 1.0);
  }
`;

const filamentVertexShader = /* glsl */ `
  attribute float aT;
  attribute float aSide;
  uniform float uEnergy;
  uniform float uLayer;
  uniform float uTime;
  uniform float uSeal;
  varying float vSide;
  varying float vT;
  varying float vDensity;
  varying float vDepth;
  varying float vKnot;
  varying float vPhase;

  ${waveChunk}
  ${frameChunk}
  ${volumeChunk}

  void main() {
    float aspect = aspectOf();
    vec2 left = toSquare(uLeft, aspect);
    vec2 right = toSquare(uRight, aspect);
    vec2 axis = right - left;
    float span = max(length(axis), .05);
    vec2 dir = axis / span;
    vec2 nrm = vec2(-dir.y, dir.x);

    float amp = amplitudeFor(span);
    float lag = uLayer * .22;
    float y = psi(aT, lag) * amp;
    float slope = dpsi(aT, lag) * amp / span;

    // Widen across the true tangent so the strand keeps an even weight where
    // the curve is steep instead of pinching.
    vec2 tangent = normalize(dir + nrm * slope);
    vec2 wide = vec2(-tangent.y, tangent.x);
    float halfWidth = (span * .0045 + .0015) * mix(1.0, 5.0, step(.8, abs(uLayer)));

    float envelope = sin(PI * aT);
    // Approaching collapse the layers draw in toward the central filament, so
    // the bundle reads as intricate rather than merely smaller.
    float condense = 1.0 - uCritical * .58;
    float lateral = y + uLayer * amp * .16 * envelope * condense;
    float zOffset = uLayer * amp * .6 * envelope * condense + psi(aT, lag + 1.57) * amp * .22;
    vec3 point = fieldPoint(aT, lateral, zOffset);
    vKnot = smoothstep(0.0, .7, uSeal);
    float angle = aT * PI * 2.0 + uLayer * .7 + uTime * .35;
    float radius = uPalmScale * (.22 + .04 * sin(angle * 3.0));
    vec3 knot = fieldPoint(.5, sin(angle) * radius, cos(angle) * radius - uPalmScale * .18);
    knot.x += sin(angle * 2.0) * radius * .3 * cameraDistance(.5) / focal();
    point = mix(point, knot, vKnot);
    vec4 projected = projectField(point);
    projected.xy += toNdc(wide * aSide * halfWidth, aspect) * projected.w;
    gl_Position = projected;
    vDepth = nearAt(point);
    vSide = aSide;
    vT = aT;
    vPhase = psiPhase(aT, lag);
    vDensity = density(aT);
  }
`;

const filamentFragmentShader = /* glsl */ `
  precision highp float;
  uniform float uEnergy;
  uniform float uPresence;
  uniform float uLayer;
  uniform float uSeal;
  varying float vSide;
  varying float vT;
  varying float vDensity;
  varying float vDepth;
  varying float vKnot;
  varying float vPhase;

  ${paletteChunk}
  ${occlusionChunk}

  void main() {
    float edge = pow(max(1.0 - abs(vSide), 0.0), 1.6);
    float ends = smoothstep(0.0, .02, vT) * smoothstep(0.0, .02, 1.0 - vT);
    // Hue is the phase; brightness is the density. The antinodes still burn
    // toward white because that is where the amplitude is, and the flanks keep
    // the colour of whatever phase they are passing through.
    vec3 color = mix(phaseTint(vPhase), SILVER, .06 + .3 * vDensity);
    float alpha = edge * ends * uPresence
      * (.26 + .62 * vDensity)
      * mix(1.0, .42, abs(uLayer))
      * (.5 + uEnergy * .5)
      * mix(1.0, .7 * (1.0 - smoothstep(.2, .8, abs(uLayer))), vKnot)
      * mix(1.0, .16, step(.8, abs(uLayer)))
      * visibilityAt(gl_FragCoord.xy / uRenderResolution, vDepth);
    gl_FragColor = vec4(color * (1.0 + uEnergy * .4), alpha);
  }
`;

const particleVertexShader = /* glsl */ `
  attribute float aT;
  attribute float aAcross;
  attribute float aSeed;
  uniform float uTime;
  uniform float uEnergy;
  uniform float uPresence;
  uniform float uPixelRatio;
  uniform float uSeal;
  varying float vAlpha;
  varying float vSeed;
  varying float vDepth;
  varying float vPhase;

  ${waveChunk}
  ${frameChunk}
  ${volumeChunk}

  void main() {
    float aspect = aspectOf();
    vec2 left = toSquare(uLeft, aspect);
    vec2 right = toSquare(uRight, aspect);
    vec2 axis = right - left;
    float span = max(length(axis), .05);
    vec2 dir = axis / span;
    vec2 nrm = vec2(-dir.y, dir.x);

    float amp = amplitudeFor(span);
    float lag = (aSeed - .5) * .24;
    float y = psi(aT, lag) * amp;

    // Two populations: grains riding the curve, and a looser cloud filling the
    // well. Both are lit by |psi|^2, so nodes stay genuinely empty.
    float cloud = step(.55, aSeed);
    float envelope = sin(PI * aT);
    float spread = mix(.18, .7, cloud) * envelope * (1.0 - uCritical * .45);
    float lateral = y + aAcross * amp * spread;
    float zOffset = cos(aSeed * 31.0 + uTime * .2) * amp * .42 * envelope;
    vec3 point = fieldPoint(aT, lateral, zOffset);
    float swirl = uTime * (.4 + uEnergy) + aSeed * 6.2831;
    float radius = uPalmScale * (.1 + .26 * aSeed);
    vec3 knot = fieldPoint(.5 + cos(swirl) * .09,
      sin(swirl) * radius, cos(swirl * 1.3) * radius - uPalmScale * .18);
    point = mix(point, knot, smoothstep(0.0, .7, uSeal));
    vPhase = psiPhase(aT, lag);
    vDepth = nearAt(point);
    vAlpha = uPresence * density(aT)
      * mix(1.0, .55, cloud)
      * exp(-aAcross * aAcross * mix(3.0, 1.1, cloud));
    vSeed = aSeed;
    gl_Position = projectField(point);
    gl_PointSize = uPixelRatio * (1.1 + uEnergy + (1.0 - cloud) * .6) * clamp(2.4 / max(-point.z, .5), .65, 1.8);
  }
`;

const particleFragmentShader = /* glsl */ `
  precision highp float;
  uniform float uEnergy;
  varying float vAlpha;
  varying float vSeed;
  varying float vDepth;
  varying float vPhase;

  ${paletteChunk}
  ${occlusionChunk}

  void main() {
    float grain = 1.0 - smoothstep(.05, .5, length(gl_PointCoord - .5));
    float alpha = grain * vAlpha * (.2 + uEnergy * .32) * visibilityAt(gl_FragCoord.xy / uRenderResolution, vDepth);
    if (alpha < .004) discard;
    // Each grain carries the phase where it sits, so the cloud shows the wave
    // turning over rather than a single flat tint.
    vec3 color = mix(phaseTint(vPhase), vec3(1.0), .04 + vSeed * .2);
    gl_FragColor = vec4(color * (1.0 + uEnergy * .7), alpha);
  }
`;

// Final grade: roll the additive highlights off instead of letting them clip
// to flat white, then vignette and grain the whole frame together.
const gradeFragmentShader = /* glsl */ `
  precision highp float;
  uniform sampler2D tDiffuse;
  uniform sampler2D tVisible;
  uniform sampler2D tLight;
  uniform vec2 uResolution;
  uniform float uTime;
  uniform float uEnergy;
  uniform float uSeal;
  uniform float uCritical;
  uniform float uPresence;
  varying vec2 vUv;

  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }
  float luma(vec3 c) { return dot(c, vec3(.299, .587, .114)); }
  vec3 sourceAt(vec2 uv) {
    vec3 light = texture2D(tVisible, clamp(uv, .001, .999)).rgb;
    return light * smoothstep(.12, .75, luma(light));
  }
  void main() {
    vec2 pixel = 1.0 / max(uResolution, vec2(1.0));
    float litArea = smoothstep(.0005, .025, luma(texture2D(tLight, vUv).rgb));
    vec2 radial = vUv - .5;
    vec2 fringe = normalize(radial + vec2(.001)) * pixel
      * (.35 + uEnergy * 1.4 + uSeal * .6 + uCritical * .8) * litArea * uPresence;
    // Very small, source-local spectral separation rather than global blur.
    vec3 color = vec3(texture2D(tDiffuse, vUv + fringe).r,
      texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - fringe).b);
    vec3 flare = vec3(0.0);
    for (int i = -6; i <= 6; i++) {
      float tap = float(i);
      vec2 reach = vec2(tap * (8.0 + uEnergy * 10.0) * pixel.x, 0.0);
      flare += sourceAt(vUv + reach) * exp(-abs(tap) * .48);
    }
    // Thin anamorphic streaks and a soft halo grow with confinement.
    color += flare * (.035 + uEnergy * .065 + uSeal * .025 + uCritical * .05) * uPresence;
    // Roll highlights off on luminance rather than per channel: mapping each
    // channel separately pulls a bright coloured source toward white, which is
    // where the hue was being lost.
    float bright = luma(color);
    float rolled = 1.0 - exp(-bright);
    color *= mix(1.0, rolled / max(bright, 1e-4), smoothstep(.42, 1.35, bright));
    vec2 offset = vUv - .5;
    offset.x *= uResolution.x / max(uResolution.y, 1.0);
    color *= .9 + .1 * (1.0 - smoothstep(.65, 1.4, length(offset)));
    color += (hash(gl_FragCoord.xy + fract(uTime) * 431.0) - .5) * .008;
    gl_FragColor = vec4(color, 1.0);
  }
`;

// Bones of the hand, and how thick each is relative to the knuckle span. The
// rig is built from these as real geometry so the depth buffer resolves a hand
// covering its own palm without any special casing.
const BONES: [number, number, number][] = [
  [1, 2, .35], [2, 3, .29], [3, 4, .24],
  [5, 6, .24], [6, 7, .21], [7, 8, .18],
  [9, 10, .25], [10, 11, .22], [11, 12, .19],
  [13, 14, .24], [14, 15, .21], [15, 16, .18],
  [17, 18, .22], [18, 19, .19], [19, 20, .16],
];

const RIG_STEPS = 7;
// Ping-pong pairs for the irradiance blur. More widens the light's reach; the
// quality profile chooses how many.
const DEFAULT_BLUR_PASSES = 6;
const MAX_RIG_NODES = 2 * BONES.length * RIG_STEPS + 8;
/** Landmarks the face model can return; 468, or 478 with irises. */
const MAX_FACE_VERTICES = 478;
/** Cheek to cheek on the canonical face; the same figure the tracker uses. */
const FACE_WIDTH_MM = 145;

const rigVertexShader = /* glsl */ `
  varying float vNear;
  varying vec3 vNormal;
  void main() {
    vec4 world = instanceMatrix * vec4(position, 1.0);
    // Nearness is read back out of the world z the instance was placed at, so a
    // sphere carries its own curvature into the depth it reports.
    vNear = world.z * .5 + .5;

    // Normals under a non-uniform scale need the inverse scale, not the matrix.
    // GLSL ES has no inverse(), so the basis is pulled apart by hand: column
    // lengths are the scale, the normalised columns are the rotation.
    mat3 m = mat3(instanceMatrix);
    vec3 scale = vec3(length(m[0]), length(m[1]), length(m[2]));
    mat3 rotation = mat3(m[0] / scale.x, m[1] / scale.y, m[2] / scale.z);
    vNormal = normalize(rotation * (normal / scale));

    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const rigFragmentShader = /* glsl */ `
  precision highp float;
  varying float vNear;
  varying vec3 vNormal;
  void main() {
    // r: nearness, gb: the normal's screen-facing components, a: coverage.
    vec3 n = normalize(vNormal);
    gl_FragColor = vec4(clamp(vNear, 0.0, 1.0), n.x * .5 + .5, n.y * .5 + .5, 1.0);
  }
`;

function createFilamentGeometry(segments: number) {
  const vertexCount = (segments + 1) * 2;
  const positions = new Float32Array(vertexCount * 3);
  const ts = new Float32Array(vertexCount);
  const sides = new Float32Array(vertexCount);
  const indices: number[] = [];

  for (let i = 0; i <= segments; i += 1) {
    const t = i / segments;
    for (let side = 0; side < 2; side += 1) {
      const index = i * 2 + side;
      positions[index * 3] = t * 2 - 1;
      ts[index] = t;
      sides[index] = side === 0 ? -1 : 1;
    }
    if (i < segments) {
      const a = i * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aT', new THREE.BufferAttribute(ts, 1));
  geometry.setAttribute('aSide', new THREE.BufferAttribute(sides, 1));
  geometry.setIndex(indices);
  return geometry;
}

function seededRandom(seed: number) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createParticleGeometry(count: number) {
  const random = seededRandom(0x2c73a1e5);
  const positions = new Float32Array(count * 3);
  const ts = new Float32Array(count);
  const across = new Float32Array(count);
  const seeds = new Float32Array(count);

  for (let i = 0; i < count; i += 1) {
    ts[i] = random();
    across[i] = (random() * 2 - 1) * Math.pow(random(), .55);
    seeds[i] = random();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aT', new THREE.BufferAttribute(ts, 1));
  geometry.setAttribute('aAcross', new THREE.BufferAttribute(across, 1));
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 8);
  return geometry;
}

export function createWaveEngine(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  profile: QualityProfile,
  onFrameRate?: (fps: number) => void,
  options: WaveEngineOptions = {},
): WaveEngine {
  let requestedCameraCompositing = options.cameraCompositing ?? true;
  let diagnosticStage: WaveDiagnosticStage = 'composite';
  let cameraCompositing = requestedCameraCompositing;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    // A transparent drawing buffer is the iOS escape hatch: the native video
    // stays below it even when a camera texture cannot be sampled by WebGL.
    alpha: true,
    premultipliedAlpha: false,
    antialias: false,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(cameraCompositing ? 0x010304 : 0x000000, cameraCompositing ? 1 : 0);
  // Older iPad GPUs can sample half-float textures but cannot render into
  // them. Falling back to 8-bit buffers preserves the whole interaction,
  // minus a little highlight headroom, instead of producing a black canvas.
  const supportsHdrTargets = renderer.capabilities.isWebGL2
    ? renderer.extensions.has('EXT_color_buffer_float')
    : renderer.extensions.has('EXT_color_buffer_half_float');
  const renderTargetType = supportsHdrTargets ? THREE.HalfFloatType : THREE.UnsignedByteType;

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -2, 2);
  const envScene = new THREE.Scene();
  // The wave's light is rendered on its own so it can be blurred into an
  // irradiance buffer; mainScene only holds the composite.
  const emissiveScene = new THREE.Scene();
  const mainScene = new THREE.Scene();
  const screenGeometry = new THREE.PlaneGeometry(2, 2);

  const videoTexture = new THREE.VideoTexture(video);
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;

  // Shared state. Every material reads the same objects, so one update per
  // frame reaches all of them.
  const resolution = new THREE.Vector2(1, 1);
  const weights = new Float32Array(MODES);
  const phases = new Float32Array(MODES);
  const modePhase = new Float32Array(MODES);
  const currentLeft = new THREE.Vector2(-.34, 0);
  const currentRight = new THREE.Vector2(.34, 0);
  const texel = new THREE.Vector2(1, 1);
  const occluderTexel = new THREE.Vector2(1 / 320, 1 / 180);
  const emission = new THREE.Vector3(1, .4, .32);
  const shared = {
    uPhaseChroma: { value: PHASE_CHROMA },
    uResolution: { value: resolution },
    uRenderResolution: { value: new THREE.Vector2(1, 1) },
    uEndpointDepth: { value: new THREE.Vector2(.5, .5) },
    uPalmScale: { value: .1 },
    uOcclusion: { value: 0 },
    tOccluder: { value: null as THREE.Texture | null },
    uTexel: { value: texel },
    uAmbient: { value: .55 },
    uSeal: { value: 0 },
    uEmission: { value: emission },
    uLeft: { value: currentLeft },
    uRight: { value: currentRight },
    uWeights: { value: weights },
    uPhases: { value: phases },
    uTime: { value: 0 },
    uConfinement: { value: 0 },
    uCritical: { value: 0 },
    uEnergy: { value: 0 },
    uPresence: { value: 0 },
    uMode: { value: 0 },
    uPulse: { value: 0 },
  };

  const cameraUniforms = {
    tVideo: { value: videoTexture },
    uResolution: shared.uResolution,
    uVideoSize: { value: new THREE.Vector2(1280, 720) },
    uEnergy: shared.uEnergy,
    uCameraCompositing: { value: cameraCompositing ? 1 : 0 },
  };
  const cameraMaterial = new THREE.ShaderMaterial({
    vertexShader: screenVertexShader,
    fragmentShader: cameraFragmentShader,
    uniforms: cameraUniforms,
    depthTest: false,
    depthWrite: false,
  });
  envScene.add(new THREE.Mesh(screenGeometry, cameraMaterial));

  function updateCameraCompositing() {
    cameraCompositing = requestedCameraCompositing && diagnosticStage === 'composite';
    cameraUniforms.uCameraCompositing.value = cameraCompositing ? 1 : 0;
    renderer.setClearColor(cameraCompositing ? 0x010304 : 0x000000, cameraCompositing ? 1 : 0);
  }

  const environmentTarget = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
  });
  const cameraProbeTarget = new THREE.WebGLRenderTarget(8, 8, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
  });

  // --- Hand rig, rendered as depth-tested geometry into its own target ------
  const rigScene = new THREE.Scene();
  const rigCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -2, 2);
  const rigTarget = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: true,
  });
  const frontTarget = rigTarget.clone();
  shared.tOccluder.value = frontTarget.texture;
  const palmTarget = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true });
  const palmScene = new THREE.Scene();
  const palmGeometry = new THREE.BufferGeometry();
  // Two hands: one sector per silhouette sample across three ring bands, two
  // triangles each, plus one quad per finger bone.
  const PALM_VERTS = 2 * (PALM_CONTOUR_POINTS * 3 * 6 + BONES.length * 6);
  const palmPositions = new Float32Array(PALM_VERTS * 3);
  const palmGlows = new Float32Array(PALM_VERTS);
  const palmFlows = new Float32Array(PALM_VERTS * 2);
  const palmFields = new Float32Array(PALM_VERTS);
  const dynamic = (data: Float32Array, size: number) =>
    new THREE.BufferAttribute(data, size).setUsage(THREE.DynamicDrawUsage);
  palmGeometry.setAttribute('position', dynamic(palmPositions, 3));
  palmGeometry.setAttribute('aGlow', dynamic(palmGlows, 1));
  palmGeometry.setAttribute('aFlow', dynamic(palmFlows, 2));
  palmGeometry.setAttribute('aField', dynamic(palmFields, 1));
  const palmMaterial = new THREE.ShaderMaterial({
    vertexShader: palmVertexShader,
    fragmentShader: palmFragmentShader,
    uniforms: { uTime: shared.uTime, uEnergy: shared.uEnergy, uSeal: shared.uSeal },
    side: THREE.DoubleSide,
  });
  /**
   * A landmark surface written into the geometry buffer: r is nearness, gb the
   * screen-facing components of the normal, a coverage.
   *
   * The normal is the screen-space derivative of the interpolated position,
   * which is only meaningful if z is in the same units as x and y. Nearness
   * spans a metre and a bit over 0..1 while x and y span the viewport, so
   * `uDepthToNdc` lifts it into their units first. Without that term every
   * surface reports as facing the camera, which is the real reason
   * landmark-driven relighting ends up looking like a decal.
   *
   * `uDepthBias` pushes the surface back before it is allowed to cut the
   * field. Hands use none; the face, whose distance is the least certain thing
   * in the scene, uses enough that a close call goes to the field.
   */
  const createSurfaceMaterial = () => new THREE.ShaderMaterial({
    vertexShader: /* glsl */ `
      varying float vDepth;
      varying vec3 vSurface;
      uniform vec2 uResolution;
      uniform float uDepthToNdc;
      uniform float uDepthBias;
      void main() {
        vDepth = position.z - uDepthBias;
        vSurface = vec3(position.x * uResolution.x / uResolution.y, position.y, position.z * uDepthToNdc);
        // Match the rig camera's projection: world z spans -1..1 inside
        // orthographic near/far -2..2, so clip z is -world.z / 2.
        gl_Position = vec4(position.xy, .5 - vDepth, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      varying float vDepth;
      varying vec3 vSurface;
      void main() {
        vec3 n = normalize(cross(dFdx(vSurface), dFdy(vSurface)));
        n *= n.z < 0.0 ? -1.0 : 1.0;
        gl_FragColor = vec4(clamp(vDepth, 0.0, 1.0), n.xy * .5 + .5, 1.0);
      }
    `,
    uniforms: {
      uResolution: shared.uResolution,
      uDepthToNdc: { value: 1 },
      uDepthBias: { value: 0 },
    },
    side: THREE.DoubleSide,
  });
  const palmRigMaterial = createSurfaceMaterial();
  const faceSurfaceMaterial = createSurfaceMaterial();
  const palmRigMesh = new THREE.Mesh(palmGeometry, palmRigMaterial);
  palmRigMesh.frustumCulled = false;
  rigScene.add(palmRigMesh);
  const palmMesh = new THREE.Mesh(palmGeometry, palmMaterial);
  palmMesh.frustumCulled = false;
  palmScene.add(palmMesh);
  let trackedPalms: PalmFrame[] = [];
  let trackedScales: number[] = [];
  let fingerEmission = profile.fingerEmission;
  // Where the field's light is born, in the tracker's own screen-plus-nearness
  // space. The palm and finger surfaces are shaded against this, so the same
  // position that draws the wave is the position that lights the hands.
  const fieldAnchor = { x: 0, y: 0, z: .5 };

  // --- Face, as an actual surface -----------------------------------------
  // 852 triangles from the landmark tesselation, with normals taken from
  // screen-space derivatives of the interpolated position. That is what gives
  // a cheekbone an orientation of its own; an ellipsoid cannot have one.
  const faceGeometry = new THREE.BufferGeometry();
  const facePositions = new Float32Array(MAX_FACE_VERTICES * 3);
  faceGeometry.setAttribute('position', dynamic(facePositions, 3));
  faceGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 4);
  let faceIndices: Uint16Array | null = null;
  const faceMesh = new THREE.Mesh(faceGeometry, faceSurfaceMaterial);
  faceMesh.frustumCulled = false;
  faceMesh.visible = false;
  rigScene.add(faceMesh);

  // Coverage the hand rig cannot reach: torso, arms, shoulders, hair. Drawn
  // first and depth-tested, so the rig's per-finger depth wins wherever the
  // two overlap and the body only fills in behind it.
  const personTexel = new THREE.Vector2(1 / 320, 1 / 180);
  const depthTexel = new THREE.Vector2(1 / 252, 1 / 252);
  const depthUniform = { value: null as THREE.Texture | null };
  const hasDepthUniform = { value: 0 };
  const personMaterial = new THREE.ShaderMaterial({
    vertexShader: personVertexShader,
    fragmentShader: personFragmentShader,
    uniforms: {
      tPerson: { value: null as THREE.Texture | null },
      tDepth: depthUniform,
      uResolution: shared.uResolution,
      uVideoSize: cameraUniforms.uVideoSize,
      uPersonTexel: { value: personTexel },
      uDepthTexel: { value: depthTexel },
      uHasDepth: hasDepthUniform,
      uBodyDepth: { value: .1 },
    },
  });
  const personMesh = new THREE.Mesh(screenGeometry, personMaterial);
  personMesh.frustumCulled = false;
  personMesh.renderOrder = -1;
  personMesh.visible = false;
  rigScene.add(personMesh);
  let personTexture: THREE.DataTexture | null = null;
  let personVersion = -1;
  let personPacked = new Uint8Array(0);
  let depthTexture: THREE.DataTexture | null = null;
  let depthVersion = -1;

  const rigMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 12, 8),
    new THREE.ShaderMaterial({
      vertexShader: rigVertexShader,
      fragmentShader: rigFragmentShader,
      transparent: false,
    }),
    MAX_RIG_NODES,
  );
  rigMesh.frustumCulled = false;
  rigMesh.count = 0;
  rigScene.add(rigMesh);
  const rigMatrix = new THREE.Matrix4();
  const rigPosition = new THREE.Vector3();
  const rigScale = new THREE.Vector3();
  const rigQuaternion = new THREE.Quaternion();

  const emissiveTarget = new THREE.WebGLRenderTarget(1, 1, {
    // Preserve hot highlights through diffusion instead of clipping before bloom.
    type: renderTargetType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
  });
  const visibleTarget = emissiveTarget.clone();
  const blurTargets = [0, 1].map(() => new THREE.WebGLRenderTarget(1, 1, {
    type: renderTargetType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
  }));

  const emissiveMaterial = new THREE.ShaderMaterial({
    vertexShader: screenVertexShader,
    fragmentShader: emissiveFragmentShader,
    uniforms: { ...shared, tPalm: { value: palmTarget.texture } },
    depthTest: false,
    depthWrite: false,
  });
  const emissiveField = new THREE.Mesh(screenGeometry, emissiveMaterial);
  emissiveField.frustumCulled = false;
  emissiveField.renderOrder = -10;
  emissiveScene.add(emissiveField);

  const blurMaterial = new THREE.ShaderMaterial({
    vertexShader: screenVertexShader,
    fragmentShader: blurFragmentShader,
    uniforms: { tSource: { value: null as THREE.Texture | null }, uDirection: { value: new THREE.Vector2() } },
    depthTest: false,
    depthWrite: false,
  });
  const blurScene = new THREE.Scene();
  const blurQuad = new THREE.Mesh(screenGeometry, blurMaterial);
  blurQuad.frustumCulled = false;
  blurScene.add(blurQuad);

  const lightTexel = new THREE.Vector2(1, 1);
  const compositeMaterial = new THREE.ShaderMaterial({
    vertexShader: screenVertexShader,
    fragmentShader: compositeFragmentShader,
    uniforms: {
      ...shared,
      tEnvironment: { value: environmentTarget.texture },
      tEmissive: { value: visibleTarget.texture },
      tPalm: { value: palmTarget.texture },
      tLight: { value: blurTargets[0].texture },
      tOccluder: { value: rigTarget.texture },
      tFrontOccluder: { value: frontTarget.texture },
      tPerson: { value: null as THREE.Texture | null },
      tDepth: depthUniform,
      uVideoSize: cameraUniforms.uVideoSize,
      uHasDepth: hasDepthUniform,
      uHasSkin: { value: 0 },
      uOccluderTexel: { value: occluderTexel },
      uLightTexel: { value: lightTexel },
      uHasOccluder: { value: 0 },
      uCameraCompositing: cameraUniforms.uCameraCompositing,
    },
    depthTest: false,
    depthWrite: false,
  });
  const composite = new THREE.Mesh(screenGeometry, compositeMaterial);
  composite.frustumCulled = false;
  mainScene.add(composite);

  const filamentGeometry = createFilamentGeometry(256);
  const filamentMaterials: THREE.ShaderMaterial[] = [];
  const layerCount = 9;
  for (let i = 0; i < layerCount; i += 1) {
    const material = new THREE.ShaderMaterial({
      vertexShader: filamentVertexShader,
      fragmentShader: filamentFragmentShader,
      uniforms: { ...shared, uLayer: { value: (i / (layerCount - 1)) * 2 - 1 } },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    filamentMaterials.push(material);
    const strand = new THREE.Mesh(filamentGeometry, material);
    strand.frustumCulled = false;
    strand.renderOrder = i;
    emissiveScene.add(strand);
  }

  let blurPasses = Math.max(1, Math.min(profile.blurPasses, DEFAULT_BLUR_PASSES));
  const fullParticleCount = Math.min(profile.particles, window.innerWidth < 720 ? 3600 : 8000);
  const particleGeometry = createParticleGeometry(fullParticleCount);
  const particleMaterial = new THREE.ShaderMaterial({
    vertexShader: particleVertexShader,
    fragmentShader: particleFragmentShader,
    uniforms: { ...shared, uPixelRatio: { value: 1 } },
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.frustumCulled = false;
  particles.renderOrder = layerCount + 1;
  emissiveScene.add(particles);

  const composerTarget = new THREE.WebGLRenderTarget(1, 1, { type: renderTargetType });
  const composer = new EffectComposer(renderer, composerTarget);
  composer.addPass(new RenderPass(mainScene, camera));
  const bloomPass = supportsHdrTargets ? new UnrealBloomPass(new THREE.Vector2(1, 1), .3, .4, .5) : null;
  if (bloomPass) composer.addPass(bloomPass);
  const gradePass = new ShaderPass({
    // ShaderPass clones the uniforms it is handed, so this pass is fed
    // explicitly each frame rather than sharing objects with the scene.
    uniforms: {
      tDiffuse: { value: null },
      uResolution: { value: new THREE.Vector2(1, 1) },
      uTime: { value: 0 },
    },
    vertexShader: screenVertexShader,
    fragmentShader: gradeFragmentShader,
  });
  gradePass.uniforms.tVisible = { value: visibleTarget.texture };
  gradePass.uniforms.tLight = { value: blurTargets[1].texture };
  gradePass.uniforms.uEnergy = { value: 0 };
  gradePass.uniforms.uSeal = { value: 0 };
  gradePass.uniforms.uCritical = { value: 0 };
  gradePass.uniforms.uPresence = { value: 0 };
  composer.addPass(gradePass);

  // Landmarks as delivered by the tracker, in screen pixels with nearness.
  let rigHands: RigPoint[][] = [];
  let rigFace: FieldTracking['face'] = null;

  const target = {
    left: new THREE.Vector2(-.34, 0),
    right: new THREE.Vector2(.34, 0),
    confinement: 0,
    presence: 0,
    mode: 0,
    seal: 0,
    pulse: 0,
    facing: 1,
    endpointDepth: new THREE.Vector2(.5, .5),
    state: 'dormant' as FieldTracking['state'],
  };
  let currentConfinement = 0;
  let currentPresence = 0;
  let currentMode = 0;
  let currentSeal = 0;
  let currentPulse = 0;
  let currentFacing = 0;
  let currentCritical = 0;
  let occluderActive = false;
  let handRigCount = 0;
  let lastTrackingAt = 0;
  let width = 1;
  let height = 1;
  let pixelRatio = 1;
  let animationFrame = 0;
  let destroyed = false;
  let renderedParticleCount = fullParticleCount;
  let fpsWindowStart = performance.now();
  let fpsFrames = 0;
  let lowFpsWindows = 0;
  // Timer over Clock: it hooks the Page Visibility API, so returning to a
  // backgrounded tab resumes instead of jumping the wave forward.
  const timer = new THREE.Timer();
  timer.connect(document);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function updateSpectrum(confinement: number, delta: number) {
    // Fixed mode populations: narrowing L alone raises kinetic energy.
    // Promoting n just because hands close would imply unmodelled excitation.
    let total = 0;
    for (let i = 0; i < MODES; i += 1) {
      weights[i] = Math.exp(-i * i / 1.6);
      total += weights[i];
    }
    for (let i = 0; i < MODES; i += 1) weights[i] /= total;

    // E grows with n and with confinement, so tighter wells visibly beat
    // faster. Phase is accumulated so changing the rate never jumps.
    for (let i = 0; i < MODES; i += 1) {
      const n = i + 1;
      modePhase[i] = (modePhase[i] + .28 * n * n * Math.pow(1 / relativeWellWidth(confinement), 2) * delta) % (Math.PI * 2);
      phases[i] = modePhase[i];
    }
  }

  function updateAllUniforms(time: number, delta: number) {
    const easing = (tau: number) => 1 - Math.exp(-delta / tau);
    // The tracker already filters the anchors. A second long lerp detaches
    // emission from moving palms; copy them and ease only appearance.
    currentLeft.copy(target.left);
    currentRight.copy(target.right);
    currentConfinement += (target.confinement - currentConfinement) * easing(.055);
    currentPresence += (target.presence - currentPresence) * easing(.12);
    currentMode += (target.mode - currentMode) * easing(.12);
    currentSeal += (target.seal - currentSeal) * easing(.05);
    // The impulse already decays in the state machine; follow it closely.
    currentPulse += (target.pulse - currentPulse) * easing(.04);
    currentFacing += (target.facing - currentFacing) * easing(.1);
    shared.uEndpointDepth.value.copy(target.endpointDepth);

    const energy = normalisedEnergy(currentConfinement);
    updateSpectrum(currentConfinement, delta * (reduceMotion ? .24 : 1));

    // The room falls dark as the well tightens, so the field ends up being
    // nearly the only light in it.
    //
    // The source's colour by state, chosen rather than predicted: a broad
    // state spills faint ice-blue, confinement drives it through silver to a
    // neutral white core, and only at high confinement does a little warmth
    // enter -- pale gold, kept well short of amber. The core stays neutral on
    // purpose. A bright source landing on skin reads as white with colour in
    // its falloff, and a saturated tint over a whole hand reads as a filter.
    const source = emissionFor(energy, currentSeal);
    emission.set(source.r, source.g, source.b);
    shared.uAmbient.value = .76 - currentPresence * (.12 + energy * .12);
    shared.uSeal.value = target.state === 'clasped' ? Math.max(currentSeal, .7) : currentSeal;

    // Near collapse: the well is very narrow and nothing is sealed yet. Taken
    // continuously off the dial rather than off the state flag, so the
    // treatment ramps in with the gesture; the flag exists for the copy, which
    // does need a discrete name for it.
    const nearCollapse = THREE.MathUtils.smoothstep(currentConfinement, .58, .95)
      * (1 - THREE.MathUtils.smoothstep(shared.uSeal.value, .72, 1));
    currentCritical += (nearCollapse - currentCritical) * easing(.07);

    shared.uTime.value = time;
    shared.uConfinement.value = currentConfinement;
    shared.uCritical.value = currentCritical;
    shared.uEnergy.value = energy;
    // Palms turned edge-on stop presenting a face for the field to span, so it
    // weakens rather than clinging to the tracked points. The floor is high
    // enough that a legitimate pose -- both palms forward, held overhead --
    // still carries a field; only a true edge-on hand fades it back.
    const facing = .55 + .45 * currentFacing;
    // Tracking loss fades the field at its last anchors; no screen-wide trace.
    shared.uPresence.value = currentPresence * facing;
    shared.uMode.value = currentMode;
    shared.uPulse.value = currentPulse;

    gradePass.uniforms.uTime.value = time;
    gradePass.uniforms.uResolution.value.copy(resolution);
    gradePass.uniforms.uEnergy.value = energy;
    gradePass.uniforms.uSeal.value = shared.uSeal.value;
    gradePass.uniforms.uCritical.value = currentCritical;
    gradePass.uniforms.uPresence.value = shared.uPresence.value;
    if (bloomPass) {
      bloomPass.strength = .22 + energy * .34 + currentSeal * .12 + currentPulse * .12
        + currentCritical * .2;
      bloomPass.radius = .58 + energy * .18;
    }
  }

  /**
   * Place a chain of spheres along a bone. Radius and depth are interpolated,
   * so a finger is a tapered solid rather than a flat stamp, and the depth
   * buffer sorts a hand against its own palm for free.
   */
  function addRigNode(index: number, a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }, t: number, radius: number, aspect: number) {
    const x = a.x + (b.x - a.x) * t;
    const y = a.y + (b.y - a.y) * t;
    const z = a.z + (b.z - a.z) * t;
    rigPosition.set(
      (x / width * 2 - 1) * aspect,
      1 - y / height * 2,
      z * 2 - 1,
    );
    rigScale.set(radius, radius, Math.min(radius * .24, .045));
    rigMatrix.compose(rigPosition, rigQuaternion, rigScale);
    rigMesh.setMatrixAt(index, rigMatrix);
  }

  function buildRig() {
    const aspect = width / Math.max(height, 1);
    let index = 0;
    for (const hand of rigHands) {
      if (hand.length < 21) continue;
      // Everything is sized off the knuckle span, so the rig scales with how
      // close the hand is to the camera.
      const spanPx = Math.hypot(hand[5].x - hand[17].x, hand[5].y - hand[17].y);
      const palmLength = Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y);
      const unit = (Math.max(spanPx, palmLength * .78) / Math.max(height, 1)) * 2;
      for (const [from, to, weight] of BONES) {
        for (let step = 0; step < RIG_STEPS; step += 1) {
          if (index >= MAX_RIG_NODES) break;
          addRigNode(index, hand[from], hand[to], step / (RIG_STEPS - 1), unit * weight * .5, aspect);
          index += 1;
        }
      }
    }
    handRigCount = index;

    // Metric scaling for the derivative normals. The tracker reports, per
    // hand, how many screen pixels one unit of nearness is worth at that
    // distance; converting that to viewport units puts z in the same units as
    // x and y, which is the only condition under which a screen-space
    // derivative is a surface normal at all.
    const ndcPerPixel = 2 / Math.max(height, 1);
    const pixelsPerNearness = trackedScales.length
      ? trackedScales.reduce((sum, value) => sum + value, 0) / trackedScales.length
      : height * 2;
    palmRigMaterial.uniforms.uDepthToNdc.value = pixelsPerNearness * ndcPerPixel;
    if (rigFace) {
      // Same construction for the head, from its own measured apparent width.
      const facePixelsPerMm = rigFace.radiusX * 2 / FACE_WIDTH_MM;
      faceSurfaceMaterial.uniforms.uDepthToNdc.value =
        NEARNESS_SPAN_MM * facePixelsPerMm * ndcPerPixel;
    }

    rigMesh.count = index;
    rigMesh.instanceMatrix.needsUpdate = true;
    occluderActive = index > 0 || personMesh.visible || faceMesh.visible;
    rigCamera.left = -aspect;
    rigCamera.right = aspect;
    rigCamera.updateProjectionMatrix();
    compositeMaterial.uniforms.uHasOccluder.value = occluderActive ? 1 : 0;
    rigQuaternion.identity();
  }

  /**
   * The palm as a lit surface. Rings are offset inward along the silhouette's
   * own normal rather than scaled toward a centre point: scaling drives any
   * outline to a disc as it shrinks, so the bright core -- which is the part
   * the eye actually reads -- ends up circular however good the outline was.
   * Offsetting keeps the palm's shape all the way in.
   *
   * Brightness is N.L against the real field position, in a space where the
   * landmarks' nearness has been scaled into the same units as their pixels.
   * A palm turned edge-on therefore dims because it is turned away, not only
   * because its outline has collapsed.
   */
  const PALM_RINGS = [0, .2, .5, .82];
  const RING_GLOW = [0, .92, 1, .62];

  function lightAt(
    px: number, py: number, pz: number,
    normal: RigPoint, scale: number, radius: number,
  ) {
    const dx = fieldAnchor.x - px;
    const dy = fieldAnchor.y - py;
    const dz = (fieldAnchor.z - pz) * scale;
    const distance = Math.hypot(dx, dy, dz) || 1;
    const nDotL = (normal.x * dx + normal.y * dy + normal.z * dz) / distance;
    // A little wrap, so a surface just past the terminator is not black. The
    // field is a broad near-field source, not a point at infinity.
    const wrapped = Math.max((nDotL + .28) / 1.28, 0);
    const reach = radius * 4.5;
    return wrapped / (1 + (distance * distance) / (reach * reach));
  }

  function buildPalmSurfaces() {
    let count = 0;
    const palms = trackedPalms.slice(0, 2);
    const push = (
      point: { x: number; y: number; z: number },
      glow: number,
      angle: number,
      ringT: number,
      field: number,
    ) => {
      if (count >= PALM_VERTS) return;
      palmPositions[count * 3] = point.x / width * 2 - 1;
      palmPositions[count * 3 + 1] = 1 - point.y / height * 2;
      palmPositions[count * 3 + 2] = point.z;
      palmGlows[count] = glow;
      palmFlows[count * 2] = angle;
      palmFlows[count * 2 + 1] = ringT;
      palmFields[count] = field;
      count += 1;
    };

    palms.forEach((palm, index) => {
      const scale = trackedScales[index] ?? 0;
      const other = palms[1 - index];
      // Toward the opposite palm when there are two; otherwise out along the
      // hand's own long axis, which is where a single-hand pinch sits.
      const axis = other
        ? { x: other.center.x - palm.center.x, y: other.center.y - palm.center.y }
        : { x: palm.up.x, y: palm.up.y };
      const length = Math.hypot(axis.x, axis.y) || 1;
      const toField = { x: axis.x / length, y: axis.y / length };

      const rings = PALM_RINGS.map(fraction =>
        fraction === 0 ? palm.contour : insetContour(palm, palm.radius * fraction));
      const samples = palm.contour.length;
      const facing = lightAt(palm.center.x, palm.center.y, palm.center.z, palm.normal, scale, palm.radius);
      const surface = facing * palm.openness;

      const vertex = (corner: number, ring: number) => {
        const i = ((corner % samples) + samples) % samples;
        const point = rings[ring][i];
        const outX = palm.contour[i].x - palm.center.x;
        const outY = palm.contour[i].y - palm.center.y;
        const reach = Math.hypot(outX, outY) || 1;
        // Which part of the rim faces the way the field leaves the hand.
        const field = .5 + .5 * (outX * toField.x + outY * toField.y) / reach;
        push(point, RING_GLOW[ring] * surface, i / samples * Math.PI * 2, 1 - PALM_RINGS[ring], field);
      };

      for (let edge = 0; edge < samples; edge += 1) {
        for (let ring = 0; ring < rings.length - 1; ring += 1) {
          vertex(edge, ring);
          vertex(edge, ring + 1);
          vertex(edge + 1, ring + 1);
          vertex(edge, ring);
          vertex(edge + 1, ring + 1);
          vertex(edge + 1, ring);
        }
      }

      // Fingers, as their own quads along the bones. Each segment is shaded at
      // its own midpoint, so a finger reaching toward the field brightens on
      // its own while the one behind it does not -- which is the difference
      // between a lit hand and a lit blob.
      const hand = rigHands[index];
      if (!fingerEmission || !hand || hand.length < 21) return;
      const unit = Math.max(
        Math.hypot(hand[5].x - hand[17].x, hand[5].y - hand[17].y),
        Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y) * .78,
      );
      for (const [from, to, weight] of BONES) {
        const a = hand[from];
        const b = hand[to];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const span = Math.hypot(dx, dy) || 1;
        const half = unit * weight * .5;
        const nx = -dy / span * half;
        const ny = dx / span * half;
        const midGlow = lightAt((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2,
          palm.normal, scale, palm.radius) * palm.openness * .85;
        const angle = Math.atan2(dy, dx) + Math.PI;
        const corners = [
          { x: a.x + nx, y: a.y + ny, z: a.z },
          { x: a.x - nx, y: a.y - ny, z: a.z },
          { x: b.x - nx, y: b.y - ny, z: b.z },
          { x: b.x + nx, y: b.y + ny, z: b.z },
        ];
        const emit = (corner: number, ringT: number) =>
          push(corners[corner], midGlow, angle, ringT, .72);
        // Bright along the spine of the bone, falling off at its edges.
        emit(0, .1); emit(1, .1); emit(2, .1);
        emit(0, .1); emit(2, .1); emit(3, .1);
      }
    });

    palmGeometry.setDrawRange(0, count);
    palmGeometry.attributes.position.needsUpdate = true;
    palmGeometry.attributes.aGlow.needsUpdate = true;
    palmGeometry.attributes.aFlow.needsUpdate = true;
    palmGeometry.attributes.aField.needsUpdate = true;
    shared.uPalmScale.value = palms.length
      ? palms.reduce((sum, palm) => sum + palm.radius, 0) / palms.length / height * 2 : .08;
  }

  /**
   * Push the tracked face landmarks into the mesh the compositor lights.
   *
   * The hull is grown about the face's own centre before it is written. The
   * landmarks stop at the edge of the face, so the mesh alone leaves the
   * hairline, the jaw and the neck unlit -- a lit oval floating on an unlit
   * head. Since this surface only ever lights and never occludes, growing it
   * costs nothing and covers what the landmarks do not reach.
   */
  const FACE_LIGHT_EXPAND = 1.22;

  function buildFaceSurface() {
    if (!rigFace || !faceIndices) {
      faceMesh.visible = false;
      return;
    }
    const source = rigFace.vertices;
    const vertices = Math.min(rigFace.count, facePositions.length / 3);
    const centerX = rigFace.center.x;
    const centerY = rigFace.center.y;
    for (let i = 0; i < vertices; i += 1) {
      const x = centerX + (source[i * 3] - centerX) * FACE_LIGHT_EXPAND;
      const y = centerY + (source[i * 3 + 1] - centerY) * FACE_LIGHT_EXPAND;
      facePositions[i * 3] = x / width * 2 - 1;
      facePositions[i * 3 + 1] = 1 - y / height * 2;
      facePositions[i * 3 + 2] = source[i * 3 + 2];
    }
    faceGeometry.attributes.position.needsUpdate = true;
    faceMesh.visible = true;
  }

  /**
   * Upload a new silhouette, reallocating only when its size changes. Coverage
   * and skin travel together in one two-channel texture: the composite uses
   * the skin channel to keep a synthetic highlight on skin rather than
   * spreading it evenly over hair and clothing, which reflect nothing like it.
   */
  function updatePersonMask(mask: FieldTracking['person'], bodyDepth: number) {
    if (!mask || DIAGNOSTIC_STAGE_ORDER[diagnosticStage] < DIAGNOSTIC_STAGE_ORDER.segmentation) {
      personMesh.visible = false;
      return;
    }
    const pixels = mask.width * mask.height;
    const image = personTexture?.image as { width: number; height: number } | undefined;
    if (!personTexture || image?.width !== mask.width || image?.height !== mask.height) {
      personTexture?.dispose();
      personPacked = new Uint8Array(pixels * 2);
      personTexture = new THREE.DataTexture(personPacked, mask.width, mask.height, THREE.RGFormat);
      personTexture.minFilter = THREE.LinearFilter;
      personTexture.magFilter = THREE.LinearFilter;
      personTexture.unpackAlignment = 1;
      personMaterial.uniforms.tPerson.value = personTexture;
      compositeMaterial.uniforms.tPerson.value = personTexture;
      personVersion = -1;
    }
    if (mask.version !== personVersion) {
      personVersion = mask.version;
      for (let i = 0; i < pixels; i += 1) {
        personPacked[i * 2] = mask.data[i];
        personPacked[i * 2 + 1] = mask.skin[i];
      }
      personTexture.needsUpdate = true;
    }
    compositeMaterial.uniforms.uHasSkin.value = mask.multiclass ? 1 : 0;
    personMaterial.uniforms.uBodyDepth.value = bodyDepth;
    personMesh.visible = true;
  }

  /** Upload the aligned dense depth map, when the depth model is running. */
  function updateDepthMap(map: FieldTracking['depth']) {
    if (!map || DIAGNOSTIC_STAGE_ORDER[diagnosticStage] < DIAGNOSTIC_STAGE_ORDER.depth) {
      hasDepthUniform.value = 0;
      return;
    }
    const image = depthTexture?.image as { width: number; height: number } | undefined;
    if (!depthTexture || image?.width !== map.width || image?.height !== map.height) {
      depthTexture?.dispose();
      depthTexture = new THREE.DataTexture(map.data, map.width, map.height, THREE.RedFormat);
      depthTexture.minFilter = THREE.LinearFilter;
      depthTexture.magFilter = THREE.LinearFilter;
      depthTexture.unpackAlignment = 1;
      depthUniform.value = depthTexture;
      depthTexel.set(1 / map.width, 1 / map.height);
      depthVersion = -1;
    }
    if (map.version !== depthVersion) {
      depthVersion = map.version;
      depthTexture.needsUpdate = true;
    }
    hasDepthUniform.value = 1;
  }

  function resize() {
    width = canvas.clientWidth || window.innerWidth;
    height = canvas.clientHeight || window.innerHeight;
    pixelRatio = Math.min(window.devicePixelRatio, window.innerWidth < 720 ? 1.15 : 1.45);
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);
    environmentTarget.setSize(Math.round(width * pixelRatio), Math.round(height * pixelRatio));
    // The rig only needs enough resolution to resolve a finger edge.
    const rigWidth = Math.max(Math.round(Math.min(width, 960)), 2);
    rigTarget.setSize(rigWidth, Math.max(Math.round(rigWidth * height / Math.max(width, 1)), 2));
    frontTarget.setSize(rigTarget.width, rigTarget.height);
    occluderTexel.set(1 / rigTarget.width, 1 / rigTarget.height);
    personTexel.copy(occluderTexel);

    // Emissive stays sharp; the irradiance it is blurred into does not need to.
    emissiveTarget.setSize(Math.round(width * pixelRatio), Math.round(height * pixelRatio));
    visibleTarget.setSize(emissiveTarget.width, emissiveTarget.height);
    palmTarget.setSize(rigTarget.width, rigTarget.height);
    shared.uRenderResolution.value.set(emissiveTarget.width, emissiveTarget.height);
    const lightWidth = Math.max(Math.round(width / 4), 2);
    const lightHeight = Math.max(Math.round(height / 4), 2);
    blurTargets.forEach((blurTarget) => blurTarget.setSize(lightWidth, lightHeight));
    lightTexel.set(1 / lightWidth, 1 / lightHeight);
    composer.setPixelRatio(pixelRatio);
    composer.setSize(width, height);
    resolution.set(width, height);
    texel.set(1 / Math.max(width, 1), 1 / Math.max(height, 1));
    particleMaterial.uniforms.uPixelRatio.value = pixelRatio;
    buildPalmSurfaces();
  }

  function watchPerformance(now: number) {
    fpsFrames += 1;
    const elapsed = now - fpsWindowStart;
    if (elapsed < 1800) return;
    const fps = fpsFrames * 1000 / elapsed;
    onFrameRate?.(fps);
    lowFpsWindows = fps < 47 ? lowFpsWindows + 1 : 0;
    if (lowFpsWindows >= 2 && renderedParticleCount > 2400) {
      renderedParticleCount = Math.max(2400, Math.floor(renderedParticleCount * .68));
      particleGeometry.setDrawRange(0, renderedParticleCount);
      lowFpsWindows = 0;
    }
    fpsFrames = 0;
    fpsWindowStart = now;
  }

  /**
   * Compile every program once, before anything needs it. The face mesh and
   * the person plane are hidden until their models produce something, so
   * without this their shaders would first be built at the moment a face walks
   * into frame -- which is a visible hitch, and the worst possible time to
   * discover a shader will not link.
   */
  function warmUp() {
    const faceWasVisible = faceMesh.visible;
    const personWasVisible = personMesh.visible;
    const personTextureWas = personMaterial.uniforms.tPerson.value;
    const warmTexture = new THREE.DataTexture(new Uint8Array([0, 0]), 1, 1, THREE.RGFormat);
    warmTexture.needsUpdate = true;
    personMaterial.uniforms.tPerson.value = personTextureWas ?? warmTexture;
    faceMesh.visible = true;
    personMesh.visible = true;
    renderer.setRenderTarget(rigTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(rigScene, rigCamera);
    faceMesh.visible = faceWasVisible;
    personMesh.visible = personWasVisible;
    personMaterial.uniforms.tPerson.value = personTextureWas;
    warmTexture.dispose();
    renderer.setClearColor(cameraCompositing ? 0x010304 : 0x000000, cameraCompositing ? 1 : 0);
    renderer.setRenderTarget(null);
  }

  function render(now: number) {
    if (destroyed) return;
    timer.update(now);
    const scale = reduceMotion ? .24 : 1;
    const delta = Math.min(timer.getDelta(), .05);
    if (lastTrackingAt && now - lastTrackingAt > 500) {
      target.presence = 0; target.state = 'dormant'; target.seal = 0; target.pulse = 0;
      rigHands = []; rigFace = null; trackedPalms = []; trackedScales = [];
      buildPalmSurfaces(); buildFaceSurface();
      updatePersonMask(null, .1);
    }
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      cameraUniforms.uVideoSize.value.set(video.videoWidth, video.videoHeight);
    }
    updateAllUniforms(timer.getElapsed() * scale, delta);

    if (diagnosticStage === 'raw' || diagnosticStage === 'transparent' || diagnosticStage === 'tracking') {
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(null);
      renderer.clear();
      watchPerformance(now);
      animationFrame = requestAnimationFrame(render);
      return;
    }

    if (cameraCompositing) {
      renderer.setRenderTarget(environmentTarget);
      renderer.clear();
      renderer.render(envScene, camera);
    }

    // The rig is cleared to fully transparent so coverage reads as alpha.
    // Pass one is every surface in the scene, at its measured distance: it is
    // what the composite lights.
    buildRig();
    faceSurfaceMaterial.uniforms.uDepthBias.value = 0;
    renderer.setRenderTarget(rigTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    if (occluderActive) renderer.render(rigScene, rigCamera);

    // Pass two decides what may cut the field, and only the hands qualify.
    // The face is deliberately absent: the wave passes across a face rather
    // than behind it, so the head lights but never occludes. The person
    // silhouette is out too -- a mask carries no depth of its own -- and so is
    // the dense depth map, which is smooth across exactly the boundaries
    // occlusion depends on.
    const allNodes = rigMesh.count;
    const personWasVisible = personMesh.visible;
    const faceWasVisible = faceMesh.visible;
    rigMesh.count = handRigCount;
    personMesh.visible = false;
    faceMesh.visible = false;
    renderer.setRenderTarget(frontTarget);
    renderer.clear();
    if (handRigCount > 0) renderer.render(rigScene, rigCamera);
    rigMesh.count = allNodes;
    personMesh.visible = personWasVisible;
    faceMesh.visible = faceWasVisible;

    // The wave's own light, then the same buffer blurred down into the
    // irradiance the composite lights the room with.
    renderer.setClearColor(0x000000, 1);
    renderer.setRenderTarget(palmTarget);
    renderer.clear();
    renderer.render(palmScene, camera);
    shared.uOcclusion.value = 0;
    renderer.setRenderTarget(emissiveTarget);
    renderer.clear();
    renderer.render(emissiveScene, camera);

    blurMaterial.uniforms.tSource.value = emissiveTarget.texture;
    for (let pass = 0; pass < blurPasses; pass += 1) {
      const reach = 1 + pass * 2.4;
      for (let axis = 0; axis < 2; axis += 1) {
        const from = pass === 0 && axis === 0 ? emissiveTarget : blurTargets[(axis + 1) % 2];
        blurMaterial.uniforms.tSource.value = from.texture;
        blurMaterial.uniforms.uDirection.value.set(
          axis === 0 ? lightTexel.x * reach : 0,
          axis === 0 ? 0 : lightTexel.y * reach,
        );
        renderer.setRenderTarget(blurTargets[axis]);
        renderer.clear();
        renderer.render(blurScene, camera);
      }
    }
    compositeMaterial.uniforms.tLight.value = blurTargets[1].texture;
    // Render the visible field with per-fragment depths; hidden light remains
    // in the irradiance buffer for edge wrap and light leaking through gaps.
    shared.uOcclusion.value = 1;
    renderer.setRenderTarget(visibleTarget);
    renderer.clear();
    renderer.render(emissiveScene, camera);

    renderer.setClearColor(cameraCompositing ? 0x010304 : 0x000000, cameraCompositing ? 1 : 0);
    renderer.setRenderTarget(null);
    renderer.clear();
    if (cameraCompositing) composer.render();
    else renderer.render(mainScene, camera);

    watchPerformance(now);
    animationFrame = requestAnimationFrame(render);
  }

  resize();
  warmUp();
  window.addEventListener('resize', resize);
  animationFrame = requestAnimationFrame(render);

  return {
    setProfile(next) {
      blurPasses = Math.max(1, Math.min(next.blurPasses, DEFAULT_BLUR_PASSES));
      fingerEmission = next.fingerEmission;
      if (next.particles < renderedParticleCount) {
        renderedParticleCount = next.particles;
        particleGeometry.setDrawRange(0, renderedParticleCount);
      }
    },
    setCameraCompositing(enabled) {
      requestedCameraCompositing = enabled;
      updateCameraCompositing();
    },
    setDiagnosticStage(stage) {
      diagnosticStage = stage;
      updateCameraCompositing();
      if (DIAGNOSTIC_STAGE_ORDER[stage] < DIAGNOSTIC_STAGE_ORDER.segmentation) updatePersonMask(null, .1);
      if (DIAGNOSTIC_STAGE_ORDER[stage] < DIAGNOSTIC_STAGE_ORDER.depth) updateDepthMap(null);
    },
    getDiagnostics() {
      return {
        webgl2: renderer.capabilities.isWebGL2,
        cameraTextureInitialized: videoTexture.image === video && video.videoWidth > 0,
        cameraCompositing,
        diagnosticStage,
      };
    },
    hasCameraTexturePixels() {
      if (destroyed || video.videoWidth < 1 || video.videoHeight < 1 || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return false;
      }

      const previousTarget = renderer.getRenderTarget();
      const previousClearColor = renderer.getClearColor(new THREE.Color());
      const previousClearAlpha = renderer.getClearAlpha();
      const previousMode = cameraUniforms.uCameraCompositing.value;
      const pixels = new Uint8Array(8 * 8 * 4);
      try {
        // The probe must use the same VideoTexture and shader path as normal
        // compositing. A successful 2D drawImage is not enough on iOS.
        cameraUniforms.uCameraCompositing.value = 1;
        renderer.setRenderTarget(cameraProbeTarget);
        renderer.setClearColor(0x000000, 1);
        renderer.clear();
        renderer.render(envScene, camera);
        renderer.readRenderTargetPixels(cameraProbeTarget, 0, 0, 8, 8, pixels);
        let brightPixels = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          if (Math.max(pixels[index], pixels[index + 1], pixels[index + 2]) > 10) brightPixels += 1;
        }
        return brightPixels >= 3;
      } catch {
        return false;
      } finally {
        cameraUniforms.uCameraCompositing.value = previousMode;
        renderer.setClearColor(previousClearColor, previousClearAlpha);
        renderer.setRenderTarget(previousTarget);
      }
    },
    setTracking(tracking) {
      lastTrackingAt = performance.now();
      target.state = tracking.state;
      target.endpointDepth.set(tracking.leftDepth, tracking.rightDepth);
      trackedPalms = tracking.palms;
      trackedScales = tracking.handScale;
      rigHands = tracking.rig;
      // The field's light is born where the wave is anchored, so the surfaces
      // are shaded against the same position the wave is drawn from.
      fieldAnchor.x = (tracking.left.x + tracking.right.x) * .5;
      fieldAnchor.y = (tracking.left.y + tracking.right.y) * .5;
      fieldAnchor.z = (tracking.leftDepth + tracking.rightDepth) * .5;
      buildPalmSurfaces();
      target.confinement = THREE.MathUtils.clamp(tracking.confinement, 0, 1);
      target.presence = THREE.MathUtils.clamp(tracking.presence, 0, 1);
      target.mode = tracking.mode === 'pinch' ? 1 : 0;
      target.seal = THREE.MathUtils.clamp(tracking.seal, 0, 1);
      target.pulse = THREE.MathUtils.clamp(tracking.pulse, 0, 1);
      target.facing = THREE.MathUtils.clamp(tracking.facing, 0, 1);
      rigFace = tracking.face;
      if (tracking.faceTriangles && faceIndices !== tracking.faceTriangles) {
        faceIndices = tracking.faceTriangles;
        faceGeometry.setIndex(new THREE.BufferAttribute(faceIndices, 1));
      }
      buildFaceSurface();
      updatePersonMask(tracking.person, tracking.bodyDepth);
      updateDepthMap(tracking.depth);
      target.left.set(tracking.left.x / width * 2 - 1, 1 - tracking.left.y / height * 2);
      target.right.set(tracking.right.x / width * 2 - 1, 1 - tracking.right.y / height * 2);
    },
    destroy() {
      destroyed = true;
      rigTarget.dispose();
      frontTarget.dispose();
      rigMesh.geometry.dispose();
      (rigMesh.material as THREE.Material).dispose();
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
      timer.disconnect();
      composer.dispose();
      bloomPass?.dispose();
      gradePass.dispose();
      videoTexture.dispose();
      environmentTarget.dispose();
      cameraProbeTarget.dispose();
      screenGeometry.dispose();
      cameraMaterial.dispose();
      emissiveMaterial.dispose();
      compositeMaterial.dispose();
      blurMaterial.dispose();
      emissiveTarget.dispose();
      visibleTarget.dispose();
      palmTarget.dispose();
      palmGeometry.dispose();
      palmMaterial.dispose();
      palmRigMaterial.dispose();
      faceSurfaceMaterial.dispose();
      faceGeometry.dispose();
      personMaterial.dispose();
      personTexture?.dispose();
      depthTexture?.dispose();
      blurTargets.forEach((t) => t.dispose());
      filamentGeometry.dispose();
      filamentMaterials.forEach((material) => material.dispose());
      particleGeometry.dispose();
      particleMaterial.dispose();
      renderer.dispose();
    },
  };
}
