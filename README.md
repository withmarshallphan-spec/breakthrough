# Too Expensive to Collapse

An upgrade of the existing Vite/vinext + React + Three.js webcam experience.
MediaPipe hand and optional face tracking still run locally in the browser.

## Run

```sh
npm install
npm run dev
```

Use the localhost URL printed by the server, or HTTPS, for camera access.
Models and WASM load from MediaPipe/CDN endpoints; video frames are processed
locally and are not uploaded by this application.

Enable the camera. Hold two palms apart, then bring them together. Raise, lower,
tilt or cross the palms to move the field. One-hand thumb/index pinch remains a
fallback. A small state label follows the hand: broad state, confining, compressed,
or releasing, with a qualitative energy word and bar. There are no numeric
readouts, titles, slides or science panels. “Just a simulator.” remains visible.
**F** or the eye button toggles labels; **Shift+D** shows the tracking rig;
**D** hides it. The underlying ideal-box model remains illustrative.

## What changed

- **Palm emission follows the palm.** `buildPalmFrame` turns the landmarks into
  a closed Catmull-Rom silhouette: the wrist, a thenar bulge over the thumb
  muscle, the knuckles carried a short way toward their own PIP joints so the
  surface reaches into the roots of the fingers, and a hypothenar bulge down
  the little-finger edge. Every control point is a linear combination of
  landmark positions, so the shape rotates, tilts and foreshortens with the
  hand. Three concentric rings of that contour are triangulated into an
  emission surface carrying angle-around-the-palm and distance-from-centre
  coordinates, which the palm shader uses for soft, slowly evolving turbulence. Emission is anisotropic: the side of the palm the field
  leaves from burns brighter. A palm turned edge-on or curled into a fist
  collapses the area its own outline encloses, and the surface dims with it.
  There are no circular palm sources, fingertip sprites or luminous graph walls.
- **Hands block the field; faces receive light.** Hand/palm geometry is rendered
  into a dedicated occlusion target. The optional face and person proxies stay
  in a separate surface target for relighting, so erroneous monocular face or
  body depth cannot erase the wave. Segmentation remains optional and retains
  its existing adaptive sampling and fallback behavior.
- **Soft scarlet light.** Low-frequency wisps, a warm white core and a crimson
  spill follow the palm surfaces. Surface normals and irradiance are gently
  filtered to soften relighting while preserving the camera's skin texture.
  Compression increases source radiance, scattering and bloom. Small chromatic
  fringes are limited to illuminated regions; anamorphic flare samples visible
  emission, so fully occluded energy does not generate an unrelated flare.
- **The field can be held over the head.** No head-relative clamp is applied.
  Face/person geometry never blocks field visibility, regardless of inferred
  depth. Per-finger depth still decides whether a hand covers a strand.
- The original standing-wave ribbons now project from camera-space 3D points.
  Each endpoint has its own estimated depth, and the strands spread along a
  local perpendicular and binormal. Nine layers include a central filament and
  faint interference sheets. Grains occupy shallow depth and change apparent
  size with camera distance. Every transverse offset tapers toward the palms.
- Fixed the former constant-depth bug: hand distances were consumed before they
  were populated. Depth now comes from the actual filtered palm endpoints.
  Object-fit cover and relative landmark depth are accounted for. No vertical
  screen band or head-relative clamp limits the field.
- The palm polygon itself occludes; tapered chains approximate fingers. The
  optional face proxy shares the approximate depth scale. A fragment on each
  strand/particle is compared against the nearest hand surface, so different
  layers can pass on different sides of a hand.
- Two emission passes separate visibility from illumination. Hidden light can
  still illuminate the edges of fingers; it does not leave a sharp visible
  ribbon through them. Refraction respects occlusion and the finite well.
- Clasping replaces the bridge with compact filament loops and a grain knot,
  shaded palm pressure, edge light, and a small local shimmer. The renderer
  explicitly consumes the gesture state, rather than guessing from a slowly
  smoothed distance. It suppresses the normal bridge immediately on clasp entry.
  Sealed light now also transmits: irradiance trapped behind the hands is added
  back through them, weighted by a grazing term so thin edges pass the most,
  and confined to the hands actually holding it. The edge catch strengthens
  with the seal, which is what reads as light escaping between the fingers.
- Opening a clasp fires a single decaying impulse from the state machine. The
  renderer spends it on a brief outward breath of the whole volume and a bloom
  lift, so release is an event rather than a fade.
- Stable hand association reduces endpoint swaps. There is no second long
  renderer delay on top of landmark filtering. Tracking timeout fades the field
  at its last position; no unanchored resting trace is drawn.

## States

`lib/field-state.ts` implements a separately testable state machine:

| State | Trigger and behavior |
| --- | --- |
| dormant | No hands; fade out at the last anchors. |
| open | Wide or opening palms; broad fixed-mode state. |
| compressing | Closing or holding a confined well; increasing energy cue. |
| clasped | Gap below 1.12 times palm size; stay clasped until gap exceeds 1.55. No normal bridge. |
| release | Reopening after clasp; 750 ms eased return, plus a decaying impulse. |

An approach envelope begins compressing the visual before clasp entry. A merged
single-hand detection near the last clasp holds its anchors for at most 1.1 s;
a complete tracking dropout holds for at most 220 ms. A lone pinch cannot enter
the two-hand clasp state. These thresholds are VFX tuning, not physical units.

## Scientific interpretation

This depicts an **extended quantum state**, not the material shape of an
electron. Electrons are treated as pointlike fundamental particles. The
longitudinal trace is a scaled real part of an illustrative standing-mode
superposition; grain brightness follows its time-averaged longitudinal density.
Transverse depth, light, shimmer and the clasp knot are cinematic additions.

For a fixed mode in an ideal one-dimensional infinite box,
`E_n = n²π²ℏ²/(2mL²)`. Halving L quadruples E without increasing n. The experience
keeps mode populations fixed instead of artificially raising n on compression.
Relative phase rates use n²/L² with a deliberately slowed display clock.

The interface uses qualitative state and energy labels only. Internally, the
width dial remains dimensionless and the fixed-mode energy follows the ideal
box inverse-square relation. There are no measured atomic values. The crimson
palette, glow, slow animation clock and compressed knot are artistic cues.

For an atomic trial state of size L, kinetic energy scales as `+A/L²`, whereas
Coulomb attraction scales as `−B/L` (A, B positive). At sufficiently small L,
kinetic energy wins and a finite energy minimum results. This scaling argument
illustrates atomic stability; the browser does **not** solve a hydrogen
Hamiltonian. Bulk matter additionally requires electron antisymmetry/Pauli
exclusion and the many-body Coulomb interactions. An atomic ground state is a
stationary state with no lower electronic state to decay into, not a radiating
classical orbit. Spatial compression here is not quantum measurement collapse.

Sources and the distinction between the model and the illustration are in the
[research notes](research/realism-science-notes.md).

## Practical limits and tuning

This is monocular VFX compositing, not calibrated AR. Apparent hand/face sizes
and a typical webcam focal length estimate depth. Different anatomy, head yaw,
fast motion and self-occlusion can cause errors. Person segmentation supplies a
silhouette and nothing else — a mask carries no depth — so the whole body is
used to estimate how light reaches the person. Only the tracked hands
occlude field fragments; head/body occlusion is intentionally disabled. Bright backgrounds still limit how convincingly synthetic light
can relight skin.

- `lib/hand-tracker.ts`: One Euro filters, apparent-size assumptions, cover mapping.
- `lib/palm-geometry.ts`: palm control points, contour resolution, bulge amounts.
- `lib/segmentation.ts`: mask width, feather radius, and the segmentation clock.
- `lib/field-state.ts`: clasp thresholds, approach envelope, loss timeouts, impulse decay.
- `lib/wave-engine.ts`: `volumeChunk`, `amplitudeFor`, `BONES`, palm rings and
  ring glows, `PERSON_RELIEF`, occlusion depth margin, `LIGHT_GAIN`, `RIM_GAIN`,
  `SEAL_TRANSMIT`, head half-depths, and local knot size.
- Rig target is capped at 960 px width; light blur runs at quarter resolution.
  Particle count adapts downward under sustained low frame rate. Face tracking
  runs at half hand-tracking rate; segmentation runs slower still. Both are
  optional and both fail soft.

## Verification

```sh
npm test
npx tsc --noEmit
npm run build
```

Regression tests cover clasp hysteresis, loss/reacquisition, pinch independence,
the release impulse, palm rotation/translation, the palm contour being a real
non-circular silhouette, moving endpoint depths, fixed-mode energy scaling, that
the internal energy ratio agrees with the ideal box, and that the uncertainty
product is width-independent and never reaches ℏ/2.

All eleven GLSL programs — including the reworked palm surface and the new
person plane — were compiled and linked against a native OpenGL context before
this was committed. That catches syntax and linkage errors only; it is not a
substitute for WebGL/browser or live-camera validation.

For a live camera check, rotate each palm edge-on; raise both hands above the
head and hold the field in front of your face; move one hand closer; pass
fingers across individual strands; step back so the torso crosses the field;
clasp slowly and quickly; reopen; hide one hand; then use an isolated pinch.
Check that emission takes the shape of the palm rather than a disc, that front
fingers block strands, that the field survives being held over the head, that
faces and torsos receive soft light without cutting off the field, that a clasp has no spanning bridge, and
that tracking loss does not leave a persistent field. **This live-camera
acceptance pass has not been performed in this change.**

## Visual reference

[Scarlet Witch film still](https://www.cultture.com/los-10-poderes-mas-extranos-que-la-bruja-escarlata-ha-tenido-nunca):
bright warm cores, diffuse red light on skin, and wispy margins. Reference only;
no film image is embedded in the simulator.
