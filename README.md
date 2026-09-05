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
fallback. **F** toggles film/teaching mode; **I** opens the science panel;
**Left/Right** change explanation cards. **Shift+D** shows the tracking rig;
**D** hides it. Teaching mode is the default; film mode keeps the clean shot.

## What changed

- Palm emission comes from a six-sided landmark boundary: wrist, thumb base,
  index/middle/ring/little MCPs. Three soft polygon rings form a triangulated
  surface, updated from the same filtered landmarks as the field endpoints.
  There are no circular palm sources, fingertip sprites or luminous graph walls.
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
  strand/particle is compared against the nearest proxy surface, so different
  layers can pass on different sides of a hand.
- Two emission passes separate visibility from illumination. Hidden light can
  still illuminate the edges of fingers; it does not leave a sharp visible
  ribbon through them. Refraction respects occlusion and the finite well.
- Clasping replaces the bridge with compact filament loops and a grain knot,
  shaded palm pressure, edge light, and a small local shimmer. The renderer
  explicitly consumes the gesture state, rather than guessing from a slowly
  smoothed distance. It suppresses the normal bridge immediately on clasp entry.
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
| release | Reopening after clasp; 750 ms eased return. |

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
now keeps mode populations fixed instead of artificially raising n on compression.
Relative phase rates use n²/L² with a deliberately slowed display clock. The
width dial is dimensionless; no hand measurement is converted to atomic units.
Color is explicitly false color, not a predicted emitted photon wavelength.
Physical-unit formula helpers in `lib/quantum.ts` are reference calculations only.

For an atomic trial state of size L, kinetic energy scales as `+A/L²`, whereas
Coulomb attraction scales as `−B/L` (A, B positive). At sufficiently small L,
kinetic energy wins and a finite energy minimum results. This scaling argument
illustrates atomic stability; the browser does **not** solve a hydrogen
Hamiltonian. Bulk matter additionally requires electron antisymmetry/Pauli
exclusion and the many-body Coulomb interactions. An atomic ground state is a
stationary state with no lower electronic state to decay into, not a radiating
classical orbit. Spatial compression here is not quantum measurement collapse.

Sources and the distinction between the model and the illustration are in the
in-app science panel and [research notes](research/realism-science-notes.md).

## Practical limits and tuning

This is monocular VFX compositing, not calibrated AR. Apparent hand/face sizes
and a typical webcam focal length estimate depth. Different anatomy, head yaw,
fast motion and self-occlusion can cause errors. The face proxy excludes hair;
there is no dense person segmentation, body reconstruction or room depth map.
Bright backgrounds also limit how convincingly synthetic light can relight skin.

- `lib/hand-tracker.ts`: One Euro filters, apparent-size assumptions, cover mapping.
- `lib/field-state.ts`: clasp thresholds, approach envelope and loss timeouts.
- `lib/wave-engine.ts`: `volumeChunk`, `amplitudeFor`, `BONES`, palm rings,
  occlusion depth margin, `LIGHT_GAIN`, `RIM_GAIN`, and local knot size.
- Rig target is capped at 960 px width; light blur runs at quarter resolution.
  Particle count adapts downward under sustained low frame rate. Face tracking
  runs at half hand-tracking rate and remains optional.

## Verification

```sh
npm test
npx tsc --noEmit
npm run build
```

Regression tests cover clasp hysteresis, loss/reacquisition, pinch independence,
palm rotation/translation, moving endpoint depths and fixed-mode energy scaling.
The GLSL programs were also compiled and linked with a native OpenGL harness;
this is not a substitute for WebGL/browser or live-camera validation.

For a live camera check, rotate each palm edge-on; raise both hands above the
head; move one hand closer; pass fingers across individual strands; clasp slowly
and quickly; reopen; hide one hand; then use an isolated pinch. Check that
emission follows skin, front fingers block strands, a clasp has no spanning
bridge, and tracking loss does not leave a persistent field. This live-camera
acceptance pass has not been performed in this change.
