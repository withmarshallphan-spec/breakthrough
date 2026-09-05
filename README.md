# Too Expensive to Collapse

A hand-tracked interactive on the quantum stability of matter. The camera feed
is the room; a confined quantum state is rendered into it as a real light
source, and everything it touches — palms, fingers, face, body — is relit from
the same field geometry that draws the wave.

MediaPipe hand, face and segmentation models run locally in the browser.
On capable machines an optional monocular depth model runs in a worker.
No video leaves the device.

## Run

```sh
npm install
npm run dev
```

Use the printed localhost URL, or HTTPS, for camera access. Models and WASM
load from MediaPipe and jsDelivr; frames are processed locally.

Enable the camera, hold two palms apart, and bring them together. One-hand
thumb/index pinch is a fallback.

| Key | |
| --- | --- |
| `F` | Film mode |
| `N` | Open the physics reader |
| `Esc` | Close the reader |
| `Shift+D` / `D` | Show / hide the tracking rig |

## What this release changed

The previous build looked flat because the surfaces it was lighting had almost
no geometry: the head was one ellipsoid plus a sphere for the nose, the body
was a silhouette on a single flat plane, and the background was a constant.
The wave was fine; there was nothing shaped for it to land on.

**Read [VFX_RESEARCH.md](VFX_RESEARCH.md) and [RESEARCH.md](RESEARCH.md) first** —
they are the pre-implementation research this build was written against, and
they carry the reasoning, the sources, and the limits.

### 2.5D geometry

- **The face is a surface, not an ellipsoid.** 468 landmarks are drawn as the
  852-triangle canonical tesselation, with normals from screen-space
  derivatives. A cheekbone now has an orientation of its own, so it catches
  light that the plane beside it does not.
  `FACE_LANDMARKS_TESSELATION` is documented as a connection list but is stored
  as consecutive edge triples; `trianglesFromTesselation` recovers the exact
  852 triangles and drops any group that does not close.
- **Head distance comes from the facial transformation matrix.** MediaPipe fits
  it by Procrustes analysis against the canonical face model, whose metric unit
  is the centimetre. Unlike apparent cheek width, it is not shortened by head
  yaw — turning your head no longer reads as leaning in. Falls back to apparent
  width, then to a fixed distance.
- **Normals are computed in metric space.** This was the single biggest cause of
  the flat look. Landmark z arrives as nearness spanning more than a metre over
  0..1, while x and y span the viewport in pixels; a cross product across those
  units is dominated by the screen-space term, so *every* surface reported as
  facing the camera. `buildPalmFrame` and the surface shader now scale z into
  pixel units first, from the tracked distance. Covered by
  `tests/geometry.test.mjs`.
- **Optional dense depth.** Depth Anything V2 Small via Transformers.js and
  WebGPU, in a worker, at roughly 10 Hz. It is affine-invariant, so each map is
  fitted to the rig's own distances by weighted least squares
  (`fitAffine`, one reweighted pass to survive outliers). It fills torso, arms,
  hair and the room — never hands or face, and never field visibility.

### Palm and finger light

- **Rings are offset inward along the silhouette, not scaled toward a centre.**
  Scaling drives any outline to a disc as it shrinks, so the bright core — the
  part the eye actually reads — was circular no matter how good the outline
  was. `insetContour` walks each contour point along its own inward normal.
- **Brightness is N·L against the real field position**, in the metric space
  above, with a wrap term and inverse-square falloff. A palm turned edge-on now
  dims because it is turned away, as well as because its outline has collapsed.
- **Fingers emit on their own.** One quad per finger bone, each shaded at its
  own midpoint, so a finger reaching toward the field brightens while the one
  behind it does not.

### Depth ordering across the head

The old rule — *the head never occludes the field* — is gone. It was a
reasonable defence when head depth came from apparent width, but it is also
what made the effect read as clamped around the head: with the head absent
from the depth buffer, the compositor could not put the field *behind* it
either.

Now two geometry passes are rendered. The first is every surface at its
measured distance, and it is what the composite lights. The second is only the
geometry trustworthy enough to cut a luminous strand — hands and the face mesh.

**Only the hands are in that second pass.** The face is deliberately absent:
the wave crosses a face rather than passing behind it, so the head lights but
never occludes. Segmentation and the neural depth map are out too — a mask has
no depth, and a neural depth map is smooth across exactly the boundaries
occlusion depends on. The face mesh still writes into the *first* pass, which
is what gives a cheekbone its own orientation under the light.

**Compression runs the whole gesture.** The dial now spans from palms at their
widest — five or six palm-widths apart — down to a clasp, so it starts
responding the instant the hands begin closing rather than waiting for them to
get near each other. It is deliberately linear in the gap: the acceleration the
eye reads comes from `E/E₀ = (L₀/L)²`, which is the physics, not from bending
the curve. The clasp gate opens at 1.45 palm-widths and holds to 1.95, so it is
easier to enter than to leave, and every smoothing constant on the path was
roughly halved.

**Near-collapse is its own state.** Between compressing and clasped there is now
`critical`: palms very close, the well narrow, nothing sealed. It is where the
image is most legible, and lumping it in with compressing left the renderer no
way to know it had arrived. A continuous `uCritical` — taken off the dial rather
than off the state flag, so the treatment ramps in with the gesture — drives:

| | open | compressing | critical |
| --- | --- | --- | --- |
| Amplitude | broad | narrowing | −34% again |
| Filament layers | spread wide | drawing in | condensed 58% toward the centre |
| Grain cloud | airy | tightening | 45% narrower |
| Core / glow width | soft | tightening | −38% / −30% |
| Radiance | low | rising | +1.5 |
| Light thrown into the room | faint | rising | +0.6 gain, +0.75 near-field |
| Refraction, flare, bloom | minimal | rising | all lifted |

Everything it drives narrows or brightens. **Nothing it drives shakes** — kinetic
energy here is momentum *spread*, and agitation would teach the wrong thing.

**Mask edges are feathered with a halftone.** The occluder renders at a
fraction of display resolution, so its boundary arrived as a visible staircase.
A rotated dot screen now dissolves the transition band into a halftone, applied
only to the shoulder — solid fingers stay solid and clear sky stays clear — and
the depth comparison band around it is wider than a cutout. The field itself and
the composite that shades its edges share the same helper, so both agree about
where a silhouette ends.

### Clasped hands

Detection adds three signals to the gap ratio, all from landmarks already
present: palm silhouettes collapsing, both palms at a comparable distance, and
fingertips of one hand inside the other's contour polygon. They widen or narrow
the gap threshold rather than replacing it (`GestureSample.evidence`), so the
gesture reads the same when no extra evidence is available.

The open bridge is off. What remains is an interior that tightens as the hands
close, seen only through them: transmission weighted by a grazing term, an edge
catch where coverage falls away between the fingers, and the compressed
filament visible through those gaps and nowhere else.

### The field as the light source

One emissive buffer, blurred into irradiance, consumed by one composite pass —
now lighting the face mesh and the depth-derived body as well as the hands.
Hue and level are separated: the emission colour is normalised to unit
luminance before a level is applied, so the ramp changes the colour without
changing the calibration every gain downstream depends on.

**A near-field source, not a screen-space bloom**, and its reach is deliberately
larger than the scene is deep, so the whole subject and the wall behind them are
lit rather than only the hands holding it. The depth falloff is generous for the
same reason — a face is most of a metre behind the hands, and a tight falloff
leaves it unlit however bright the source is. The face mesh is also grown 22%
about its own centre before it is lit: the landmarks stop at the edge of the
face, so the mesh alone left the hairline, jaw and neck dark, a lit oval on an
unlit head. Since that surface only lights and never occludes, growing it costs
nothing. The blurred buffer only knows
where light is *on screen*, so a face beside the field received almost nothing
from it however bright it was. The composite now also treats the field as what
it is — a small bright thing at a known place in the room — and lights every
surface that has a depth by distance and orientation. A face is lit the way a
struck match lights a room, not only the part of the wall it happens to overlap.

**The wave is coloured by phase, across the whole hue circle.** `|psi|²` is real
but it is not the whole state; the phase is what makes a wavefunction a wave
rather than a cloud, and `arg(psi)` is the hue while density stays in the
brightness. This is the standard domain-colouring convention — see
[RESEARCH.md §5](RESEARCH.md#part-i--the-physics).

An earlier version mixed between two poles, which is a *chord* across the wheel
rather than the wheel itself: it passed through near-white halfway round, so
half of every cycle carried almost no colour, and two phases a full π apart in
opposite directions came out identical. Three cosines at 120° give the complete
circle, so the map is one-to-one. The wheel is then normalised to fixed
luminance, which is the load-bearing step — a raw hue circle swings by roughly a
factor of two in brightness between yellow and blue, and brightness is already
carrying `|psi|²`, so an unnormalised wheel would be silently restating the
density and getting it wrong. `tests/geometry.test.mjs` checks all three
properties.

Getting that colour to actually survive to the screen took four separate
things, and it is worth listing them because any one of them alone loses it:
saturated poles; far less white mixed in by density on the way out
(`.06 + .3·|psi|²` where it was `.3 + .45·|psi|²`); **luminance-based** highlight
roll-off in the grade rather than per-channel, which was pulling every bright
coloured pixel toward white; and hue held separate from level, so the state ramp
cannot dim the colour by accident. The poles and their properties live in
`lib/field-palette.ts` and are checked by test rather than tuned in place.

The room light stays near neutral: a saturated spill on skin reads as a filter
over the shot, and nothing tints a whole hand.

There is no CSS glow anywhere in the pipeline.

### Interface

**There are exactly three pieces of text**, and the rest of the frame is the
shot:

1. **A callout that follows one hand.** The tracker draws a dot on the tracked
   palm and a hairline leader that curves out of it and arrives horizontally at
   the text — one owner for both, so the line always meets the block. It carries
   the state and the four levels.

   It locks to a single palm on acquisition and holds that slot for as long as
   it exists, so two visible hands never make it hop. **It never disappears**:
   losing tracking leaves the block exactly where it was and fades only the
   leader, because there is no longer a hand to point at. Nothing about it
   snaps — the position is smoothed with a long time constant (0.21 s,
   deliberately far slower than the tracking, since text pinned rigidly to a
   moving hand reads as panic however clean the signal is), the side it sits on
   has a 110 px hysteresis band before it will change its mind, and the overlay
   repaints every animation frame rather than only on frames that carried new
   landmarks — the models deliver at about half the display rate, and a mark
   drawn only when they do steps visibly.
2. **One standing line**, bottom centre: the gesture prompt before hands are
   found, then the model caveat.
3. **The physics reader**: eight short sections, each with its source.

Both floating blocks sit on a scrim that reaches transparent before its own
edges (`closest-side`), so text stays legible over a hand lit white by the field
without ever becoming a panel with a boundary. There are no vertical rules
anywhere; where a relationship has to be shown, it is drawn as a leader from a
dot.

- **Typography.** Alte Haas Grotesk **Bold** for display, **Instrument Serif**
  for reading copy *and for the symbols and figures* — they are body text too —
  Geist Sans for the small caption tier. The monospace was dropped: with the
  readouts drawn as meters there is no column of changing digits left to align.
  Subscripts are real `<sub>` markup, because Instrument Serif has no U+2080 and
  a literal `₀` falls back to another face looking like the letter o.
  Nothing wears a shadow. **The interface is set lowercase**, with
  three exemptions where case is meaning rather than style — see
  [RESEARCH.md, Part IV](RESEARCH.md#part-iv--typography), which also covers
  licences, loading, and what happened to the originally requested display
  face.
- **Film mode** is four hairline Lucide icons sitting directly on the viewport
  at 16 px and stroke width 1.25, no container. Everything else fades out,
  including the tracker's dot and leader.

### Quality tiers

Chosen from what the device advertises, then demoted by what it delivers. The
renderer measures the frame rate, the controller decides, and the tier change
reaches the tracker through a subscription — the depth model and the segmenter
shut down without the pipeline being rebuilt.

| | HIGH | MEDIUM | FALLBACK |
| --- | --- | --- | --- |
| Hand rig, palm surface | yes | yes | yes |
| Finger emission | yes | yes | no |
| Face mesh | yes | yes | no |
| Segmentation | multiclass | binary | off |
| Dense depth | ~10 Hz | off | off |
| Irradiance blur pairs | 6 | 4 | 3 |
| Particles | 8000 | 8000 | 3600 |

## Scientific claims

Audited in full in [RESEARCH.md, Part II](RESEARCH.md#part-ii--audit-of-every-number).
Summary: the audit found no fake precision to remove — a previous pass had
already deleted the nm and eV readouts — but found that the quantities that
*are* honestly computed were not shown at all.

Four dimensionless quantities are shown, and only these four, because only
these are exact for the ideal one-dimensional infinite square well the renderer
is actually running:

| Reading | | Range |
| --- | --- | --- |
| Well width | `L/L₀ = 2^(−c)` | 1 → 0.500 |
| Kinetic energy | `E/E₀ = (L₀/L)²` | 1 → 4.00 |
| Momentum spread | `Δp/Δp₀ = L₀/L` | 1 → 2.00 |
| Uncertainty product | `Δx·Δp` at n=1 | 0.568 ℏ, constant |

**As levels, not digits.** The relations above are exact — the tests check them
to machine precision — but the dial driving them is a smoothed estimate of the
distance between two hands, through an assumed focal length. Printing `0.946`
would claim three significant figures for an input that has nothing like that
precision. A meter states what the gesture actually supports: where the value
sits on its own range, and which way it is going.

The fourth is the point, and it is drawn as a marker that never moves: `Δx·Δp`
is independent of L, because `Δx = L√(1/12 − 1/2π²n²)` and `Δp = nπℏ/L`, so the
width cancels. Squeezing does not push the state toward the ℏ/2 bound. What it
buys is momentum spread, and that is what costs energy. A stationary *number*
invites you to wait for it to change; a marker sitting above a marked bound
shows you that it will not.

The interface labels the visualisation as a model and analogy. It does not solve
an atomic Hamiltonian, model radiation, measure an atomic length, or depict
measurement collapse — closing your hands is spatial confinement.

## Limits

- Monocular. Every distance is inferred. Unusual anatomy, extreme yaw, motion
  blur and self-occlusion degrade it.
- The virtual camera is assumed, not calibrated. Distances are consistent, not
  correct.
- Depth Anything is affine-invariant; the alignment is only as good as the rig
  samples it is fitted against, and with no hands and no face in frame there is
  nothing to fit to.
- The face mesh covers the front of the head only. Hair and the top of the
  skull are in the silhouette, which does not occlude.
- Relighting is physically motivated compositing, not radiometry. Nothing is
  calibrated in physical units and nothing claims to be.
- A brightly lit room limits how much a synthetic source can plausibly add.

## Where to tune

- `lib/quality.ts` — tier detection, the demotion ladder, per-tier features.
- `lib/hand-tracker.ts` — One Euro filters, apparent-size assumptions, clasp
  evidence weights, cover mapping.
- `lib/palm-geometry.ts` — palm control points, contour resolution, bulge
  amounts, inward-offset limits.
- `lib/face-geometry.ts` — landmark anchors, matrix plausibility bounds.
- `lib/depth-field.ts` — inference size, duty cycle, fit rejection thresholds.
- `lib/segmentation.ts` — model choice, mask width, feather, sampling clock.
- `lib/field-state.ts` — clasp thresholds, approach envelope, loss timeouts.
- `lib/guide-line.ts` — follower time constant, leader curvature, the tail.
- `lib/wave-engine.ts` — `FACE_OCCLUSION_BIAS`, `HALFTONE_DEPTH`, `PALM_RINGS`
  /`RING_GLOW`, `NEAR_FIELD_RADIUS`/`MATCH_GAIN`, `volumeChunk`,
  `amplitudeFor`, `BONES`, `DEPTH_RELIEF`, `LIGHT_GAIN`, `RIM_GAIN`,
  `SEAL_TRANSMIT`, the phase poles, and the emission ramp in
  `updateAllUniforms`.

## Verification

```sh
npm test          # 23 tests
npx tsc --noEmit
npx oxlint
npm run build
```

Tests cover clasp hysteresis and the new evidence term, loss and reacquisition,
the release impulse, palm rotation and translation, the contour being a real
non-circular silhouette, that inset rings keep the palm's shape where scaling
provably cannot, that the palm normal responds to tilt only once z has real
units, fingertip containment, openness collapse, triangle recovery from the
tesselation, matrix head distance and its rejection of implausible fits, the
face frame's fallback chain, affine depth alignment including outlier
rejection, and that `Δp/Δp₀` squared is exactly `E/E₀`.

Two real defects were found by these tests during development and fixed: the
inward contour offset was inverted, so the emission rings grew *outside* the
palm rather than inside it, and the palm normal was degenerate without metric
z scaling.

The build was then driven in Chrome against a synthetic camera feed
(`--use-fake-device-for-media-stream`), with `linkProgram` instrumented from
the page. **All 199 GL programs linked with no failures**, `gl.getError()`
returned 0, the context was not lost, and all three MediaPipe graphs plus the
WebGPU depth worker started. The engine warms every program at startup —
including the face and person shaders, which are otherwise first compiled the
moment someone walks into frame.

That instrumentation earned its place: it caught a composite shader that used
the halftone helper without declaring it, which had silently broken the whole
pipeline. It also caught a hydration mismatch on the tier-dependent copy, and a
font-variable scoping bug that had silently dropped Geist for the system sans.
The interface was screenshot and reviewed in all three states, which is how the
scrim's hard rectangular edge and the illegibility of the meter symbols over a
bright frame were found and fixed.

### Not yet done

**A live-camera acceptance pass has not been performed in this change.** A
synthetic feed contains no hands and no face, so the tracking-dependent
behaviour below is verified structurally and by unit test, not visually:

Rotate each palm edge-on and check the emission narrows and dims. Raise both
hands and hold the field in front of your face, then move it behind your head,
and check the ordering flips through a cross-fade rather than a pop. Pass
fingers across individual strands. Step back so the torso crosses the field.
Clasp slowly and quickly, then reopen. Hide one hand. Use an isolated pinch.
Check that emission takes the shape of the palm rather than a disc, that
individual fingers catch light independently, that a clasp shows an interior
rather than a bridge, and that tracking loss leaves no persistent field.
