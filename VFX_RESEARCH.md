# VFX_RESEARCH — real-time 2.5D relighting in the browser

Written before implementation, 5 September 2026. Every claim about an API was
checked against the version actually installed (`@mediapipe/tasks-vision`
1.0.1, `three` 0.185.1); every performance number is labelled as either
measured here, quoted from a source, or an estimate.

---

## 1. The problem with the current build

The pipeline already has more geometry than a CSS glow: a hand rig of tapered
sphere chains, a landmark-built palm silhouette mesh, an irradiance buffer, and
a depth-compared visibility test. What still reads flat is not the wave. It is
everything the wave is supposed to be lighting:

| Surface | What it currently is | Why it reads flat |
| --- | --- | --- |
| Face | One oriented ellipsoid plus one sphere for the nose | An ellipsoid has no cheekbone, no brow, no nasal ridge. Every facial highlight is therefore a smooth blob that slides across the face instead of catching on structure. |
| Head depth | Apparent cheek-to-cheek width against an assumed 145 mm | Head yaw shortens the apparent width, so turning your head makes the model think you leaned in. |
| Body | A silhouette mask composited on **one flat plane** at head distance | An arm reaching forward is at the same depth as the shoulder behind it. |
| Background | Assumed constant `BACKGROUND_DEPTH = 0.08` | The room cannot receive distance-correct light. |
| Head occlusion | Explicitly disabled (`handTarget` excludes face/person) | Correct as a defence against bad face depth, but it means depth ordering around the head is not computed at all — it is legislated. |

So the fix is not more light. It is more *geometry to put the light on*, and a
depth buffer trustworthy enough that the head can stop being a special case.

---

## 2. Chosen approach

**A layered 2.5D G-buffer, assembled from the cheapest sufficient source at
each pixel, and consumed by one deferred lighting pass.**

The renderer already has the right shape: a `rigTarget` holding
`(depth, normal.xy, coverage)`. The change is what writes into it, and how much
of the frame it covers.

Priority order, highest authority first — later writers never overwrite an
earlier one, because the depth test does the arbitration:

1. **Hand rig** (existing, kept). 21 landmarks/hand → 15 tapered bone chains +
   the palm silhouette mesh. Per-finger depth. This is the only source that
   knows one finger is in front of another.
2. **Face mesh** (new). 468 landmarks → the **852-triangle** canonical
   tesselation, rendered as real geometry. Normals from screen-space
   derivatives of the interpolated surface position, exactly as the palm mesh
   already does. This is what makes a cheekbone a cheekbone.
3. **Dense monocular depth** (new, optional). Depth Anything V2 Small, affine-
   aligned to the rig. Fills torso, arms, hair, background.
4. **Person segmentation** (existing, upgraded). Multiclass, so face-skin,
   body-skin, hair and clothes can be lit differently. Contributes coverage and
   class, never depth.

Then one composite pass does distance falloff, N·L, rim wrap and transmission
against that buffer.

### Why this and not the alternatives

**Why not full 3D reconstruction?** Nothing in the brief needs a mesh you could
orbit. The camera never moves. A single-viewpoint depth+normal buffer is
sufficient for occlusion and relighting and costs a fraction as much.

**Why not stay screen-space?** Because "screen-space brightness" is precisely
the failure mode described: a surface facing away from the field gets lit the
same as one facing it. Orientation-aware lighting needs normals, and normals
need geometry or depth gradients.

**Why the face mesh rather than the transformation matrix alone?** The matrix
gives head *pose* — one rigid frame. The brief asks for the cheekbone to catch
more light than the plane beside it, which is a per-triangle question the pose
matrix cannot answer. So: use both. Matrix for metric distance and a clean
basis; mesh triangles for surface orientation.

---

## 3. Face geometry

### 3.1 The tesselation is already triangles — verified

`FaceLandmarker.FACE_LANDMARKS_TESSELATION` is documented as a list of
*connections* (edges). Reading the shipped bundle shows it is in fact stored as
**consecutive edge triples**, each triple being one triangle. Extracted from
`node_modules/@mediapipe/tasks-vision/vision_bundle.cjs` and checked:

```
edges in the array                 2556
grouped as [a,b],[b,c],[c,a]        852 triangles
groups that failed that pattern       0
distinct vertices covered           468  (indices 0..467)
```

Independently recovering triangles as 3-cliques of the edge set yields 854 —
two spurious. So consecutive grouping is the exact decomposition, and it is
derived at load from the library constant rather than from a hard-coded array
copied out of a blog post.

Cost: 468 vertices × 3 floats uploaded per tracked frame = 5.6 KB, and 852
triangles drawn. Both negligible.

### 3.2 Normals

Per-triangle geometric normals, computed in the fragment shader from
`normalize(cross(dFdx(P), dFdy(P)))` where `P` is the interpolated
camera-space position. This is what the existing palm mesh does, it needs no
CPU normal pass, and it gives genuinely faceted structure — the nasal ridge and
the cheekbone edge produce a real normal discontinuity, which is exactly the
"nose edge catches light differently" behaviour asked for.

Faceting at 852 triangles is slightly visible on a very close face. A one-ring
smoothing of the normal buffer (already present as `softNormal` in the
composite) removes it without removing the ridges.

### 3.3 Metric head distance from the transformation matrix

Enabling `outputFacialTransformationMatrixes: true` returns a 4×4 matrix
`P = [R | t]` mapping the **canonical face model** into camera space. MediaPipe
documents the canonical model's metric unit as the **centimetre**, and derives
`P` by Procrustes analysis against the runtime metric landmarks under a virtual
perspective camera at the origin looking down −Z.

That gives `|t|` in centimetres directly. This replaces apparent-cheek-width
estimation, and fixes its worst failure: **head yaw no longer reads as leaning
in**, because the Procrustes fit accounts for rotation while apparent width
does not.

Fallback chain: matrix → apparent width → a fixed mid-range distance. Each step
degrades quality, none breaks the app.

Caveat worth stating: MediaPipe's virtual camera is assumed, not calibrated to
the user's actual webcam, so `|t|` is accurate up to the focal-length mismatch.
It is far better than apparent width and still not metrology.

---

## 4. Depth estimation: Depth Anything V2

### 4.1 Decision

**Yes, on the HIGH tier only, off the render thread, at 8–12 fps, and only as a
filler for surfaces the rig cannot reach.** Never as the authority for hands or
face, and never in the critical path — if it never loads, the experience is the
MEDIUM tier and nothing else changes.

### 4.2 Model and runtime

- `onnx-community/depth-anything-v2-small` via Transformers.js (ONNX Runtime
  Web) with `device: 'webgpu'`, `dtype: 'fp16'`. ViT-S encoder + DPT decoder,
  ~25 M parameters, **≈50 MB at fp16** per the model author's own announcement.
- Transformers.js is loaded by dynamic `import()` of a pinned CDN URL inside a
  module Web Worker, so it is not in the app bundle and costs nothing on the
  tiers that do not use it.
- Frames reach the worker as `ImageBitmap` transfers from
  `createImageBitmap(video)` — no pixel copy on the main thread.

### 4.3 The alignment problem, and why it is solvable here

Depth Anything V2 is trained with a **scale- and shift-invariant** objective in
disparity space. Its output is affine-invariant relative inverse depth: correct
*ordering*, arbitrary units. Standard practice is to recover metric depth by a
least-squares fit of a global scale and shift against known references.

This project has references the general case does not: the hand rig and the
face already carry distances on a real scale. So each time a depth map arrives:

```
sample the rig's nearness  r_i  at pixels where rig coverage is solid
sample the model disparity d_i  at the same pixels
solve  min over (a, b) of  sum_i ( a·d_i + b − r_i )²
```

A closed-form 2×2 least squares over a few thousand samples, well under a
millisecond. The aligned map is then only *used* where the rig has nothing —
torso, arms, hair, background — so a bad fit degrades the room, not the hands.

Guards: reject the fit if `a` is non-positive or the residual is large, and
keep the previous transform. Blend consecutive maps temporally so an 8 fps
signal does not pop at 60 fps.

### 4.4 Honest cost

I have not measured this model on this machine — it is behind a WebGPU probe
and a runtime benchmark rather than an assumption. What is known:

- The model author's public claim is real-time browser depth with WebGPU at
  ~50 MB fp16.
- WebGPU vs WASM for Transformers.js is widely reported as a large multiple,
  not a few percent.
- The one write-up specifically about Depth Anything V2 Small + WebGPU in
  production declines to give a number, saying plainly: *"There is no honest
  device-independent number."* That is the correct posture and this document
  adopts it.

So the implementation **measures instead of assuming**: it times the first few
inferences and demotes itself out of the HIGH tier if the median exceeds its
budget. Estimated, to be confirmed on device: 25–60 ms per 252 px inference on
Apple Silicon, which is 16–40 fps of headroom for a 10 fps duty cycle on a
separate queue.

### 4.5 What it is *not* used for

Not for cutting the luminous field. A neural depth map is smooth across
occlusion boundaries and temporally unstable at the pixel level; using it to
decide whether a strand of light is visible would make the field flicker along
every silhouette. Field visibility stays with the rig and the face mesh, which
are geometric and temporally filtered.

---

## 5. Depth ordering across the head

The current rule — *the head never occludes the field* — goes. It was a
reasonable defence when head depth came from apparent width, which yaw
corrupts. With a Procrustes distance and a real face mesh it is no longer
needed, and it is the direct cause of the "artificially blocked or clamped
around the head" impression: because the head never participated in the depth
buffer, the field could never be *behind* it either, so the compositor had
nothing sensible to do near the head and the result read as a clamp.

Replacement:

- The face mesh writes into the same front-occluder target the hands do.
- Field visibility is the existing soft depth comparison, unchanged in form:
  `1 − coverage · smoothstep(fieldDepth − m, fieldDepth + M, surfaceDepth)`.
- The margin is **wider for the face than for the hands** (asymmetric soft
  threshold), because face depth is less certain than finger depth. Practical
  effect: a field clearly in front of the face renders in front; a field
  clearly behind is occluded; in the uncertain band it cross-fades rather than
  popping.
- If the face mesh is unavailable, the head simply is not in the buffer and the
  field passes freely — the old behaviour, as a fallback rather than a rule.

---

## 6. Palm-shaped light, and why the current one still reads round

`buildPalmFrame` already produces a real silhouette: wrist, thenar bulge,
knuckles carried toward their PIPs, hypothenar bulge, closed Catmull-Rom. That
part is sound and stays.

Three things make it read like a sprite anyway:

1. **The rings are concentric scalings of the contour about a single centre.**
   `emit()` places each ring vertex at `center + (edge − center) · amount`. A
   star-shaped scaling of any outline tends to a disc as `amount → 0`, so the
   bright core — which is what the eye reads — *is* a disc. Fix: offset the
   contour inward along its own local normal (a polygon erosion) instead of
   scaling toward a point, so the bright core keeps the palm's shape.
2. **`aField` is a screen-space dot product** between the outward direction and
   the direction to the other palm. It has no notion of the palm's own normal,
   so a palm rotated edge-on still lights its rim the same way. Fix: shade the
   surface with the real 3D palm normal against the real 3D field position —
   `N·L` — so a palm turned away dims because it is turned away.
3. **Fingers are not emissive at all.** Only the palm polygon is. Fix: add
   emissive quads along the finger bones, each with its own segment normal, so
   individual fingers catch highlights independently.

Edge-on narrowing then comes for free twice over: the silhouette itself
collapses (already measured by `palmOpenness`), and `N·L` falls.

---

## 7. Clasped hands

Detection today is a gap-to-palm-size ratio with hysteresis (1.12 in, 1.55
out). That is robust but it cannot tell a clasp from two hands overlapping in
projection while far apart in depth.

Added evidence, all cheap and all from landmarks already present:

- **Palm openness collapse** — both silhouettes lose enclosed area as they turn
  toward each other and interlock.
- **Depth agreement** — the two palm centres must be at comparable distance;
  overlap in x/y with a large z gap is one hand in front of the other, not a
  clasp.
- **Fingertip containment** — fingertips of one hand falling inside the other's
  contour polygon.

State appearance, replacing the "knot" that is really a small sphere of
particles between two palm centres:

- Open bridge **off** (already gated by `uSeal`).
- Interior glow rendered *behind* both hand silhouettes, so it is only seen
  through them.
- Transmission through the hands, weighted by a grazing term so thin tissue
  passes most — already implemented, kept and strengthened.
- Leak through gaps: the front-occluder's own edge gradient marks the finger
  boundaries; light escapes where coverage drops, which is geometrically where
  the gaps are.
- A short compressed filament, drawn only where it is *not* covered — visible
  through the gaps, nowhere else.

---

## 8. The field as the only light source

Already true in structure: one emissive buffer, blurred into an irradiance
buffer, consumed by the composite. The additions are that the same buffer now
lights the face mesh and the depth-derived body, and that the palette follows
state rather than being a fixed scarlet:

| State | Emission | Character |
| --- | --- | --- |
| OPEN | very pale ice-blue, near white | broad, weak, large radius |
| COMPRESSING | silver → white | tighter, higher gain, more refraction |
| HIGH CONFINEMENT | neutral white core, faint warm gold in the falloff | strong but skin-plausible |
| CLASPED | white interior, warm at the leaks | seen through hands, not over them |

Deliberately avoided: tinting whole hands. The tint lives in the *falloff*, and
the core stays neutral, which is how real bright sources land on skin.

---

## 9. Quality tiers

Chosen automatically, then adjusted by measurement.

| | HIGH | MEDIUM | FALLBACK |
| --- | --- | --- | --- |
| Hand rig | full | full | full |
| Palm + finger emissive surfaces | yes | yes | palm only |
| Face | 852-tri mesh + matrix distance | 852-tri mesh | none |
| Segmentation | multiclass 256 | binary selfie | off |
| Dense depth | Depth Anything V2 S @ ~10 fps | off | off |
| Irradiance blur | 6 ping-pong pairs | 4 | 3 |
| Particles | 8000 | 8000 | 3600 |
| Refraction / dispersion | yes | yes | reduced |

Entry test: WebGPU present, `hardwareConcurrency ≥ 8`, `deviceMemory ≥ 8` (when
exposed), not a coarse pointer. Then the existing FPS watchdog demotes: first
particles, then depth, then segmentation, then the face mesh.

No tier draws a circular CSS glow. There is no CSS glow anywhere in the
pipeline; all light is rendered.

---

## 10. Frame budget

Per-signal rates, decoupled — this is already how the tracker is built and it
stays that way:

| Stage | Rate | Where |
| --- | --- | --- |
| Render + composite | display rate (60 Hz target) | main thread, WebGL |
| Hand landmarks | ~30 Hz (per video frame) | main thread, MediaPipe GPU |
| Face landmarks + matrix | ~15 Hz (every 2nd tracked frame) | main thread, MediaPipe GPU |
| Segmentation | adaptive 15–30 Hz, backs off on cost | main thread, MediaPipe GPU |
| Dense depth | 8–12 Hz, HIGH tier only | Web Worker, WebGPU |
| Depth affine fit | with each depth map | main thread, <1 ms |
| Temporal blends | every frame | shader |

Estimated GPU cost added on the HIGH tier, Apple Silicon, 1440×900:

- Face mesh into two targets: 852 tris × 2 ≈ **negligible**, well under 0.1 ms.
- Finger emissive quads: 2 hands × 15 bones × 2 tris ≈ negligible.
- Palm erosion: CPU, ~250 contour points × 2 hands per tracked frame.
- Depth texture upload (252²×1 byte, ~63 KB) at 10 Hz: negligible.
- Depth inference: contends for the same GPU. This is the only real cost and
  the reason it is gated and measured.

---

## 11. Limitations, stated plainly

- Monocular. Every distance is inferred. Unusual anatomy, extreme yaw, motion
  blur and self-occlusion all degrade it.
- The virtual camera is assumed, not calibrated. Distances are consistent, not
  correct.
- Depth Anything is affine-invariant; alignment is only as good as the rig
  samples it is fitted against. With no hands and no face in frame there is
  nothing to fit to, and dense depth is held at its last valid transform.
- Neural depth is smooth across silhouettes. It fills, it does not cut.
- Segmentation classes are approximate — hair/clothes boundaries are soft.
- Relighting is compositing, not radiometry. Nothing is calibrated in physical
  units and nothing claims to be.
- A brightly lit room limits how much a synthetic source can plausibly add;
  the effect is strongest in dim ambient light.

---

## 12. Sources

- [MediaPipe 3D Face Transform](https://developers.googleblog.com/mediapipe-3d-face-transform/) — canonical face model, centimetre metric unit, Procrustes pose estimation, virtual perspective camera.
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) — `outputFacialTransformationMatrixes`, tesselation constant.
- [MediaPipe Image Segmenter — multiclass selfie model](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) — background / hair / body-skin / face-skin / clothes / other.
- [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/html/2406.09414v2) — affine-invariant disparity training objective.
- [Depth Anything V2 Small ONNX](https://huggingface.co/onnx-community/depth-anything-v2-small) — model id and Transformers.js pipeline usage.
- [Transformers.js dtypes guide](https://huggingface.co/docs/transformers.js/en/guides/dtypes) — `device: 'webgpu'`, fp16/fp32 selection.
- [Transformers.js WebGPU video depth estimation example](https://github.com/huggingface/transformers.js/tree/v3/examples/webgpu-video-depth-estimation) — worker + per-frame pattern.
- [Xenova, Depth Anything V2 in the browser](https://x.com/xenovacom/status/1801672335830798654) — ~50 MB fp16, real-time WebGPU claim.
- [Building cinematic depth in the browser with Depth Anything V2 Small and WebGPU](https://dev.to/martindelophy/building-cinematic-depth-in-the-browser-with-depth-anything-v2-small-and-webgpu-1k07) — "There is no honest device-independent number."
- [Monocular depth estimation guide: metric vs relative](https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide) — least-squares scale/shift recovery.
- [Lucide](https://lucide.dev/license) — ISC licence, adjustable stroke width.
- [Alte Haas Grotesk, Yann Le Coroller](https://www.dafont.com/alte-haas-grotesk.font) — freeware, redistributable with its licence text.
- [Geist](https://github.com/vercel/geist-font/blob/main/LICENSE.txt) — SIL Open Font License 1.1.


---

## 13. Postscript: what changed during implementation

This document was written before the code. Three things came out differently,
and the differences are worth recording.

**The flat look had a specific cause, and it was not the face.** Landmark z
arrives as nearness spanning more than a metre over 0..1, while x and y span the
viewport in pixels. Any cross product across those units is dominated by the
screen-space term, so `buildPalmFrame` had been returning a normal of
essentially (0, 0, 1) for every hand in every pose — and the surface shader's
derivative normals had the same defect. That is why orientation appeared to do
nothing. Both now scale z into pixel units from the tracked distance first.
This was the single highest-value fix in the change and it is not the one
section 2 predicted.

**Two defects were caught by the tests rather than by eye.** The inward contour
offset in section 6 was implemented with the normal inverted, so the emission
rings grew *outside* the palm silhouette instead of inside it. And the palm
normal was asserted to respond to tilt, which it did not until the scaling above
was added. Both now have tests that would fail again.

**Hue and level had to be separated.** Section 8's palette, taken literally,
carries more than twice the luminance of the scarlet it replaced, and every gain
downstream is calibrated against that buffer. The emission colour is now
normalised to unit luminance before a level is applied, so the state ramp
changes the colour and one line decides the brightness.

Also added, not planned: a start-up pass that compiles every program, including
the face and person shaders that are otherwise first built at the moment a face
enters frame. It removes a hitch, and it means a shader that will not link fails
at load rather than in front of someone.

Verified in Chrome against a synthetic camera feed: 198 GL programs linked, none
failed, `gl.getError()` 0, all three MediaPipe graphs and the WebGPU depth worker
started. Tracking-dependent behaviour is unverified visually — see the README's
acceptance list.


---

## 14. Second revision

Changes after the first build was reviewed on a live machine.

**The field is now a source standing in the room, not only a screen-space
bloom.** Section 8 assumed the blurred irradiance buffer was enough to light the
subject. It is not: that buffer only knows where light is *on screen*, so a face
next to the field received almost nothing from it however bright it was. The
composite now also unprojects each pixel to a camera-space point using the depth
already in the G-buffer, and shades it against the field's actual position with
a wrapped Lambert term and an inverse-square falloff. That is the difference
between a glow that overlaps a face and a match that lights a room.

**The head bias went from 25 mm to 100 mm.** Section 5's cross-fade band was
correct in form but far too tight in practice — a hand held near the face put
the field right in the uncertain zone. The face mesh describes the *front* of the
face, so biasing it by roughly a head's depth places the occluding surface near
the back of the skull. The field now renders in front of the face anywhere
around it, and is hidden only when it is genuinely behind the head.

**Mask edges are dithered.** Nothing in this document anticipated it, but a
low-resolution occluder leaves a visible staircase where it cuts a luminous
strand. A rotated dot screen applied to the transition band only — solid
interiors and clear exteriors untouched — dissolves that into a halftone, and
the depth comparison band around it was widened at the same time.

**The wave is coloured by its phase.** Section 8 treated colour as a chosen
palette. It can be better than that: `arg(psi)` is a real property of the state
that `|psi|²` discards, and encoding it in hue while density stays in brightness
is the standard domain-colouring convention. The two poles are normalised to
equal luminance so the cycle changes hue and nothing else. The light the field
throws into the room stays near neutral, because a saturated spill on skin reads
as a filter over the shot.

**The interface reduced to three pieces of text.** The masthead and the fixed
instrument panel are gone. What remains is a callout the tracker places beside
the hand — with the dot and the leader drawn by the same code that positions the
text, so the line always meets the block — one standing line, and the reader.
The numeric readouts became meters, for the reason set out in
[RESEARCH.md §7a](RESEARCH.md#7a-levels-not-digits): the relations are exact but
the dial driving them is a gesture, and a digit cannot tell those two apart.

Verified again in Chrome: 199 GL programs linked, none failed. The page-side
`linkProgram` instrumentation caught a composite shader that called the new
halftone helper without declaring it — a break that would otherwise have shipped
as a black frame.


---

## 15. Third revision

**The head no longer occludes at all.** Section 5's depth comparison was the
right mechanism and the wrong answer for this piece: the wave is meant to cross
a face, not pass behind it. The face mesh is now absent from the visibility
pass entirely and present only in the lighting pass, which is where it was
earning its keep — a cheekbone with its own orientation. The bias constant is
gone with it. Stated plainly so it is not mistaken for the old blanket rule:
this is a deliberate art direction, not a limitation of the depth estimate.

**The near field got much larger.** `NEAR_FIELD_RADIUS` went from 0.9 to 2.1
camera units — deliberately larger than the tracked depth range is deep — and
`MATCH_GAIN` roughly doubled. The source now lights the subject and the wall
behind them, not just the hands holding it.

**Colour needed four fixes, not one.** Saturating the poles alone did nothing
visible, because three other things were throwing the hue away downstream: the
filament mixed 30-75% white into it by density, the particles another 22-50%,
and the final grade rolled highlights off **per channel**, which pulls any
bright coloured pixel toward white exactly where the wave is brightest. The
roll-off is now computed on luminance and applied as a scale, so the channel
ratios survive it. The poles and the emission ramp moved into
`lib/field-palette.ts` so their properties — equal luminance, saturation,
level held separate from hue — are asserted by test instead of tuned by eye.

**Compression was unreachable.** The confinement curve did not saturate until
the palms were nearly five palm-widths closer than a clasp, so a natural closing
gesture spent its whole range in the bottom third of the dial. Curve, clasp
gate, approach envelope and every smoothing constant on that path were retuned
together; the state-machine tests were updated to the new thresholds rather than
worked around.

**Two typographic errors caught by screenshot.** `text-transform: lowercase`
turned Δp into δp and E into e — different quantities, not a style. And the
standing line failed to step aside for the reader because the rule is a
following-sibling selector and the element sat *before* the reader in the DOM.
Neither would have shown up in a type check.


---

## 16. Fourth revision: the three states

The brief for this pass was a legible progression rather than a switch, and it
exposed that the confinement dial had been the wrong shape twice over. It had
been mapping only the last third of the closing gesture, so the hands could move
a long way inward before anything on screen changed. It now spans the whole
gesture, from palms at their widest to a clasp, and stays **linear in the gap**
— the acceleration the eye reads should come from `E/E0 = (L0/L)^2`, which is
real, not from a curve bent to feel good.

`critical` is a new state, and a new continuous uniform. The state exists so the
copy can name the thing; the uniform exists because a flag cannot ramp. It is
derived from the dial and falls away as a clasp takes over, and it drives
amplitude, layer spread, grain spread, core and glow width, radiance, emission
gain, near-field strength, refraction, flare and bloom — every one of them
narrowing or brightening. Nothing it drives shakes: kinetic energy here is
momentum spread, and agitation would be teaching the wrong thing.

Two smaller findings. The face's lit area had been the landmark hull, which
stops at the edge of the face and left a lit oval floating on an unlit head; it
is now grown 22% about its own centre, which is free because that surface never
occludes. And `DEPTH_RANGE` was tight enough that a face standing the usual
distance behind the hands received almost nothing regardless of source
brightness.

Typographically, the readouts moved into the body face — they are body text —
which surfaced that Instrument Serif has no U+2080, so `L₀` was falling back to
another face and arriving as `Lo`. Real `<sub>` markup fixes it and keeps the
symbol in one typeface. The monospace was dropped entirely: with the readouts
drawn as meters, nothing left in the interface is a column of changing digits.


---

## 17. Fifth revision

**The hue mapping was wrong, not just timid.** Section 8 and the fourth revision
both described a two-pole phase ramp. A ramp between two colours is a chord
across the hue circle, and `arg(psi)` lives on the circle: the map was not
injective, so two phases equally far from a pole in opposite directions rendered
identically, and half of every cycle passed through near-white carrying almost
no colour at all. It is now the full circle -- three cosines at 120 degrees --
normalised to fixed luminance so brightness is left carrying `|psi|^2` alone.
The properties are asserted in `tests/geometry.test.mjs` rather than judged by
eye: constant luminance around the wheel, all three hue regions reached, and
opposed phases provably distinguishable.

**The callout stops chasing.** Three separate causes of the impression of
jitter, all fixed:

1. It followed whichever palm was more open, so two visible hands made it hop
   between them. It now locks to one slot on acquisition and holds it.
2. It was hidden whenever tracking lapsed, so a dropped frame took the text
   away and put it back. It is never hidden now -- the block holds its position
   and only the leader fades, because only the leader has anything to point at.
3. It repainted only on frames that carried new landmarks. The models run at
   about half the display rate, so the mark stepped even though the signal
   underneath was already filtered. The overlay now advances its follower and
   repaints every animation frame, and the text block carries a short linear
   transform tween for the same reason.

Position smoothing was also slowed well past the tracking's own filtering (0.21
s), and the side it sits on given a 110 px hysteresis band. Text pinned rigidly
to a moving hand reads as panic however clean the underlying signal is; the fix
is deliberate lag, not more filtering.
