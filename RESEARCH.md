# RESEARCH — physics, copy, numbers, typography, iconography

Written before implementation, 5 September 2026. Companion to
[VFX_RESEARCH.md](VFX_RESEARCH.md), which covers the rendering pipeline.
Supersedes and absorbs [research/realism-science-notes.md](research/realism-science-notes.md).

---

## Part I — The physics

### 1. The question the exhibit is actually about

Not "why is there an uncertainty principle", and not "what does an electron
look like". The question is:

> An electron is attracted to a nucleus by a potential with no floor —
> −e²/4πε₀r runs to −∞ as r → 0. Why does the electron not fall in, and why
> does a kilogram of matter occupy a litre rather than a point?

Those are two different questions with two different answers, and conflating
them is the most common error in popular explanations. The first is **stability
of the first kind**; the second is **stability of the second kind**. Lieb's
1976 review draws exactly this distinction and it is the spine of the copy.

### 2. Stability of the first kind — one atom

**The energy is bounded below.** Formally `E₀ > −∞`.

The mechanism is the kinetic-energy cost of localisation. Take a family of
normalised trial states of characteristic size `L`. Localising a state to a
region of size `L` forces a momentum spread of order `ℏ/L` — this is the
uncertainty relation used as a variational tool, not as a measurement
statement. Kinetic energy `⟨p²⟩/2m` therefore scales as `+A/L²`, while Coulomb
attraction scales as `−B/L`, with `A, B > 0`:

```
E(L) ≈ A/L² − B/L
```

The two terms have different powers, so the sum has a genuine minimum rather
than diverging:

```
dE/dL = 0  →  L* = 2A/B      E(L*) = −B²/4A
```

That finite `L*` **is** the size of the atom. For hydrogen the same algebra
with `A = ℏ²/2m` and `B = e²/4πε₀` returns the Bohr radius (5.29 × 10⁻¹¹ m) and
−13.6 eV, which is the standard textbook check that the scaling argument has
the right content.

Two things the copy must say and that are usually skipped:

- **This is a variational bound, not a derivation of the ground state.** It
  proves the energy cannot run away; solving the Schrödinger equation is what
  gives the actual eigenvalue.
- **A ground state does not radiate.** The classical objection — an orbiting
  charge radiates and spirals in — dissolves because a stationary state has no
  lower electronic state to decay into. Nothing is orbiting.

### 3. Stability of the second kind — bulk matter

**The energy is bounded below *proportionally to the number of particles***.
Formally, for `N` electrons and `K` fixed nuclei,

```
E₀ ≥ −C · (N + K)
```

with `C` independent of `N` and `K`. This linearity is not a technicality: it
is what makes energy **extensive**, what makes the thermodynamic limit exist,
and therefore what makes thermodynamics apply to matter at all. Without it,
merging two rocks would release energy that grows faster than the amount of
rock, and matter would implode as it accumulated.

**One-atom stability does not imply it.** This is the crucial and
counter-intuitive step. Each atom being individually stable says nothing about
whether `N` atoms have energy `∝ N`.

**Fermionic antisymmetry is required.** Dyson and Lenard proved the bound for
fermions in *Stability of Matter I & II* (1967, 1968). The same proof does not
survive for bosons: the boson ground-state energy scales as `−N^{7/5}`, which
is superlinear, so bosonic "matter" is not stable in this sense. That
contrast — same Coulomb forces, same quantum mechanics, only the statistics
changed, and the result is qualitatively different — is the single most
persuasive statement available about what Pauli exclusion actually does.

**Lieb and Thirring (1975)** then gave the proof that made it comprehensible:
a lower bound on the kinetic energy of a fermionic system in terms of an
integral of its own density,

```
T ≥ K ∫ ρ(x)^{5/3} dx
```

the same `ρ^{5/3}` form as Thomas–Fermi theory. This is where exclusion enters
quantitatively: pushing `N` fermions into a small volume forces the density up,
and the kinetic-energy floor rises faster than linearly in the density, which is
what beats the Coulomb attraction. Their constant was also enormously better
than Dyson–Lenard's — the literature records roughly **10¹⁴ Rydberg per
particle down to order 10 Rydberg** — which is the difference between a bound
that exists and a bound that means something.

**Exclusion is not a force.** No term in the Hamiltonian corresponds to it. It
is a constraint on which wavefunctions are admissible: the many-electron state
must be antisymmetric under exchange of any two electrons (including spin). The
energetic consequence is that the low-lying single-particle states cannot all be
occupied at once, so adding electrons costs kinetic energy. Copy must not
describe it as electrons "repelling" or "refusing to share".

### 4. What the visualisation actually models

An **ideal one-dimensional infinite square well** — a fixed superposition of
its stationary modes, with the well width driven by the gesture.

```
ψ_n(x) = √(2/L) sin(nπx/L)        E_n = n²π²ℏ² / 2mL² = n²h² / 8mL²
```

Modelled honestly:

- Mode populations are **fixed**. Squeezing does not promote `n`, because
  nothing in the model excites the state; the point is that `E` rises at fixed
  `n` purely from `L`.
- Grain brightness follows the time-averaged `|ψ|²`, so nodes are genuinely
  empty. That is the probability density, and it is why the visual has
  structure instead of being a glowing tube.
- Phase advance rate uses `n²/L²`, matching `E_n`, on a deliberately slowed
  display clock.

Not modelled, and the copy must not imply otherwise: a hydrogen Hamiltonian,
many-electron dynamics, moving-wall dynamics, radiation, measurement collapse.
**Closing your hands is spatial confinement, not wavefunction collapse.** This
is worth saying explicitly because the gesture invites exactly that
misreading.

### 5. Does the research change the effect?

Reviewed the candidate representations — probability density, momentum-space
broadening, spatial frequency, corral-style standing waves, point-cloud
volumes — against what is on screen.

**Keep:** the standing-mode superposition with visible nodes and `|ψ|²`-driven
grain density. It is a truthful picture of a confined state, it is the standard
pedagogical object, and it survives being made cinematic.

**Change, and colour the wave by phase — over the whole hue circle.** A
probability density is real but it is not the whole state: `|psi|^2` throws the
phase away, and the phase is what makes a wavefunction a wave rather than a
cloud. Encoding density in brightness and `arg(psi)` in hue is the standard
domain-colouring convention, and it is the encoding Lundeen et al. rely on when
they distinguish amplitude from phase.

The mapping has to be the **full circle**, not a two-colour ramp. `arg(psi)`
lives on a circle; a ramp between two poles is a chord across it, and a chord is
not injective — two phases equally far from a pole in opposite directions get
the same colour, and the state looks the same at moments when it demonstrably is
not. Three cosines at 120 degrees give the circle, so hue and phase are
one-to-one.

The wheel is normalised to **fixed luminance**. This is not a stylistic choice:
a raw hue circle varies by about a factor of two in luminance between yellow and
blue, brightness is already carrying `|psi|^2`, and an unnormalised wheel would
therefore make the colour a second, wrong statement about the density.

Colouring the wave is not a departure from the research — it is the research's
own recommendation, and the earlier flat tint was the less truthful choice.

**Change, because it is the actual mechanism and it is currently only implied:**
spatial frequency and momentum spread. For a fixed mode, `k_n = nπ/L`, so
squeezing the well shortens the wavelength — the structure gets *finer*, not
just brighter and faster. And `Δp = nπℏ/L` widens by exactly the same factor.
The renderer already shortens the on-screen wavelength as a side effect of the
hands coming together; making that legible, and showing the momentum spread
widening alongside it, is the honest version of "confinement costs energy".

**Reject:**
- Electrons as bouncing balls. The electron is treated as pointlike; what has
  extent is the state.
- Random shaking as shorthand for kinetic energy. Kinetic energy here is
  momentum *spread*, not agitation, and shaking would teach the wrong thing.
- Turning it into a labelled textbook diagram. Axes and formulas on screen
  would trade the exhibit for a lecture slide.

---

## Part II — Audit of every number

Rule applied: **a number appears only if the code computes it from the model
that is actually running.** Anything mapped from hand distance for effect is
removed rather than disclaimed.

### 6. Current state of the build

Audited `app/page.tsx`, `lib/quantum.ts`, `lib/wave-engine.ts`. Findings:

| Quantity | Where | Verdict |
| --- | --- | --- |
| nm well width | **not present** | Removed in a previous pass. Was decorative — no atomic length can be inferred from a gesture. Stays removed. |
| eV energies | **not present** | `levelEnergy()` exists in `lib/quantum.ts` but nothing renders it. |
| `n = 1` label | **not present** | The visual is a 6-mode superposition, so labelling it `n=1` would have been wrong anyway. |
| "Energy: Low / Rising / High" | `app/page.tsx` | Qualitative. Honest but says nothing. |
| Energy bar | `normalisedEnergy(confinement)` | Derived: `(E/E₀ − 1)/3`. Real, but unlabelled, so the viewer cannot know what it measures. |

So the audit finds **no fake precision to remove** — the previous pass already
did that — but it also finds that the honest quantities that *are* computed are
not shown at all. The exhibit is currently mute where it could legitimately
speak.

### 7. What may be displayed, and why each is legitimate

Four dimensionless quantities, all exact for the model that is running, all
verified numerically against `lib/quantum.ts`:

| Readout | Expression | Range | Why it is honest |
| --- | --- | --- | --- |
| Well width | `L/L₀ = 2^(−c)` | 1 → 0.500 | The dial itself, dimensionless by construction. No atomic length claimed. |
| Kinetic energy | `E/E₀ = (L₀/L)²` | 1 → 4.00 | Exact for a fixed mode of an ideal box. Follows from `E_n ∝ 1/L²`. |
| Momentum spread | `Δp/Δp₀ = L₀/L` | 1 → 2.00 | Exact: `Δp = nπℏ/L`. This is the mechanism, stated directly. |
| Uncertainty product | `Δx·Δp = 0.568 ℏ` at `n = 1` | constant | Exact and **independent of `L`**: `Δx = L√(1/12 − 1/2π²n²)`, `Δp = nπℏ/L`, so `L` cancels. |

Numerically verified by running `lib/quantum.ts`:

```
GROUND_UNCERTAINTY        0.5678618083866118      (> 0.5, as required)
uncertaintyProduct(2)     1.6702898352371223
relativeWellWidth(1)      0.5        energyRatio(1)   4
normalisedEnergy(0)       0          normalisedEnergy(1)  1
```

The uncertainty product is the honest punchline and it will be shown as a
constant, on purpose: **squeezing does not push the state toward the ℏ/2
bound.** It cannot — the product is fixed by the mode. What squeezing buys is
momentum spread, and momentum spread is what costs energy. A viewer who expects
it to drop toward 0.5 and watches it refuse to move has learned something real.

### 7a. Levels, not digits

A second audit question, asked after the first implementation: *are these
numbers approximated?*

The **relations** are not. Each is exact for the ideal box, and the tests check
them to machine precision. But the **input** is: the confinement dial is a
smoothed estimate of the distance between two hands, from landmarks, through an
assumed focal length. Printing `L/L₀ = 0.946` therefore claims three significant
figures for a quantity whose input has nothing like that precision. The relation
is exact; the argument fed to it is not, and a digit does not distinguish
between those two things.

So the interface shows **meters, not digits**. A meter states exactly what the
gesture supports: where the value sits on its own range, and which way it is
moving. Each carries its symbol so the relation is still named. The one place a
figure would have been legitimate — the uncertainty product, which is a constant
of the mode and depends on no input at all — is instead shown as a fixed marker
sitting above the ℏ/2 bound, because *that it does not move* is the reading, and
a stationary number invites the viewer to wait for it to change.

Meter construction follows the standard contract: one ratio against its own
range, a track that is a dim step of the same ramp the fill uses, labels in text
tokens rather than in the data colour, and a single sequential ramp — ice to
white — shared with the field itself, so the fill and the light in the room are
visibly the same scale.

Everything else stays qualitative. The visualisation is labelled in the
interface as a **model** — an ideal one-dimensional box, used as an analogy —
not as a simulation of an atom.

---

## Part III — Copy plan

### 8. Register and audience

Strong high-school through early undergraduate. Closer to a museum wall text
written by someone who knows the mathematics than to a video script. Real
terminology, defined at first use, never used as decoration.

Banned constructions, with the reason: *"the wave gets angry"* (attributes
intent), *"nature doesn't like being squeezed"* (teleology), *"the electron
becomes a cloud"* (conflates the particle with its state), *"observation
collapses it"* (irrelevant here and actively misleading given the gesture).

### 9. Planned sections

Short enough for an interactive — target 40–70 words each, two to four
sentences.

1. **The problem** — Coulomb attraction has no floor; classically the electron
   falls in. Define the Coulomb potential.
2. **What has extent** — the state, not the particle. Define wavefunction and
   probability density. Point at the nodes on screen as the evidence.
3. **The cost of localisation** — narrow the state, widen the momentum
   distribution, raise the kinetic energy. Define momentum spread. This is the
   section the gesture is directly driving.
4. **Where the size comes from** — `+A/L²` against `−B/L`, a finite minimum,
   and that minimum is the atom. Note it is a variational argument.
5. **Why one atom is not enough** — introduce stability of the second kind and
   the linear-in-`N` bound, and say plainly that atomic stability does not
   imply it.
6. **Antisymmetry** — the Pauli principle as a constraint on admissible
   wavefunctions, not a force. The Dyson–Lenard fermion bound against the
   boson `−N^{7/5}` scaling as the decisive contrast.
7. **What this is and is not** — an ideal 1D box driven by a gesture; the three
   dimensionless readouts; not an atom, not a measurement, not collapse.

Each section carries its own source link. State-linked lines (short, one
sentence) accompany OPEN / COMPRESSING / HIGH CONFINEMENT / CLASPED and
describe *what the visualisation is representing at that moment*, not how it
feels.

### 10. Sources

- Elliott H. Lieb, *The stability of matter*, Rev. Mod. Phys. **48**, 553–569 (1976) — the first/second-kind distinction. https://doi.org/10.1103/RevModPhys.48.553
- Elliott H. Lieb, [*Quantum Mechanics, The Stability of Matter and Quantum Electrodynamics*](https://arxiv.org/abs/math-ph/0401004) (2004) — review; the "−1/|x| singularity" framing.
- F. J. Dyson and A. Lenard, [*Stability of Matter I*](https://pubs.aip.org/aip/jmp/article-abstract/8/3/423/235627/Stability-of-Matter-I), J. Math. Phys. **8**, 423 (1967); *II*, **9**, 698 (1968) — the fermion bound and the boson contrast.
- E. H. Lieb and W. E. Thirring, *Bound for the Kinetic Energy of Fermions which Proves the Stability of Matter*, Phys. Rev. Lett. **35**, 687–689 (1975), erratum 1116 — the `ρ^{5/3}` kinetic-energy inequality.
- E. H. Lieb and R. Seiringer, *The Stability of Matter in Quantum Mechanics*, Cambridge University Press (2010) — standard reference text.
- N. Straumann, [*The Role of the Exclusion Principle for Atoms to Stars: A Historical Account*](https://arxiv.org/abs/quant-ph/0403199) (2004).
- [OpenStax University Physics Vol. 3, §7.4 The Quantum Particle in a Box](https://openstax.org/books/university-physics-volume-3/pages/7-4-the-quantum-particle-in-a-box) — `E_n`, `ψ_n`, and the well's `Δx`.
- [MIT 8.04 / Cornell notes, *The Energy of Confinement*](https://muchomas.lassp.cornell.edu/8.04/Lecs/lec_Heisenberg/node3.html) — the `A/L² − B/L` variational argument.
- [MIT 5.61, Hydrogen Atom I](https://ocw.mit.edu/courses/5-61-physical-chemistry-fall-2017/resources/mit5_61f17_lec20/) — Bohr radius and ground-state energy as the check on that argument.
- [CERN, Electron glossary](https://opendata.cern.ch/glossary/Electron) — the electron as a pointlike elementary particle.

---

## Part IV — Typography

### 11. Display face

`GMisk` was searched on Google Fonts, Fontshare, dafont, MyFonts, Behance,
1001fonts and general web search. **No typeface of that name is publicly
distributable or licensable**, under that spelling or the near variants. It was
therefore not substituted silently — it was raised, and the replacement was
chosen by the project owner:

**Alte Haas Grotesk**, Yann Le Coroller (2007). A grotesk drawn to look like
Helvetica letterpress-printed in an old Müller-Brockmann book — slightly
softened corners, uneven weight. Editorial and warm rather than technical,
which is the intended contrast against Geist Sans.

**Licence, verified from the file shipped with the font:**

> These fonts are freeware and can be distributed as long as they are together
> with this text file. — yann le coroller

So the legal way to load it is: **self-host, and ship the licence text with
it.** Done — the original `.ttf` files were downloaded from the author's
distribution on dafont, converted to WOFF2 with `fontTools`, and placed in
`public/fonts/alte-haas-grotesk/` **together with `LICENSE.txt`**, as the
licence requires. Two weights, Regular and Bold, 48 KB and 49 KB.

Not used: any third-party font CDN mirror (no licence chain), and no webfont
service — the font is not on one.

### 12. Body face

**Geist Sans**, Vercel, **SIL Open Font License 1.1**. Already in the project
via `next/font/google`, which self-hosts the files at build time — the correct
mechanism, no runtime request to Google, no CLS. Kept unchanged. `Geist Mono`,
same licence, is kept for the numeric readouts only, where tabular figures
matter.

### 13. Hierarchy

Revised twice. The display face is set in **Bold**, the reading copy is
**Instrument Serif** (Google Fonts, SIL OFL 1.1, weight 400 only), everything is
a size larger than a screen interface would normally take, nothing wears a
shadow, and the interface is set **lowercase**.

Lowercasing is applied once, on the root, and taken back off in exactly three
places — because there it would be an error rather than a style:

| Exempt | Why |
| --- | --- |
| Meter symbols | `text-transform: lowercase` turns **Δ into δ** and **E into e**. Δp is a momentum spread; δp is not. E is an energy; e is an elementary charge. |
| The physics prose | It carries `N`, `K`, `T`, `ψ` — and `N` lowercased collides with the `n` used for the mode number in the same document — plus Coulomb, Pauli, Dyson, Lenard, Lieb and Thirring. |
| Citations | They are the real titles of real papers. |

Legibility rests on a scrim rather than a shadow: a radial gradient sized with
`closest-side`, so it reaches transparent on its own nearest edge and can never
present a straight boundary.

| Role | Face | Treatment |
| --- | --- | --- |
| Exhibit title, section heads, state | Alte Haas Grotesk **Bold** | tight leading, slightly negative tracking |
| Reading copy, meter labels, symbols and figures | Instrument Serif 400 | 1.55 line height, ~52ch measure |
| Small captions and controls | Geist Sans 400 | the one tier a display serif stops being legible at |

Two faces carry almost everything. A monospace was dropped once the readouts
became meters: nothing left in the interface is a column of changing digits, so
tabular figures had no job, and the symbols are body text like the prose around
them. Subscripts are set as real `<sub>` markup rather than U+2080, which
Instrument Serif does not contain — a literal `₀` falls back to another face and
arrives looking like the letter o. No uppercase — the letterforms carry the
hierarchy, not the casing. No condensed faces, no letterspaced sci-fi lettering,
no glow on text, and no vertical rules anywhere: where a relationship between
two things has to be shown, it is drawn as a leader line from a dot.

---

## Part V — Iconography

### 14. Choice: Lucide

**[Lucide](https://lucide.dev/license), ISC licence.** Already a dependency
(`lucide-react` 1.31.0), so no new supply chain. Chosen over the alternatives
for one specific reason: Lucide's stroke width is a numeric prop, so a
`strokeWidth={1.25}` hairline set is available from the same icons that ship at
2. Phosphor (MIT) offers hand-drawn weights, which are better at heavy sizes
but do not go as thin; Material Symbols (Apache 2.0) is a variable font and
excellent, but adding a second icon system for no gain is worse than using the
one already installed consistently.

Icons used, all at 16 px and `strokeWidth 1.25`, monochrome, no container, no
panel: `Aperture` (capture), `Clapperboard` (film mode), `Maximize` /
`Minimize` (fullscreen), `Eye` / `EyeOff` (interface visibility), `SwitchCamera`
(camera source, only when more than one exists).

Treatment: 28 px hit target, no border, no background, low base opacity rising
on hover, tooltip only where the icon is genuinely ambiguous. They sit directly
on the viewport.
