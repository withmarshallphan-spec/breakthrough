# Scientific basis for the upgraded confinement illustration

Reviewed 5 September 2026. This note supplements the existing research report;
it describes the claims actually made by the browser experience. Citations were
re-fetched on 5 September 2026.

## The question the piece answers

Why doesn't ordinary matter collapse, when negative electrons are attracted to
positive nuclei? The chain the captions walk through is:

1. Coulomb attraction alone has no floor — classically the electron falls in.
2. What has spatial extent is the electron's quantum state, not the electron;
   an electron is treated as a pointlike elementary particle.
3. Localizing a state into a smaller region forces a wider momentum spread, and
   therefore a higher kinetic energy. This is the cost the gesture is dialling.
4. For a hydrogen-like trial state of size L those two terms scale differently,
   +A/L² against −B/L, so the total turns around at a finite size instead of
   running to −infinity. That finite minimum is the size of the atom.
5. For bulk matter, one more ingredient is required: fermionic antisymmetry.
6. A ground state is stationary, so it does not radiate its way inward.

Steps 3 and 4 are what the visual illustrates. Steps 2, 5 and 6 are asserted in
the captions and are not modelled by anything on screen.

## Atomic stability

A negative electron and positive nucleus attract through the Coulomb potential.
For a family of localized, similarly shaped trial wavefunctions with size L,
kinetic energy scales as A/L² and Coulomb energy as −B/L, for positive constants
A and B. This gives a finite minimum, rather than unbounded inward shrinkage.
The uncertainty relation motivates the localization cost; the scaling argument
is not a rigorous many-body stability proof. A ground-state atom also does not
follow a radiating classical orbit: its electronic state has no lower energy
state into which to decay.

- [MIT 6.007, lecture 38: examples of the uncertainty principle](https://ocw.mit.edu/courses/6-007-electromagnetic-energy-from-motors-to-lasers-spring-2011/c51a3b6c694e74f0d3daebfb8a0a0932_MIT6_007S11_lec38.pdf)
- [MIT 5.61, Hydrogen Atom I](https://ocw.mit.edu/courses/5-61-physical-chemistry-fall-2017/resources/mit5_61f17_lec20/)

## Stability of bulk matter

Preventing one atom from collapsing and keeping arbitrarily large collections
of matter stable are distinct questions. Fermionic antisymmetry, expressed
pedagogically by Pauli exclusion, is essential to the latter together with
kinetic energy and attractive/repulsive Coulomb interactions. Spin is part of
the complete single-electron state. Exclusion is not a new classical force.

The sharp form of the claim is Dyson and Lenard's: for fermions the ground-state
energy of N electrons with K fixed nuclei is bounded below by a constant times
the particle number, which is what makes matter extensive and non-collapsing;
for the same system made of bosons the bound degrades to −C(N + K)^(7/5), which
is not extensive. Exclusion is a consequence of antisymmetry, not a new
classical repulsive force between electrons.

- [F. J. Dyson and A. Lenard, Stability of Matter. I, J. Math. Phys. 8, 423 (1967)](https://pubs.aip.org/aip/jmp/article-abstract/8/3/423/235627/Stability-of-Matter-I) — the original proof and the fermion/boson contrast.
- [Elliott H. Lieb, Quantum Mechanics, The Stability of Matter and Quantum Electrodynamics (2004)](https://arxiv.org/abs/math-ph/0401004) — review of why matter is stable "despite the serious −1/|x| singularity of the Coulomb potential". Verified title, author and year.

## What is being shown

The electron is treated as an elementary pointlike particle; its quantum state
can nevertheless be spatially extended. The illustration uses ideal box modes
sin(nπx/L), not atomic orbitals. For a fixed n, the box relation E_n ∝ 1/L² is
exact. The mode weights stay fixed in the visualization so squeezing the well
does not falsely imply that the principal mode number must increase. Amplitudes
use the square roots of mode populations; longitudinal grain brightness follows
the time average of the mode-density sum, up to a display normalization.

- [CERN, Introduction to Particle Physics](https://videos.cern.ch/record/3015252?t=0)
- [CERN, Electron glossary](https://opendata.cern.ch/glossary/Electron)
- [OpenStax, The Quantum Particle in a Box](https://openstax.org/books/university-physics-volume-3/pages/7-4-the-quantum-particle-in-a-box)

## The numbers on screen

Only three quantities are displayed, and none of them is an absolute:

| Readout | Value | Status |
| --- | --- | --- |
| Well width | L/L₀, from 1 to ½ | The dial itself. Dimensionless by construction. |
| Kinetic energy | E/E₀ = (L₀/L)² | Exact for a fixed mode of an ideal box. OpenStax gives Eₙ = n²π²ℏ²/(2mL²), verified. |
| Uncertainty | Δx Δp = 0.568 ℏ at n = 1 | Exact, and independent of L: Δx = L√(1/12 − 1/2π²n²) and Δp = nπℏ/L, so the width cancels. |

The third is the honest punchline. Squeezing does not push the state toward the
ℏ/2 bound; the product is fixed by the mode. What confinement buys is momentum
spread, and that is what costs energy.

An earlier revision anchored the well to a fixed 0.88–0.54 nm range and carried
helpers for transition energies, emitted wavelengths and spectral bands. Those
implied both a measured atomic length and emitted photons, neither of which the
experience models, so they were deleted rather than disclaimed. `lib/quantum.ts`
now contains only physical constants, relations exact for the ideal box, and a
colour ramp that is labelled as false colour.

## What the compositing claims

Depth is estimated monocularly, from apparent size against assumed physical
widths for a hand and a face. Person segmentation contributes a silhouette and
nothing else — a mask has no depth — so the body is composited on a single plane
at the head's estimated distance, with the hand rig overriding it wherever it
reaches. Rim light, transmission through clasped hands, refraction and the
irradiance that relights the room are physically motivated compositing, not
measured radiometry: none of it is calibrated, and none of it is claimed to be.

The app does not calculate the time-dependent dynamics of moving walls, solve a
hydrogen or many-electron Hamiltonian, model radiation, measure atomic lengths,
or produce physical light from electrons. Its color, glow, transverse volume,
slow animation clock, clasp knot and refraction are visual cues. Closing hands
means spatial confinement, not a measurement-induced wavefunction collapse.
