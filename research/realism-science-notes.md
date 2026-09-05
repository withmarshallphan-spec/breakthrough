# Scientific basis for the upgraded confinement illustration

Reviewed 5 September 2026. This note supplements the existing research report;
it describes the claims actually made by the browser experience.

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

- [Elliott H. Lieb, Quantum Mechanics, The Stability of Matter and Quantum Electrodynamics (2004)](https://arxiv.org/html/math-ph/0401004v1), especially sections 1–3.

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

The app does not calculate the time-dependent dynamics of moving walls, solve a
hydrogen or many-electron Hamiltonian, model radiation, measure atomic lengths,
or produce physical light from electrons. Its color, glow, transverse volume,
slow animation clock, clasp knot and refraction are visual cues. Closing hands
means spatial confinement, not a measurement-induced wavefunction collapse.
