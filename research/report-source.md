# Visual Direction for “Too Expensive to Collapse”

**Audience:** filmmaker and prototype designer  
**Date:** 2026-09-04  
**Scope:** Scientific and visual grounding for a qualitative, hand-controlled 3D quantum-confinement shot.  
**Assumptions:** The experience is explicitly conceptual rather than numerically predictive; its job is to communicate the kinetic-energy cost of localization in a legible cinematic image.

## Executive answer

Replace the single curve with a point-sampled 3D field contained by a faint boundary. The field should occupy almost the whole view when unconstrained and contract as the gesture closes. Density, opacity, and point size should communicate spatial probability; a restrained two-color phase cycle should imply that the wavefunction is complex and dynamic. Compression should preserve the basic standing-wave topology while making the world-space structure smaller, the phase motion faster, and the emission more intense.

## Evidence and implications

1. A wavefunction in a three-dimensional box is a function across the volume and separates into spatial components; its energy contains contributions from all three directions and scales inversely with the square of the box length. **Implication:** the visual object should fill a volume, and compressing its dimensions should raise activity without requiring fake numerical readouts. [Simon Fraser University, “Particle in a 3D box”](https://www.sfu.ca/~boal/385lecs/385lec17.pdf)

2. The uncertainty-principle argument gives a nonzero kinetic-energy cost merely from confinement to a finite region, independent of the mechanism producing that confinement. **Implication:** the hands can act as a conceptual boundary without pretending to model a particular atom or potential. [MIT Physics 8.04 notes, “The Energy of Confinement”](https://muchomas.lassp.cornell.edu/8.04/Lecs/lec_Heisenberg/node3.html)

3. The original quantum-corral experiment produced discrete resonances and a spatial density pattern dominated by the eigenstate density of an electron trapped in a two-dimensional box. **Visual reference:** use a quiet boundary plus interference structure inside it, rather than a solid object or a particle orbit. [Crommie, Lutz, and Eigler, Science 262 (1993)](https://pubmed.ncbi.nlm.nih.gov/17841867/)

4. QMBlender argues that 3D wavefunctions are usefully shown as point clouds sampled throughout space because outer isosurfaces can hide internal layers and smooth away high-frequency detail. **Visual reference:** a deep particle volume, not one opaque shell. [Figueiras et al., Journal of Computational Science 35 (2019)](https://doi.org/10.1016/j.jocs.2019.06.001)

5. Direct wavefunction measurements distinguish complex amplitude and phase from probability density. **Visual reference:** encode density in brightness/opacity and phase in a limited color transition. [Lundeen et al., Nature 474 (2011)](https://doi.org/10.1038/nature10120)

## Recommended composition

- Full-bleed webcam with a volumetric particle field occupying the camera space.
- Tens of thousands of depth-sorted-looking luminous points, sampled with a box-state density bias but retaining a faint population throughout the volume.
- One nearly invisible wire boundary for depth and confinement; no axes, formulas, panels, or explanatory labels.
- Cool mint and pale gold as opposite phase cues; brightness carries density.
- Compression contracts the field in x, y, and z, accelerates phase, increases flicker and point intensity, and tightens the boundary.
- UI reduced to a camera glyph, a live dot, a fullscreen glyph, and a short row of energy pips.

## Material limitations

- This is not a numerical Schrödinger solver, and the point cloud is not a measurement.
- Faster motion and emission under compression are cinematic encodings of increased energy cost, not literal observables.
- A rectangular boundary is chosen for the particle-in-a-box analogy; it should not be read as an atomic orbital.

## Claim-to-source ledger

| Claim | Source | Publisher / author | Date | Access note |
|---|---|---|---|---|
| 3D box wavefunctions span the volume; energy adds across x/y/z and scales with inverse length squared | “Particle in a 3D box” | David Boal, Simon Fraser University | 2003 | Public lecture PDF |
| Finite localization imposes a minimum kinetic-energy cost | “The Energy of Confinement” | Tomas Arias, MIT Physics 8.04 | 1995 | Public course notes hosted by Cornell |
| Atomic corrals produce confined standing-wave density patterns and discrete resonances | “Confinement of electrons to quantum corrals on a metal surface” | Crommie, Lutz, Eigler / Science | 1993 | Abstract indexed by PubMed; DOI 10.1126/science.262.5131.218 |
| Particle clouds can represent 3D wavefunction density while retaining interior and high-frequency structure | “QMBlender” | Figueiras et al. / Journal of Computational Science | 2019 | Publisher abstract and article text; DOI 10.1016/j.jocs.2019.06.001 |
| Wavefunction phase and modulus-squared density are distinct quantities | “Direct measurement of the quantum wavefunction” | Lundeen et al. / Nature | 2011 | Public author-hosted PDF; DOI 10.1038/nature10120 |

## Search record and stop decision

Searched official university course materials, original quantum-corral research, original wavefunction-measurement research, scientific-visualization papers, and image results for quantum corrals, 3D probability clouds, phase-color encoding, and WebGL wavefunction rendering. Stopped after the physical premise, 3D representation method, phase/density encoding, and boundary reference each had direct support and further results repeated the same visual patterns.
