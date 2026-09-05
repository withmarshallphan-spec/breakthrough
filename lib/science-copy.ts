import type { FieldState } from './field-state';

export type Source = { label: string; href: string };

export type Section = {
  id: string;
  /** Short label for the index. */
  index: string;
  title: string;
  body: string[];
  source: Source;
};

/**
 * Exhibit copy for the interactive. Written for strong high-school through
 * early undergraduate: real terminology, defined where it first appears, and
 * nothing asserted that the visualisation does not represent. Sections are
 * short on purpose -- this is wall text beside a moving image, not a chapter.
 */
export const SECTIONS: Section[] = [
  {
    id: 'coulomb',
    index: '01',
    title: 'A potential with no floor',
    body: [
      'An electron and a nucleus attract through the Coulomb potential, which falls off as −1/r. It has no lower bound: the closer the electron falls, the more energy is released, without limit.',
      'Classically, matter should collapse immediately. It does not. The reason is not a force holding the electron away from the nucleus.',
    ],
    source: {
      label: 'Lieb, Quantum Mechanics, The Stability of Matter and QED (2004)',
      href: 'https://arxiv.org/abs/math-ph/0401004',
    },
  },
  {
    id: 'state',
    index: '02',
    title: 'What actually has extent',
    body: [
      'The electron is treated as a pointlike elementary particle. What occupies space is its quantum state, described by a wavefunction ψ.',
      'The squared magnitude |ψ|² is the probability density — where the state carries weight and where it carries none. The dark bands across the field are nodes: places the density genuinely vanishes.',
    ],
    source: {
      label: 'OpenStax University Physics III, §7.4 The Quantum Particle in a Box',
      href: 'https://openstax.org/books/university-physics-volume-3/pages/7-4-the-quantum-particle-in-a-box',
    },
  },
  {
    id: 'localisation',
    index: '03',
    title: 'Localisation costs kinetic energy',
    body: [
      'Position spread and momentum spread are conjugate: narrowing a state in space forces its momentum distribution to widen. Their product cannot fall below ℏ/2.',
      'A wider momentum distribution means a larger ⟨p²⟩, and kinetic energy is ⟨p²⟩/2m. Confinement therefore has a price, paid in kinetic energy. That price is what the distance between your hands is setting.',
    ],
    source: {
      label: 'MIT 8.04 notes, The Energy of Confinement',
      href: 'https://muchomas.lassp.cornell.edu/8.04/Lecs/lec_Heisenberg/node3.html',
    },
  },
  {
    id: 'size',
    index: '04',
    title: 'Where the size of an atom comes from',
    body: [
      'For a family of trial states of characteristic size L, kinetic energy scales as +A/L² and Coulomb attraction as −B/L. The powers differ, so the total turns around at a finite L rather than running to −∞.',
      'That minimum is the size of the atom: for hydrogen, about 0.053 nm at −13.6 eV. It is a variational argument — it proves the energy cannot run away, and does not by itself solve for the ground state.',
    ],
    source: {
      label: 'MIT 5.61, Hydrogen Atom I',
      href: 'https://ocw.mit.edu/courses/5-61-physical-chemistry-fall-2017/resources/mit5_61f17_lec20/',
    },
  },
  {
    id: 'bulk',
    index: '05',
    title: 'One atom is not enough',
    body: [
      'That argument gives stability of the first kind: the energy is bounded below. Bulk matter needs something stronger. For N electrons and K nuclei the ground-state energy must satisfy E₀ ≥ −C(N + K), with C independent of how much matter there is.',
      'This linearity is what makes energy extensive, what makes a thermodynamic limit exist, and therefore what makes thermodynamics apply to matter at all. Stability of one atom does not imply it.',
    ],
    source: {
      label: 'Lieb, The stability of matter, Rev. Mod. Phys. 48, 553 (1976)',
      href: 'https://doi.org/10.1103/RevModPhys.48.553',
    },
  },
  {
    id: 'exclusion',
    index: '06',
    title: 'Antisymmetry, not repulsion',
    body: [
      'The Pauli exclusion principle is not a force — no term in the Hamiltonian corresponds to it. It is a constraint on which states are admissible: the many-electron wavefunction must change sign when any two electrons are exchanged.',
      'Dyson and Lenard proved the linear bound holds for fermions. For bosons it fails: the ground-state energy falls as −N⁷ᐟ⁵, faster than linearly. Same Coulomb forces, same quantum mechanics — only the statistics differ.',
    ],
    source: {
      label: 'Dyson & Lenard, Stability of Matter I, J. Math. Phys. 8, 423 (1967)',
      href: 'https://pubs.aip.org/aip/jmp/article-abstract/8/3/423/235627/Stability-of-Matter-I',
    },
  },
  {
    id: 'floor',
    index: '07',
    title: 'The kinetic floor',
    body: [
      'Lieb and Thirring made the mechanism quantitative: for fermions the kinetic energy is bounded below by an integral of the density itself, T ≥ K∫ρ⁵ᐟ³ — the same form Thomas–Fermi theory predicts.',
      'Crowding fermions into a small volume raises ρ, and this floor rises faster than the Coulomb attraction can, which is what stops the collapse. Their bound also cut the constant in the proof from astronomically large to physically meaningful.',
    ],
    source: {
      label: 'Lieb & Thirring, Phys. Rev. Lett. 35, 687 (1975)',
      href: 'https://doi.org/10.1103/PhysRevLett.35.687',
    },
  },
  {
    id: 'model',
    index: '08',
    title: 'What this visualisation is',
    body: [
      'An ideal one-dimensional infinite square well, with fixed mode populations, whose width is driven by the distance between your hands. Brightness follows the time-averaged |ψ|², which is why the nodes stay dark.',
      'Only three quantities are displayed, because only three are exact within that model. It is not an atom, not a measurement, and not wavefunction collapse: closing your hands is spatial confinement and nothing more.',
    ],
    source: {
      label: 'CERN, the electron as an elementary particle',
      href: 'https://opendata.cern.ch/glossary/Electron',
    },
  },
];

/**
 * One line per state, describing what the image is representing at that
 * moment. Deliberately not affective: the visualisation is not feeling
 * anything and neither is the state it depicts.
 */
export const STATE_NOTES: Record<FieldState, { title: string; note: string }> = {
  dormant: {
    title: 'Waiting',
    note: 'Show both palms to the camera. The distance between them sets the width of the well.',
  },
  open: {
    title: 'Broad state',
    note: 'A wide well. Long wavelengths, a narrow momentum distribution, low kinetic energy.',
  },
  compressing: {
    title: 'Confining',
    note: 'The well is narrowing. Spatial structure tightens and the momentum distribution widens with it.',
  },
  critical: {
    title: 'Near collapse',
    note: 'The narrowest the state gets before the hands close on it. Short wavelengths, a wide momentum distribution, and the kinetic-energy cost near its maximum.',
  },
  clasped: {
    title: 'Compressed',
    note: 'Held at the smallest width the gesture reaches. Kinetic energy scales as 1/L², so it has risen fourfold.',
  },
  release: {
    title: 'Releasing',
    note: 'The well reopens. Momentum spread narrows and the kinetic-energy cost falls away.',
  },
};

export type ReadoutId = 'width' | 'energy' | 'momentum' | 'uncertainty';

/**
 * Every readout the interface is allowed to show, with the reason it is
 * allowed. Each one is exact for the ideal box that the renderer is actually
 * running, and each is dimensionless -- no atomic length is inferred from a
 * gesture, so none is displayed.
 */
export const READOUTS: Record<ReadoutId, { label: string; symbol: string; note: string }> = {
  width: {
    label: 'Well width',
    symbol: 'L / L₀',
    note: 'The dial itself, as a fraction of the open width. Dimensionless by construction.',
  },
  energy: {
    label: 'Kinetic energy',
    symbol: 'E / E₀',
    note: 'Exact for a fixed mode of an ideal box: E ∝ 1/L². Mode populations never change, so this is the cost of confinement alone.',
  },
  momentum: {
    label: 'Momentum spread',
    symbol: 'Δp / Δp₀',
    note: 'Exact: Δp = nπℏ/L. Halving the width doubles the spread. This is the mechanism, stated directly.',
  },
  uncertainty: {
    label: 'Uncertainty product',
    symbol: 'Δx·Δp',
    note: 'Fixed by the mode and independent of L — the width cancels. Squeezing does not push the state toward the ℏ/2 bound; it buys momentum spread, and that is what costs energy.',
  },
};

export const MODEL_CAVEAT =
  'Model and analogy: an ideal one-dimensional infinite square well, not a solution of any atomic Hamiltonian.';
