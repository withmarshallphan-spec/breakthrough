'use client';

import { useEffect, useRef, useState } from 'react';
import { Camera, Expand, Info, Video, X } from 'lucide-react';
import type { WaveEngine } from '@/lib/wave-engine';
import type { FieldState, HandTracker } from '@/lib/hand-tracker';
import { normalisedEnergy } from '@/lib/quantum';

// The argument the film makes, in the order it makes it. These are the captions
// the experience is built to teach; the field on screen is the illustration.
const BEATS = [
  {
    title: 'Why doesn’t matter collapse?',
    body: 'Negative electrons attract positive nuclei. Electrical attraction favors a smaller atom. Why does an atom keep a finite size?',
  },
  {
    title: 'An extended state, a pointlike electron',
    body: 'An electron is treated as a pointlike fundamental particle. Its wavefunction can extend through space. The light illustrates that state; it is not the electron’s material shape.',
  },
  {
    title: 'Localization has an energy cost',
    body: 'A narrower position distribution requires a larger momentum spread: Δx Δp ≥ ħ/2. For a fixed mode in a box, halving the width quadruples the kinetic energy.',
  },
  {
    title: 'An atom has a lowest-energy size',
    body: 'For a hydrogen-like trial state of size L, kinetic energy grows as +A/L² and attraction as −B/L. At very small L the positive cost wins, giving a finite energy minimum.',
  },
  {
    title: 'Many electrons also obey exclusion',
    body: 'Two electrons cannot share the same complete quantum state, including spin. Together with quantum kinetic energy and Coulomb interactions, this is essential to the stability of bulk matter.',
  },
  {
    title: 'A ground state does not spiral inward',
    body: 'An atomic ground state is a stationary quantum state, not a classical orbit that radiates away its energy. There is no lower electronic state for it to decay into.',
  },
] as const;

const STATE_COPY: Record<FieldState, string> = {
  dormant: 'Waiting',
  open: 'Open',
  compressing: 'Compressing',
  clasped: 'Sealed',
  release: 'Releasing',
};

// What is happening physically, in the moment it happens. This is the running
// commentary on the gesture, distinct from the argument in the caption panel.
const STATE_NOTE: Record<FieldState, { title: string; body: string }> = {
  dormant: {
    title: 'No well',
    body: 'Raise both palms to define an illustrative well, or pinch with one hand. Move them anywhere in the frame.',
  },
  open: {
    title: 'A wide well',
    body: 'Your palms represent boundaries. A wider well allows a broader quantum state and a lower kinetic energy for each fixed mode.',
  },
  compressing: {
    title: 'Paying to confine',
    body: 'Narrowing the well shortens every allowed wavelength. Shorter wavelength means higher momentum, and the kinetic energy climbs as 1/L².',
  },
  clasped: {
    title: 'Confinement, made visible',
    body: 'The bridge gives way to a trapped knot. Its light is a visual metaphor for increasing confinement cost; this is not a new quantum phase or trapped photons.',
  },
  release: {
    title: 'Letting go',
    body: 'Reopening lowers the illustrated fixed-mode energy scale. The state broadens again. The gesture does not model the dynamics of work or photon emission.',
  },
};

export default function Home() {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const webglRef = useRef<HTMLCanvasElement>(null);
  const trackingRef = useRef<HTMLCanvasElement>(null);
  const waveRef = useRef<WaveEngine | null>(null);
  const handTrackerRef = useRef<HandTracker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastMeterUpdateRef = useRef(0);
  const [cameraState, setCameraState] = useState<'idle' | 'starting' | 'live' | 'error'>('idle');
  const [trackingLabel, setTrackingLabel] = useState('Camera ready');
  const [confinement, setConfinement] = useState(0);
  const [fieldState, setFieldState] = useState<FieldState>('dormant');
  const [hands, setHands] = useState(0);
  const [filmMode, setFilmMode] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [beat, setBeat] = useState(0);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [activeCameraId, setActiveCameraId] = useState('');
  const [switchingCamera, setSwitchingCamera] = useState(false);

  useEffect(() => {
    let mounted = true;
    void import('@/lib/wave-engine').then(({ createWaveEngine }) => {
      if (mounted && webglRef.current && videoRef.current) {
        waveRef.current = createWaveEngine(webglRef.current, videoRef.current);
      }
    });

    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.target instanceof HTMLSelectElement || event.target instanceof HTMLInputElement) return;
      const key = event.key.toLowerCase();
      if (key === 'd') return handTrackerRef.current?.setDebug(event.shiftKey);
      if (key === 'escape') return setShowInfo(false);
      if (key === 'i') return setShowInfo((current) => !current);
      if (key === 'arrowright') return setBeat((current) => Math.min(current + 1, BEATS.length - 1));
      if (key === 'arrowleft') return setBeat((current) => Math.max(current - 1, 0));
      if (key !== 'f') return;
      setFilmMode((current) => {
        const next = !current;
        handTrackerRef.current?.setDebug(false);
        return next;
      });
    };
    window.addEventListener('keydown', onKey);

    return () => {
      mounted = false;
      window.removeEventListener('keydown', onKey);
      handTrackerRef.current?.destroy();
      waveRef.current?.destroy();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function startCamera(deviceId?: string) {
    const previousStream = streamRef.current;
    let nextStream: MediaStream | null = null;
    let cameraAttached = false;
    if (previousStream) setSwitchingCamera(true);
    else setCameraState('starting');

    try {
      nextStream = await navigator.mediaDevices.getUserMedia({
        video: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }),
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = nextStream;
        await videoRef.current.play();
        handTrackerRef.current?.destroy();
        handTrackerRef.current = null;
      }

      previousStream?.getTracks().forEach((track) => track.stop());
      streamRef.current = nextStream;
      cameraAttached = true;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((candidate) => candidate.kind === 'videoinput');
      const actualDeviceId = nextStream.getVideoTracks()[0]?.getSettings().deviceId ?? deviceId ?? videoDevices[0]?.deviceId ?? '';
      setActiveCameraId(actualDeviceId);
      setCameraDevices(videoDevices);
      setCameraState('live');
      setTrackingLabel('Loading hand tracking');

      if (videoRef.current && trackingRef.current) {
        const { createHandTracker } = await import('@/lib/hand-tracker');
        handTrackerRef.current?.destroy();
        const tracker = await createHandTracker(videoRef.current, trackingRef.current, (update) => {
          waveRef.current?.setTracking(update);
          const now = performance.now();
          if (now - lastMeterUpdateRef.current > 110) {
            lastMeterUpdateRef.current = now;
            setConfinement(update.confinement);
            setHands(update.hands);
            setFieldState(update.state);
            setTrackingLabel(update.label);
          }
        });
        tracker.setDebug(false);
        handTrackerRef.current = tracker;
      }
    } catch {
      if (cameraAttached) {
        setCameraState('live');
        setTrackingLabel('Tracking unavailable; ambient mode');
      } else if (previousStream) {
        nextStream?.getTracks().forEach((track) => track.stop());
        if (videoRef.current) videoRef.current.srcObject = previousStream;
        setCameraState('live');
        setTrackingLabel('Camera switch unavailable');
      } else {
        nextStream?.getTracks().forEach((track) => track.stop());
        setCameraState('error');
        setTrackingLabel('Camera unavailable');
      }
    } finally {
      setSwitchingCamera(false);
    }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) void stageRef.current?.requestFullscreen();
    else void document.exitFullscreen();
  }

  function showFilmMode() {
    setFilmMode(true);
    handTrackerRef.current?.setDebug(false);
  }

  // The bars show proportionalities, never values. The relations are the real
  // ones for a particle in a box; what is missing on purpose is any number,
  // because hand separation is a dial and not a measurement of anything.
  const idle = hands === 0;
  const confinementWord = idle ? 'None' : confinement > .76 ? 'High' : confinement > .36 ? 'Rising' : 'Low';
  const costWord = idle ? 'None' : confinement > .7 ? 'High' : confinement > .34 ? 'Rising' : 'Low';
  const cost = normalisedEnergy(confinement);
  const current = BEATS[beat];
  const note = STATE_NOTE[fieldState];

  return (
    <main ref={stageRef} className={`stage ${filmMode ? 'film-mode' : 'experience-mode'}`}>
      <h1 className="sr-only">Too Expensive to Collapse</h1>
      <video ref={videoRef} className="camera-feed" muted playsInline />
      <canvas ref={webglRef} className="wave-canvas" aria-hidden="true" />
      <canvas ref={trackingRef} className="tracking-canvas" aria-hidden="true" />
      <output className="sr-only" aria-live="polite">{trackingLabel}</output>

      {cameraState !== 'live' && (
        <section className={`camera-gate ${cameraState === 'error' ? 'has-error' : ''}`} aria-label={trackingLabel}>
          <button
            type="button"
            className={cameraState === 'starting' ? 'is-loading' : ''}
            onClick={() => void startCamera()}
            disabled={cameraState === 'starting'}
            aria-label={cameraState === 'error' ? 'Try camera again' : 'Enable camera'}
          >
            <Video size={19} strokeWidth={1.35} />
          </button>
          <p>{cameraState === 'starting' ? 'Starting camera…' : cameraState === 'error' ? 'Camera unavailable. Check camera permission and try again.' : 'Enable camera. Open your palms, then bring them together.'}</p>
          <span>Camera frames stay in this browser. F: film mode · I: the science</span>
        </section>
      )}

      {!filmMode && cameraState === 'live' && cameraDevices.length > 0 && (
        <label className="camera-picker">
          <Camera size={15} strokeWidth={1.35} aria-hidden="true" />
          <span>Camera</span>
          <select
            value={activeCameraId}
            onChange={(event) => void startCamera(event.target.value)}
            disabled={switchingCamera}
            aria-label="Choose camera"
          >
            {cameraDevices.map((device, index) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Camera ${index + 1}`}
              </option>
            ))}
          </select>
        </label>
      )}

      {/* Film mode is the effect and the camera, nothing else. */}
      {!filmMode && (
        <div className="experience-ui">
          <header className="scene-header">
            <div>
              <p className="eyebrow">Conceptual · illustrative model</p>
              <h2>Too Expensive<br />to Collapse</h2>
            </div>
            <div className="system-status">
              <span className={`status-light ${hands > 0 ? 'is-live' : ''}`} />
              {STATE_COPY[fieldState]}
            </div>
          </header>

          <aside className="state-note" aria-live="polite">
            <p className="micro-label">{note.title}</p>
            <p>{note.body}</p>
          </aside>

          <section className="beat" aria-live="polite">
            <p className="beat-index">{beat + 1} / {BEATS.length}</p>
            <h3>{current.title}</h3>
            <p className="beat-body">{current.body}</p>
            <div className="beat-nav">
              <button type="button" onClick={() => setBeat((c) => Math.max(c - 1, 0))} disabled={beat === 0}>Back</button>
              <button type="button" onClick={() => setBeat((c) => Math.min(c + 1, BEATS.length - 1))} disabled={beat === BEATS.length - 1}>Next</button>
            </div>
          </section>

          <footer className="instrument-panel">
            <section className="instruction">
              <p className="micro-label">Gesture</p>
              <p>Bring your palms together <span>or pinch thumb to index</span></p>
            </section>

            <section className="readings" aria-label="Field state">
              <div className="reading">
                <span className="reading-label">Confinement</span>
                <strong className="reading-value">{confinementWord}</strong>
                <div className="reading-track"><i style={{ transform: `scaleX(${idle ? 0 : confinement})` }} /></div>
                <span className="reading-note">position spread Δx narrowing</span>
              </div>
              <div className="reading">
                <span className="reading-label">Kinetic energy cost</span>
                <strong className="reading-value">{costWord}</strong>
                <div className="reading-track"><i className="warm" style={{ transform: `scaleX(${idle ? 0 : cost})` }} /></div>
                <span className="reading-note">Fixed mode: Eₙ ∝ 1/L²</span>
              </div>
              <p className="readings-caveat">
                Illustrative quantum state. Light and color are visual cues, not measured energy or emitted photons.
              </p>
            </section>

            <button type="button" className="mode-button" onClick={showFilmMode}>F&nbsp;&nbsp; Film mode</button>
            <button type="button" className="fullscreen-button" onClick={toggleFullscreen} aria-label="Toggle fullscreen">
              <Expand size={18} strokeWidth={1.25} />
            </button>
          </footer>
        </div>
      )}

      <button
        type="button"
        className="info-button"
        onClick={() => setShowInfo((current) => !current)}
        aria-expanded={showInfo}
        aria-label="What am I looking at?"
      >
        <Info size={16} strokeWidth={1.4} />
      </button>

      {showInfo && (
        <section className="info-panel" aria-label="About this visualisation">
          <button type="button" className="info-close" onClick={() => setShowInfo(false)} aria-label="Close">
            <X size={16} strokeWidth={1.5} />
          </button>

          <h2>Why matter has a size</h2>
          <p>
            Why doesn’t ordinary matter collapse even though electrons are attracted to nuclei?
            Localizing a quantum state raises its kinetic energy. In many-electron matter, the
            Pauli exclusion principle also plays an essential role.
          </p>

          <ol className="info-beats">
            {BEATS.map((entry) => (
              <li key={entry.title}>
                <strong>{entry.title}</strong>
                <span>{entry.body}</span>
              </li>
            ))}
          </ol>

          <h3>What you are looking at</h3>
          <p>
            Your palms stand in for the walls of a one-dimensional infinite square well.
            The hero trace illustrates the real part of a superposition of standing modes.
            Grain brightness follows its time-averaged longitudinal probability density;
            the depth layers make that idea visible in the room. The grains are not individual electrons.
            An electron is treated as pointlike; its quantum state can be spatially extended.
          </p>
          <p>
            Close your hands and the span narrows, the structure gets finer, and the field grows
            more intense. Seal your palms and there is no longer a bridge to see — only the light
            that appears to leak between your fingers. The knot and shimmer are cinematic metaphors.
          </p>

          <h3>Honest limits</h3>
          <ul>
            <li>
              This is a pedagogical analogy for confinement. It does not solve the hydrogen
              Hamiltonian, electron interactions, radiation, or the dynamics of moving boundaries.
              Compression here is spatial confinement, not measurement-induced wavefunction collapse.
            </li>
            <li>
              Eₙ = n²π²ħ²/(2mL²) is exact for the ideal infinite box at fixed n.
              E(L) ≈ A/L² − B/L, with A and B positive, is a scaling argument for a
              hydrogen-like trial state, not the energy calculated by this visual.
              Hand distance has no calibrated atomic units. Color, brightness and animation time are artistic.
            </li>
            <li>
              Hand and face landmarks provide approximate depth and silhouettes. Fingers can
              block individual strands, but hair, clothing, untracked hands and fast motion are
              not fully reconstructed. This is not calibrated room depth or person segmentation.
            </li>
          </ul>

          <h3>Further reading</h3>
          <ul className="info-links">
            <li>
              <a href="https://openstax.org/books/university-physics-volume-3/pages/7-4-the-quantum-particle-in-a-box" target="_blank" rel="noreferrer">
                OpenStax, University Physics Vol. 3 — The Quantum Particle in a Box
              </a>
            </li>
            <li>
              <a href="https://ocw.mit.edu/courses/5-61-physical-chemistry-fall-2017/resources/mit5_61f17_lec20/" target="_blank" rel="noreferrer">
                MIT OpenCourseWare — Hydrogen Atom I
              </a>
            </li>
            <li>
              <a href="https://arxiv.org/abs/math-ph/0401004" target="_blank" rel="noreferrer">
                Elliott H. Lieb — Quantum Mechanics and the Stability of Matter
              </a>
            </li>
          </ul>
        </section>
      )}
    </main>
  );
}
