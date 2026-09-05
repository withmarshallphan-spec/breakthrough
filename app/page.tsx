'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { BookOpen, Clapperboard, Eye, EyeOff, Maximize, Minimize, SwitchCamera, Video } from 'lucide-react';
import type { WaveEngine } from '@/lib/wave-engine';
import type { FieldState, HandTracker } from '@/lib/hand-tracker';
import { createQualityController, type QualityProfile } from '@/lib/quality';
import {
  energyRatio,
  GROUND_UNCERTAINTY,
  momentumRatio,
  relativeWellWidth,
} from '@/lib/quantum';
import { MODEL_CAVEAT, READOUTS, SECTIONS, STATE_NOTES } from '@/lib/science-copy';

// This route has no server data or request-dependent output. Declaring it
// static lets Vinext emit the HTML required by GitHub Pages.
export const dynamic = 'force-static';

/**
 * Instrument Serif has no U+2080, so a literal subscript zero falls back to
 * another face and arrives looking like the letter o -- `L / L₀` reads as
 * `L / Lo`. Splitting on the character and emitting a real `<sub>` keeps the
 * whole symbol in one typeface and says what it means.
 */
function Symbols({ text }: { text: string }) {
  const parts = text.split('₀');
  return (
    <>
      {parts.map((part, index) => (
        <span key={`${part}-${index}`}>
          {part}
          {index < parts.length - 1 && <sub>0</sub>}
        </span>
      ))}
    </>
  );
}

/** Every icon at the same size and hairline weight, so the set reads as one. */
const ICON = { size: 16, strokeWidth: 1.25 } as const;

/**
 * The uncertainty meter's axis, in units of hbar. It starts at the bound the
 * uncertainty principle sets, so the reader can see the ground state sitting
 * above it -- and see that squeezing never moves it down.
 */
const UNCERTAINTY_FLOOR = .5;
const UNCERTAINTY_CEILING = .75;
const UNCERTAINTY_MARK =
  (GROUND_UNCERTAINTY - UNCERTAINTY_FLOOR) / (UNCERTAINTY_CEILING - UNCERTAINTY_FLOOR);

export default function Home() {
  const calloutRef = useRef<HTMLElement>(null);
  const lastTrackingRef = useRef(0);
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const webglRef = useRef<HTMLCanvasElement>(null);
  const trackingRef = useRef<HTMLCanvasElement>(null);
  const waveRef = useRef<WaveEngine | null>(null);
  const handTrackerRef = useRef<HandTracker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastMeterUpdateRef = useRef(0);
  // Held in state rather than a ref so it is created once without being read
  // during render, and so the initial profile can seed the panel below.
  const [quality] = useState(createQualityController);

  const [cameraState, setCameraState] = useState<'idle' | 'starting' | 'live' | 'error'>('idle');
  const [trackingLabel, setTrackingLabel] = useState('Camera ready');
  const [confinement, setConfinement] = useState(0);
  const [fieldState, setFieldState] = useState<FieldState>('dormant');
  const [hands, setHands] = useState(0);
  const [filmMode, setFilmMode] = useState(false);
  const [interfaceHidden, setInterfaceHidden] = useState(false);
  const [readerOpen, setReaderOpen] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [activeCameraId, setActiveCameraId] = useState('');
  const [switchingCamera, setSwitchingCamera] = useState(false);
  const [profile, setProfile] = useState<QualityProfile>(quality.profile);

  useEffect(() => {
    let mounted = true;
    const stopWatching = quality.subscribe((next) => {
      setProfile(next);
      waveRef.current?.setProfile(next);
    });

    void import('@/lib/wave-engine').then(({ createWaveEngine }) => {
      if (mounted && webglRef.current && videoRef.current) {
        waveRef.current = createWaveEngine(
          webglRef.current,
          videoRef.current,
          quality.profile,
          // The renderer measures; the controller decides. A sustained shortfall
          // drops a tier, and the tier drop reaches the tracker through the
          // same subscription, so the depth model and the segmenter shut down
          // without the pipeline being rebuilt.
          (fps) => quality.observe(fps),
        );
      }
    });

    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.target instanceof HTMLSelectElement || event.target instanceof HTMLInputElement) return;
      const key = event.key.toLowerCase();
      if (key === 'd') return handTrackerRef.current?.setDebug(event.shiftKey);
      if (key === 'escape') return setReaderOpen(false);
      if (key === 'n') return setReaderOpen(current => !current);
      if (key !== 'f') return;
      setFilmMode((current) => {
        const next = !current;
        handTrackerRef.current?.setDebug(false);
        handTrackerRef.current?.setGuide(!next);
        if (next) setReaderOpen(false);
        return next;
      });
    };
    const onFullscreen = () => setFullscreen(Boolean(document.fullscreenElement));

    window.addEventListener('keydown', onKey);
    document.addEventListener('fullscreenchange', onFullscreen);
    const watchdog = window.setInterval(() => {
      // The callout is deliberately not hidden here. Losing tracking should not
      // take the text away -- the tracker holds the block where it was, and the
      // leader fades on its own because there is no hand left to point at.
      if (lastTrackingRef.current && performance.now() - lastTrackingRef.current > 650) {
        setHands(0);
      }
    }, 250);

    return () => {
      mounted = false;
      stopWatching();
      window.clearInterval(watchdog);
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('fullscreenchange', onFullscreen);
      handTrackerRef.current?.destroy();
      waveRef.current?.destroy();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [quality]);

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
          lastTrackingRef.current = now;
          // Positioned imperatively: the tracker delivers this at video rate,
          // and a React render per frame would cost more than the whole
          // compositor. The tracker has already smoothed the position, so this
          // is a straight write with no easing of its own on top.
          const callout = calloutRef.current;
          if (callout) {
            const { x, y, side } = update.callout;
            callout.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
            callout.dataset.side = side;
            // Revealed once, on the first frame that has a position to show,
            // and never taken away again.
            callout.dataset.ready = 'true';
          }
          if (now - lastMeterUpdateRef.current > 110) {
            lastMeterUpdateRef.current = now;
            setConfinement(update.confinement);
            setHands(update.hands);
            setFieldState(update.state);
            setTrackingLabel(update.label);
          }
        }, quality);
        tracker.setDebug(false);
        tracker.setGuide(!filmMode);
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

  function toggleFilm() {
    setFilmMode((current) => {
      const next = !current;
      handTrackerRef.current?.setGuide(!next);
      if (next) setReaderOpen(false);
      return next;
    });
  }

  const note = STATE_NOTES[fieldState];
  /**
   * Levels, not digits. Each relation below is exact for the ideal box the
   * renderer is running, but the dial driving it is a hand-distance estimate,
   * so printing three decimals would claim a precision the input does not have.
   * A meter states the same thing at the precision the gesture actually
   * supports: where the value sits on its own range, and which way it is going.
   */
  const meters = useMemo(() => {
    const held = fieldState === 'clasped' ? 1 : confinement;
    return [
      { id: 'width' as const, level: (relativeWellWidth(held) - .5) / .5, rising: false },
      { id: 'energy' as const, level: (energyRatio(held) - 1) / 3, rising: true },
      { id: 'momentum' as const, level: momentumRatio(held) - 1, rising: true },
    ];
  }, [confinement, fieldState]);

  return (
    <main
      ref={stageRef}
      className={`stage ${filmMode ? 'film-mode' : ''} ${interfaceHidden ? 'interface-hidden' : ''}`}
      aria-label="Quantum confinement instrument"
    >
      <video ref={videoRef} className="camera-feed" muted playsInline />
      <canvas ref={webglRef} className="wave-canvas" aria-hidden="true" />
      <canvas ref={trackingRef} className="tracking-canvas" aria-hidden="true" />
      <output className="sr-only" aria-live="polite">{trackingLabel}</output>

      {cameraState !== 'live' && (
        <section className="camera-gate">
          <h1 className="display">Too Expensive to Collapse</h1>
          <p>
            An interactive on the quantum stability of matter. Hand tracking runs entirely in this
            browser; no video leaves the device.
          </p>
          <button type="button" onClick={() => void startCamera()} disabled={cameraState === 'starting'}>
            <Video {...ICON} />
            {cameraState === 'starting' ? 'Starting…' : cameraState === 'error' ? 'Retry camera' : 'Enable camera'}
          </button>
        </section>
      )}

      {/* The state, its levels, and nothing else -- placed by the tracker, which
          also draws the dot on the palm and the leader that reaches this block,
          so the line always meets the text. */}
      <aside ref={calloutRef} className="callout" aria-hidden={filmMode}>
        <p className="callout-state">{note.title}</p>
        <div className="meters">
          {meters.map(({ id, level, rising }) => (
            <div className="meter" key={id}>
              <span className="meter-head">
                <span className="meter-label">{READOUTS[id].label}</span>
                <span className="meter-symbol"><Symbols text={READOUTS[id].symbol} /></span>
              </span>
              <span className="meter-track">
                <i data-rising={rising} style={{ transform: `scaleX(${Math.max(0, Math.min(1, level))})` }} />
              </span>
            </div>
          ))}
          <div className="meter" data-fixed="true">
            <span className="meter-head">
              <span className="meter-label">{READOUTS.uncertainty.label}</span>
              <span className="meter-symbol"><Symbols text="Δx·Δp ≥ ℏ/2" /></span>
            </span>
            {/* The tick is the bound; the marker is the ground state sitting
                above it, and it does not move however hard the well is
                squeezed. That refusal is the reading. */}
            <span className="meter-track">
              <u />
              <b style={{ left: `${(UNCERTAINTY_MARK * 100).toFixed(1)}%` }} />
            </span>
          </div>
        </div>
      </aside>

      <aside className="reader" data-open={readerOpen} aria-hidden={!readerOpen}>
        <header>
          <p className="meta">The physics</p>
          <h2 className="display" style={{ marginTop: 12 }}>The stability of matter</h2>
          <p>
            Eight notes on why a Coulomb potential with no floor still produces atoms of a definite
            size, and why bulk matter needs an argument that a single atom does not.
          </p>
        </header>
        {SECTIONS.map((section) => (
          <article key={section.id}>
            <div className="article-head">
              <span className="numeral">{section.index}</span>
              <h3 className="section-head">{section.title}</h3>
            </div>
            {section.body.map((paragraph) => <p key={paragraph.slice(0, 24)}>{paragraph}</p>)}
            <a href={section.source.href} target="_blank" rel="noreferrer noopener">{section.source.label}</a>
          </article>
        ))}
        {/* Tier-dependent, so it is only rendered once the camera is live and
            the client has measured the machine. Rendering it during the server
            pass would describe a tier that only the server believes in. */}
        {cameraState === 'live' && (
          <article>
            <div className="article-head">
              <span className="numeral">—</span>
              <h3 className="section-head">Rendering</h3>
            </div>
            <p>
              Relighting runs on landmark geometry: a hand rig, a palm surface built from the
              silhouette, {profile.faceMesh ? 'and an 852-triangle face mesh' : 'and no face mesh on this tier'}
              {profile.denseDepth ? ', with dense monocular depth filling the rest of the room' : ''}. Quality
              tier <span className="numeral">{profile.tier}</span>.
            </p>
          </article>
        )}
      </aside>

      {/* The one standing line of text. It warns, and nothing more. It sits
          after the reader in the DOM because the rule that steps it aside when
          the reader opens is a following-sibling selector. */}
      {cameraState === 'live' && (
        <p className="warning">
          {trackingLabel.includes('unavailable')
            ? 'Hand tracking unavailable. Reload to retry.'
            : trackingLabel.includes('Loading')
              ? 'Finding your hands…'
              : hands === 0
                ? 'Show both palms, or pinch with one hand.'
                : MODEL_CAVEAT}
        </p>
      )}

      <nav className="tools" aria-label="Capture controls">
        {cameraState === 'live' && cameraDevices.length > 1 && (
          <label className="camera-picker" title="Camera source">
            <SwitchCamera {...ICON} aria-hidden="true" />
            <select
              value={activeCameraId}
              onChange={(event) => void startCamera(event.target.value)}
              disabled={switchingCamera}
              aria-label="Choose camera"
            >
              {cameraDevices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>{device.label || 'Camera'}</option>
              ))}
            </select>
          </label>
        )}
        {cameraState === 'live' && !filmMode && (
          <button type="button" onClick={() => setReaderOpen(current => !current)} data-active={readerOpen} aria-label="The physics" title="The physics · N">
            <BookOpen {...ICON} />
          </button>
        )}
        <button type="button" onClick={toggleFilm} data-active={filmMode} aria-label={filmMode ? 'Leave film mode' : 'Film mode'} title="Film mode · F">
          <Clapperboard {...ICON} />
        </button>
        <button type="button" onClick={() => setInterfaceHidden(current => !current)} data-active={interfaceHidden} aria-label={interfaceHidden ? 'Show interface' : 'Hide interface'} title="Interface">
          {interfaceHidden ? <EyeOff {...ICON} /> : <Eye {...ICON} />}
        </button>
        <button type="button" onClick={toggleFullscreen} aria-label="Toggle fullscreen" title="Fullscreen">
          {fullscreen ? <Minimize {...ICON} /> : <Maximize {...ICON} />}
        </button>
      </nav>
    </main>
  );
}
