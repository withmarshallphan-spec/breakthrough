'use client';

import { useEffect, useRef, useState } from 'react';
import { Camera, Expand, Eye, EyeOff, Video } from 'lucide-react';
import type { WaveEngine } from '@/lib/wave-engine';
import type { FieldState, HandTracker } from '@/lib/hand-tracker';
import { normalisedEnergy } from '@/lib/quantum';

const STATE_COPY: Record<FieldState, { action: string; energy: string }> = {
  dormant: { action: 'Show your palms', energy: '' },
  open: { action: 'Broad state', energy: 'Low' },
  compressing: { action: 'Confining', energy: 'Rising' },
  clasped: { action: 'Compressed', energy: 'High' },
  release: { action: 'Releasing', energy: 'Falling' },
};

export default function Home() {
  const labelRef = useRef<HTMLElement>(null);
  const lastTrackingRef = useRef(0);
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
      if (key !== 'f') return;
      setFilmMode((current) => {
        const next = !current;
        handTrackerRef.current?.setDebug(false);
        return next;
      });
    };
    window.addEventListener('keydown', onKey);
    const watchdog = window.setInterval(() => {
      if (lastTrackingRef.current && performance.now() - lastTrackingRef.current > 650) {
        if (labelRef.current) labelRef.current.style.opacity = '0';
        setHands(0);
      }
    }, 250);

    return () => {
      mounted = false;
      window.clearInterval(watchdog);
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
          lastTrackingRef.current = now;
          const label = labelRef.current;
          if (label && stageRef.current) {
            const { clientWidth: width, clientHeight: height } = stageRef.current;
            const anchor = update.left.x > update.right.x ? update.left : update.right;
            const x = Math.max(16, Math.min(width - 186, anchor.x + 32));
            const y = Math.max(16, Math.min(height - 132, anchor.y + 44));
            label.style.transform = `translate3d(${x}px, ${y}px, 0)`;
            label.style.opacity = update.hands > 0 ? '1' : '0';
          }
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

  const note = STATE_COPY[fieldState];
  const energy = fieldState === 'clasped' ? 'High'
    : fieldState === 'release' ? 'Falling'
    : confinement > .72 ? 'High' : note.energy;
  const cost = normalisedEnergy(confinement);

  return (
    <main ref={stageRef} className={`stage ${filmMode ? 'film-mode' : ''}`} aria-label="Quantum state simulator">
      <video ref={videoRef} className="camera-feed" muted playsInline />
      <canvas ref={webglRef} className="wave-canvas" aria-hidden="true" />
      <canvas ref={trackingRef} className="tracking-canvas" aria-hidden="true" />
      <output className="sr-only" aria-live="polite">{trackingLabel}</output>

      {cameraState !== 'live' && (
        <section className="camera-gate">
          <button type="button" onClick={() => void startCamera()} disabled={cameraState === 'starting'}>
            <Video size={18} strokeWidth={1.5} />
            {cameraState === 'starting' ? 'Starting…' : cameraState === 'error' ? 'Retry camera' : 'Enable camera'}
          </button>
          <p>{cameraState === 'error' ? 'Check camera permission.' : 'Open palms. Bring them together.'}</p>
        </section>
      )}

      {cameraState === 'live' && hands === 0 && !filmMode && (
        <p className="gesture-hint">{trackingLabel.includes('unavailable') ? 'Hand tracking unavailable. Reload to retry.' : trackingLabel.includes('Loading') ? 'Finding your hands…' : 'Show your palms, or pinch.'}</p>
      )}

      <aside ref={labelRef} className="hand-label" aria-hidden={filmMode || hands === 0} data-energy={energy.toLowerCase()}>
        <span className="hand-action">{note.action}</span>
        <span className="hand-energy">Energy <strong>{energy}</strong></span>
        <span className="energy-line" aria-hidden="true"><i style={{ transform: `scaleX(${fieldState === 'clasped' ? 1 : cost})` }} /></span>
      </aside>

      <footer className="simulator-footer">
        <span className="simulator-note">Just a simulator.</span>
        <div className="scene-controls">
          {cameraState === 'live' && cameraDevices.length > 1 && (
            <label className="camera-picker">
              <Camera size={16} aria-hidden="true" />
              <select value={activeCameraId} onChange={(event) => void startCamera(event.target.value)} disabled={switchingCamera} aria-label="Choose camera">
                {cameraDevices.map((device) => <option key={device.deviceId} value={device.deviceId}>{device.label || 'Camera'}</option>)}
              </select>
            </label>
          )}
          <button type="button" onClick={() => setFilmMode(current => !current)} aria-label={filmMode ? 'Show state labels' : 'Hide state labels'} title="Toggle labels · F">
            {filmMode ? <EyeOff size={17} /> : <Eye size={17} />}
          </button>
          <button type="button" onClick={toggleFullscreen} aria-label="Toggle fullscreen" title="Fullscreen"><Expand size={17} /></button>
        </div>
      </footer>
    </main>
  );
}
