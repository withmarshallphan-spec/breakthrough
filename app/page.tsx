'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { BookOpen, Bug, Clapperboard, Eye, EyeOff, Maximize, Minimize, SwitchCamera, Video } from 'lucide-react';
import type { RenderFeature, WaveDiagnosticStage, WaveEngine } from '@/lib/wave-engine';
import type { FieldState, HandTracker, TrackerSubsystem } from '@/lib/hand-tracker';
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

type CameraCapabilities = MediaTrackCapabilities & {
  exposureMode?: string[];
  focusMode?: string[];
  whiteBalanceMode?: string[];
};

type CameraConstraints = MediaTrackConstraints & {
  exposureMode?: string;
  focusMode?: string;
  whiteBalanceMode?: string;
};

// iPadOS identifies itself as macOS in desktop-mode Safari, so touch support
// is part of the platform readout in the diagnostics panel.
const APPLE_TABLET = typeof navigator !== 'undefined' && (
  /iPad/.test(navigator.userAgent)
  || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
);
// Native video is the source of truth on every device. The GPU-only camera
// composite remains a diagnostic stage, never the default presentation path.
const GPU_SAFE_MODE = false;

declare const __BUILD_ID__: string;

const DIAGNOSTIC_STAGES: Array<{ id: WaveDiagnosticStage; label: string }> = [
  { id: 'raw', label: '1 Raw video' },
  { id: 'transparent', label: '2 Transparent canvas' },
  { id: 'tracking', label: '3 Hand overlay' },
  { id: 'wave', label: '4 Wave' },
  { id: 'segmentation', label: '5 Segmentation' },
  { id: 'depth', label: '6 Depth / refraction' },
  { id: 'composite', label: '7 Final composite' },
];

const RENDER_FEATURES: Array<{ id: RenderFeature; label: string }> = [
  { id: 'relighting', label: 'Relighting' },
  { id: 'depth', label: 'Depth render' },
  { id: 'refraction', label: 'Refraction' },
  { id: 'segmentation', label: 'Segmentation render' },
  { id: 'particles', label: 'Particles / volume' },
  { id: 'finalComposite', label: 'Final composite' },
];

const TRACKER_FEATURES: Array<{ id: TrackerSubsystem; label: string }> = [
  { id: 'hands', label: 'Hand tracking' },
  { id: 'face', label: 'Face tracking' },
  { id: 'segmentation', label: 'Segmentation inference' },
  { id: 'depth', label: 'Depth inference' },
];

type SubsystemMetricRow = {
  name: string;
  fps: number;
  lastDuration: number;
  updates: number;
  skipped: number;
  writesTexture: boolean;
  clearsTarget: boolean;
  preservesPrevious: boolean;
};

type RuntimeDiagnostics = Record<string, string>;

async function waitForVideoFrame(video: HTMLVideoElement) {
  if (video.videoWidth < 1 || video.readyState < 2) {
    await new Promise<void>((resolve) => {
      const finish = () => {
        video.removeEventListener('loadeddata', finish);
        video.removeEventListener('playing', finish);
        window.clearTimeout(timeout);
        resolve();
      };
      const timeout = window.setTimeout(finish, 900);
      video.addEventListener('loadeddata', finish, { once: true });
      video.addEventListener('playing', finish, { once: true });
    });
  }

  if (video.videoWidth > 0 && video.readyState >= 2 && 'requestVideoFrameCallback' in video) {
    await new Promise<void>((resolve) => {
      const timeout = window.setTimeout(resolve, 350);
      video.requestVideoFrameCallback(() => {
        window.clearTimeout(timeout);
        resolve();
      });
    });
  }
}

function nativeVideoHasPixels(video: HTMLVideoElement) {
  if (video.videoWidth < 1 || video.videoHeight < 1 || video.readyState < 2) return false;
  const probe = document.createElement('canvas');
  probe.width = 8;
  probe.height = 8;
  const context = probe.getContext('2d', { willReadFrequently: true });
  if (!context) return false;
  try {
    context.drawImage(video, 0, 0, probe.width, probe.height);
    const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
    let brightPixels = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (Math.max(pixels[index], pixels[index + 1], pixels[index + 2]) > 10) brightPixels += 1;
    }
    return brightPixels >= 3;
  } catch {
    return false;
  }
}

async function cameraPipelineIsHealthy(video: HTMLVideoElement, engine: WaveEngine | null) {
  await waitForVideoFrame(video);
  // Check the decoded HTML frame first, then the exact WebGL VideoTexture path.
  // Safari can satisfy the first check while producing a black GPU texture.
  return video.videoWidth > 0
    && video.readyState >= 2
    && nativeVideoHasPixels(video)
    && (engine ? engine.hasCameraTexturePixels() : true);
}

async function stabiliseCamera(track: MediaStreamTrack) {
  const capabilities = track.getCapabilities?.() as CameraCapabilities | undefined;
  if (!capabilities) return;

  const continuous = (modes?: string[]) => modes?.includes('continuous') ? 'continuous' : undefined;
  const constraints: CameraConstraints = {};
  const exposureMode = continuous(capabilities.exposureMode);
  const focusMode = continuous(capabilities.focusMode);
  const whiteBalanceMode = continuous(capabilities.whiteBalanceMode);

  if (exposureMode) constraints.exposureMode = exposureMode;
  if (focusMode) constraints.focusMode = focusMode;
  if (whiteBalanceMode) constraints.whiteBalanceMode = whiteBalanceMode;

  // Unsupported camera controls are deliberately omitted. Browsers expose a
  // different subset here, and an optional control must never block the feed.
  if (Object.keys(constraints).length) {
    try {
      await track.applyConstraints(constraints);
    } catch {
      // The stream is already usable; retain it when a camera rejects tuning.
    }
  }
}

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
  const [rendererAvailable, setRendererAvailable] = useState(true);
  const gpuSafeModeRef = useRef(GPU_SAFE_MODE);
  const [gpuSafeMode, setGpuSafeMode] = useState(GPU_SAFE_MODE);
  const diagnosticStageRef = useRef<WaveDiagnosticStage>('wave');
  const [diagnosticStage, setDiagnosticStage] = useState<WaveDiagnosticStage>('wave');
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics>({});
  const renderFeaturesRef = useRef<Record<RenderFeature, boolean>>({
    relighting: true,
    depth: true,
    refraction: true,
    segmentation: true,
    particles: true,
    finalComposite: true,
  });
  const [renderFeatures, setRenderFeatures] = useState(renderFeaturesRef.current);
  const trackerFeaturesRef = useRef<Record<TrackerSubsystem, boolean>>({
    hands: true,
    face: true,
    segmentation: true,
    depth: true,
  });
  const [trackerFeatures, setTrackerFeatures] = useState(trackerFeaturesRef.current);
  const [subsystemMetrics, setSubsystemMetrics] = useState<SubsystemMetricRow[]>([]);

  function enableGpuSafeMode() {
    gpuSafeModeRef.current = true;
    setGpuSafeMode(true);
    // This is intentionally independent of renderer availability. A healthy
    // WebGL canvas can still have an unusable camera texture on iOS.
    waveRef.current?.setCameraCompositing(false);
    handTrackerRef.current?.setSafeMode(true);
  }

  function setRenderStage(stage: WaveDiagnosticStage) {
    diagnosticStageRef.current = stage;
    setDiagnosticStage(stage);
    // Sampling the video through Three is an opt-in diagnostic. The normal
    // experience always leaves the browser's video layer visible underneath.
    waveRef.current?.setCameraCompositing(
      stage === 'composite' && renderFeaturesRef.current.finalComposite && !gpuSafeModeRef.current,
    );
    waveRef.current?.setDiagnosticStage(stage);
    handTrackerRef.current?.setGuide(!filmMode && DIAGNOSTIC_STAGES.findIndex(({ id }) => id === stage) >= 2);
  }

  function toggleRenderFeature(feature: RenderFeature) {
    const next = !renderFeaturesRef.current[feature];
    renderFeaturesRef.current = { ...renderFeaturesRef.current, [feature]: next };
    setRenderFeatures(renderFeaturesRef.current);
    waveRef.current?.setRenderFeature(feature, next);
    if (feature === 'finalComposite') {
      waveRef.current?.setCameraCompositing(
        next && diagnosticStageRef.current === 'composite' && !gpuSafeModeRef.current,
      );
    }
  }

  function toggleTrackerFeature(feature: TrackerSubsystem) {
    const next = !trackerFeaturesRef.current[feature];
    trackerFeaturesRef.current = { ...trackerFeaturesRef.current, [feature]: next };
    setTrackerFeatures(trackerFeaturesRef.current);
    handTrackerRef.current?.setSubsystem(feature, next);
  }

  function toggleGpuSafeMode() {
    if (!gpuSafeModeRef.current) {
      enableGpuSafeMode();
      return;
    }
    gpuSafeModeRef.current = false;
    setGpuSafeMode(false);
    waveRef.current?.setCameraCompositing(false);
    // Rebuild optional stages from the live user gesture. Safe mode tears them
    // down intentionally, so turning it off needs a fresh tracker instance.
    if (streamRef.current) void startCamera(activeCameraId || undefined);
  }

  useEffect(() => {
    waveRef.current?.setDiagnosticsEnabled(diagnosticsOpen);
  }, [diagnosticsOpen]);

  useEffect(() => {
    let mounted = true;
    const stopWatching = quality.subscribe((next) => {
      setProfile(next);
      waveRef.current?.setProfile(next);
    });

    const canvas = webglRef.current;
    const fallBackToVideo = () => {
      try {
        waveRef.current?.destroy();
      } catch {
        // A lost graphics context cannot always dispose every GPU resource.
      }
      waveRef.current = null;
      setRendererAvailable(false);
    };
    let contextLostCount = 0;
    const onContextLost = (event: Event) => {
      event.preventDefault();
      contextLostCount += 1;
      fallBackToVideo();
    };
    canvas?.addEventListener('webglcontextlost', onContextLost);

    void import('@/lib/wave-engine').then(({ createWaveEngine }) => {
      if (!mounted || !canvas || !videoRef.current) return;
      try {
        waveRef.current = createWaveEngine(
          canvas,
          videoRef.current,
          quality.profile,
          // The renderer measures; the controller decides. A sustained shortfall
          // drops a tier, and the tier drop reaches the tracker through the
          // same subscription, so the depth model and the segmenter shut down
          // without the pipeline being rebuilt.
          (fps) => quality.observe(fps),
          { cameraCompositing: false },
        );
        waveRef.current.setDiagnosticsEnabled(diagnosticsOpen);
        waveRef.current.setDiagnosticStage(diagnosticStageRef.current);
        if (videoRef.current.srcObject && diagnosticStageRef.current === 'composite') {
          void cameraPipelineIsHealthy(videoRef.current, waveRef.current).then((healthy) => {
            if (mounted && !healthy) enableGpuSafeMode();
          });
        }
      } catch {
        fallBackToVideo();
      }
    });

    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.target instanceof HTMLSelectElement || event.target instanceof HTMLInputElement) return;
      const key = event.key.toLowerCase();
      if (key === 'i') return toggleGpuSafeMode();
      if (key === 'x') return setDiagnosticsOpen(current => !current);
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
    const blackProbe = document.createElement('canvas');
    blackProbe.width = 12;
    blackProbe.height = 12;
    const blackProbeContext = blackProbe.getContext('2d', { willReadFrequently: true });
    let lastProbedVideoTime = -1;
    let sawLitCameraFrame = false;
    let blackFlashCount = 0;
    let lastBlackFlashAt = 0;
    // This measures the decoded camera separately from the WebGL canvas. A
    // black event here points at camera/ML contention; a stable camera with a
    // black compositor points at a render target or final pass instead.
    const blackWatchdog = window.setInterval(() => {
      const video = videoRef.current;
      if (!video || !blackProbeContext || video.readyState < 2 || video.currentTime === lastProbedVideoTime) return;
      lastProbedVideoTime = video.currentTime;
      try {
        blackProbeContext.drawImage(video, 0, 0, blackProbe.width, blackProbe.height);
        const pixels = blackProbeContext.getImageData(0, 0, blackProbe.width, blackProbe.height).data;
        let brightPixels = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          if (Math.max(pixels[index], pixels[index + 1], pixels[index + 2]) > 10) brightPixels += 1;
        }
        if (brightPixels >= 6) sawLitCameraFrame = true;
        else if (sawLitCameraFrame && brightPixels < 2) {
          blackFlashCount += 1;
          lastBlackFlashAt = performance.now();
        }
      } catch {
        // A protected or transient video frame is not a black-frame signal.
      }
    }, 90);
    const diagnosticWatchdog = window.setInterval(() => {
      const video = videoRef.current;
      const track = streamRef.current?.getVideoTracks()[0];
      const wave = waveRef.current?.getDiagnostics();
      const tracker = handTrackerRef.current?.getDiagnostics();
      const canvas = webglRef.current;
      const settings = track?.getSettings();
      const userAgent = navigator.userAgent;
      const browser = /CriOS/.test(userAgent) ? 'Chrome iOS' : /FxiOS/.test(userAgent) ? 'Firefox iOS'
        : /Safari/.test(userAgent) ? 'Safari' : /Chrome/.test(userAgent) ? 'Chrome' : 'Unknown';
      const timestamp = (value: number) => value ? `${value.toFixed(0)} ms` : 'none';
      const trackerRows: SubsystemMetricRow[] = tracker
        ? (Object.entries(tracker.subsystems) as Array<[string, SubsystemMetricRow]>).map(([name, metric]) => ({ name, ...metric }))
        : [];
      const renderRows: SubsystemMetricRow[] = wave
        ? (Object.entries(wave.subsystems) as Array<[string, SubsystemMetricRow]>).map(([name, metric]) => ({ name, ...metric }))
        : [];
      setSubsystemMetrics([...trackerRows, ...renderRows]);
      setDiagnostics({
        build: __BUILD_ID__,
        platform: `${APPLE_TABLET ? 'Apple touch' : navigator.platform} / ${browser}`,
        'video.readyState': String(video?.readyState ?? 0),
        'video.videoWidth': String(video?.videoWidth ?? 0),
        'video.videoHeight': String(video?.videoHeight ?? 0),
        'video.paused': String(video?.paused ?? true),
        'video.ended': String(video?.ended ?? false),
        'track.readyState': track?.readyState ?? 'none',
        'track.muted': String(track?.muted ?? false),
        'camera resolution': settings?.width && settings.height ? `${settings.width} x ${settings.height}` : 'unknown',
        dpr: String(window.devicePixelRatio),
        'canvas CSS': canvas ? `${canvas.clientWidth} x ${canvas.clientHeight}` : 'none',
        'canvas backing': canvas ? `${canvas.width} x ${canvas.height}` : 'none',
        WebGL2: wave?.webgl2 ? 'yes' : 'no',
        WebGPU: (navigator as Navigator & { gpu?: unknown }).gpu ? 'available' : 'no',
        segmentation: tracker?.segmentation ? `active (${tracker.segmentationDelegate})` : 'off',
        depth: tracker?.depth ? 'active' : 'off',
        'webcam texture': wave?.cameraTextureInitialized ? 'initialized' : 'not initialized',
        'render mode': wave?.cameraCompositing ? 'final composite' : 'native video + transparent overlay',
        stage: wave?.diagnosticStage ?? diagnosticStageRef.current,
        frame: String(wave?.frame ?? 0),
        'render FPS': wave?.renderFps?.toFixed(1) ?? '0',
        'black flashes, video': String(blackFlashCount),
        'last black flash, video': timestamp(lastBlackFlashAt),
        'black flashes, composite': String(wave?.blackFlashCount ?? 0),
        'last black flash, composite': timestamp(wave?.lastBlackFlashAt ?? 0),
        'WebGL context losses': String(contextLostCount),
        'last camera frame': timestamp(wave?.lastCameraFrameAt ?? 0),
        'last segmentation frame': timestamp(wave?.lastSegmentationFrameAt ?? 0),
        'last depth frame': timestamp(wave?.lastDepthFrameAt ?? 0),
        'last relighting frame': timestamp(wave?.lastRelightingFrameAt ?? 0),
        'render targets': wave?.renderTargets ?? 'none',
        'render target resizes': String(wave?.resizeCount ?? 0),
        'last render resize': timestamp(wave?.lastResizeAt ?? 0),
        'shader programs': String(wave?.shaderProgramCount ?? 0),
        'person texture allocations': String(wave?.personTextureAllocations ?? 0),
        'depth texture allocations': String(wave?.depthTextureAllocations ?? 0),
        'texture null this frame': String(wave?.nullTextureThisFrame ?? false),
        'clears without draw': wave?.clearsWithoutDraw ?? 'unknown',
        'canvas element': wave?.canvasIdentity ?? 'none',
      });
    }, 400);

    return () => {
      mounted = false;
      stopWatching();
      window.clearInterval(watchdog);
      window.clearInterval(blackWatchdog);
      window.clearInterval(diagnosticWatchdog);
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('fullscreenchange', onFullscreen);
      canvas?.removeEventListener('webglcontextlost', onContextLost);
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
          // A fixed, modest cadence avoids exposure hunting at 60 fps under
          // indoor lighting and leaves enough time for MediaPipe to track.
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      });

      const [cameraTrack] = nextStream.getVideoTracks();
      if (cameraTrack) await stabiliseCamera(cameraTrack);

      if (videoRef.current) {
        videoRef.current.setAttribute('webkit-playsinline', 'true');
        videoRef.current.srcObject = nextStream;
        await videoRef.current.play();
        handTrackerRef.current?.destroy();
        handTrackerRef.current = null;
        if (diagnosticStageRef.current === 'composite') {
          const pipelineHealthy = await cameraPipelineIsHealthy(videoRef.current, waveRef.current);
          if (!pipelineHealthy) enableGpuSafeMode();
        }
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
        }, quality, {
          safeMode: gpuSafeModeRef.current,
          onAdvancedStageFailure: enableGpuSafeMode,
        });
        tracker.setDebug(false);
        (Object.keys(trackerFeaturesRef.current) as TrackerSubsystem[]).forEach((feature) => {
          tracker.setSubsystem(feature, trackerFeaturesRef.current[feature]);
        });
        tracker.setGuide(!filmMode && DIAGNOSTIC_STAGES.findIndex(({ id }) => id === diagnosticStageRef.current) >= 2);
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
      data-renderer={rendererAvailable ? 'webgl' : 'video'}
      data-diagnostic-stage={diagnosticStage}
      aria-label="Quantum confinement instrument"
    >
      <video ref={videoRef} className="camera-feed" autoPlay muted playsInline />
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
        <button type="button" onClick={() => setDiagnosticsOpen(current => !current)} data-active={diagnosticsOpen} aria-label="Diagnostics" title="Diagnostics · X">
          <Bug {...ICON} />
        </button>
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
      {diagnosticsOpen && (
        <aside className="diagnostics" aria-label="Camera pipeline diagnostics">
          <header>
            <p className="meta">camera diagnostics</p>
            <button type="button" onClick={toggleGpuSafeMode} data-active={gpuSafeMode}>I GPU safe mode</button>
          </header>
          <div className="diagnostic-stages" role="group" aria-label="Render stages">
            {DIAGNOSTIC_STAGES.map(({ id, label }) => (
              <button key={id} type="button" onClick={() => setRenderStage(id)} data-active={diagnosticStage === id}>{label}</button>
            ))}
          </div>
          <div className="diagnostic-features" role="group" aria-label="Render subsystem toggles">
            {RENDER_FEATURES.map(({ id, label }) => (
              <button key={id} type="button" onClick={() => toggleRenderFeature(id)} data-active={renderFeatures[id]}>{label}</button>
            ))}
          </div>
          <div className="diagnostic-features" role="group" aria-label="Tracking subsystem toggles">
            {TRACKER_FEATURES.map(({ id, label }) => (
              <button key={id} type="button" onClick={() => toggleTrackerFeature(id)} data-active={trackerFeatures[id]}>{label}</button>
            ))}
          </div>
          <div className="diagnostic-graph" aria-label="Subsystem timing">
            {subsystemMetrics.map((metric) => (
              <div key={metric.name} title={`updates ${metric.updates}; skipped ${metric.skipped}; writes texture ${metric.writesTexture}; clears target ${metric.clearsTarget}; preserves previous ${metric.preservesPrevious}`}>
                <span>{metric.name}</span>
                <i><b style={{ transform: `scaleX(${Math.min(1, metric.fps / 60)})` }} /></i>
                <strong>{metric.fps.toFixed(0)}</strong>
                <em>{metric.lastDuration.toFixed(1)} ms</em>
                <small>{metric.writesTexture ? 'W' : '-'}{metric.clearsTarget ? 'C' : '-'} s{metric.skipped} {metric.preservesPrevious ? 'hold' : 'drop'}</small>
              </div>
            ))}
          </div>
          <dl>
            {Object.entries(diagnostics).map(([label, value]) => (
              <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
            ))}
          </dl>
        </aside>
      )}
    </main>
  );
}
