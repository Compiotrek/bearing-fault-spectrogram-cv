import { useEffect, useMemo, useState } from "react";
import {
  getGradCam,
  getPrediction,
  getSamples,
  getSpectrogram,
  getVariants,
  type GradCamData,
  type PredictionData,
  type SampleMetadata,
  type SpectrogramData,
  type VariantComparisonData,
} from "./api";
import { GradCamViewer } from "./components/GradCamViewer";
import {
  LiveReplayConsole,
  type ReplayStatus,
} from "./components/LiveReplayConsole";
import { ModelReadoutRack } from "./components/ModelReadoutRack";
import { SampleSelector } from "./components/SampleSelector";
import { SpectrogramCanvas } from "./components/SpectrogramCanvas";
import { VariantComparison } from "./components/VariantComparison";

export default function App() {
  const [samples, setSamples] = useState<SampleMetadata[]>([]);
  const [selected, setSelected] = useState<SampleMetadata | null>(null);
  const [spectrogram, setSpectrogram] = useState<SpectrogramData | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [gradcam, setGradcam] = useState<GradCamData | null>(null);
  const [variants, setVariants] = useState<VariantComparisonData | null>(null);
  const [replayStatus, setReplayStatus] =
    useState<ReplayStatus>("READY");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getSamples(controller.signal)
      .then((result) => {
        setSamples(result);
        const initial =
          result.find(
            (sample) =>
              sample.split === "test" &&
              sample.variant === "clean" &&
              sample.label === "ball",
          ) ?? result[0];
        setSelected(initial ?? null);
      })
      .catch((reason: Error) => setError(reason.message));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setError(null);
    setSpectrogram(null);
    setPrediction(null);
    setGradcam(null);
    setVariants(null);
    setReplayStatus("READY");

    getSpectrogram(selected.sample_id, controller.signal)
      .then(setSpectrogram)
      .catch((reason: Error) => setError(reason.message));
    getPrediction(selected.sample_id, controller.signal)
      .then(setPrediction)
      .catch((reason: Error) => setError(reason.message));
    getGradCam(selected.sample_id, controller.signal)
      .then(setGradcam)
      .catch((reason: Error) => setError(reason.message));
    getVariants(selected.sample_id, controller.signal)
      .then(setVariants)
      .catch((reason: Error) => setError(reason.message));
    return () => controller.abort();
  }, [selected]);

  const sampleCaption = useMemo(() => {
    if (!selected) return "Waiting for dataset";
    return `LOAD ${selected.load} · ${selected.variant.toUpperCase()} · START ${selected.window_start}`;
  }, [selected]);

  return (
    <main className="lab-workstation">
      <header className="machine-header">
        <div className="machine-id">
          <span>BF</span>
          <b>04</b>
        </div>
        <div className="app-identity">
          <p>Drive-end vibration bench</p>
          <h1>Noise-Robust Bearing Fault Detection</h1>
          <span>
            CWRU / 12 kHz acquisition / spectral fault inspection
          </span>
        </div>
        <div className="machine-spec" aria-label="Acquisition configuration">
          <div>
            <span>Acquisition</span>
            <strong>12 kHz</strong>
          </div>
          <div>
            <span>Window</span>
            <strong>2048</strong>
          </div>
          <div>
            <span>STFT</span>
            <strong>256 / 128</strong>
          </div>
        </div>
        <div className="bench-state">
          <span>System</span>
          <strong>BENCH READY</strong>
          <small>{sampleCaption}</small>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <strong>DATA PATH FAULT</strong>
          <span>{error}</span>
        </div>
      )}

      <SampleSelector
        samples={samples}
        selectedId={selected?.sample_id ?? null}
        onSelect={setSelected}
      />

      <div className="primary-bench">
        <div className="scope-stack">
          {spectrogram && selected ? (
            <>
              <LiveReplayConsole
                data={spectrogram}
                sampleRate={selected.sample_rate}
                onStatusChange={setReplayStatus}
              />
              <details className="scope-frame scope-frame--compact">
                <summary>Full time-frequency record</summary>
                <div className="scope-frame__viewport scope-frame__viewport--compact">
                  <SpectrogramCanvas data={spectrogram} />
                </div>
                <div className="scope-frame__metadata-rail">
                  <span>STATIC VIEW</span>
                  <span>{spectrogram.width} FRAMES</span>
                  <span>{selected.sample_rate / 1000} KHZ</span>
                </div>
              </details>
            </>
          ) : (
            <section className="scope-frame scope-frame--loading">
              <header className="scope-frame__title-rail">
                <div>
                  <span className="module-number">02 / Signal monitor</span>
                  <h2>Spectrogram stream</h2>
                </div>
                <span className="status-tag">INITIALIZING</span>
              </header>
              <div className="scope-frame__viewport">
                <div className="instrument-state">
                  <span className="state-meter" />
                  <strong>Preparing spectrogram stream</strong>
                  <small>Allocating replay buffer and display channels</small>
                </div>
              </div>
              <div className="scope-frame__metadata-rail">
                <span>BUFFER --</span>
                <span>FRAME 000</span>
                <span>INPUT CH A</span>
              </div>
            </section>
          )}
        </div>

        <ModelReadoutRack
          prediction={prediction}
          loading={!prediction}
          replayStatus={replayStatus}
        />
      </div>

      <div className="analysis-bench">
        <GradCamViewer
          gradcam={gradcam}
          spectrogram={spectrogram}
          loading={!gradcam || !spectrogram}
        />
        <VariantComparison comparison={variants} loading={!variants} />
      </div>
    </main>
  );
}
