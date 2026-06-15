import { useState } from "react";
import type { GradCamData, SpectrogramData } from "../api";
import {
  AttributionCanvas,
  type AttributionMode,
} from "./AttributionCanvas";

const MODES: Array<{ id: AttributionMode; label: string }> = [
  { id: "topology", label: "Contours" },
  { id: "focus", label: "Regions" },
  { id: "overlay", label: "Overlay" },
];

export function GradCamViewer({
  gradcam,
  spectrogram,
  loading,
}: {
  gradcam: GradCamData | null;
  spectrogram: SpectrogramData | null;
  loading: boolean;
}) {
  const [mode, setMode] = useState<AttributionMode>("topology");

  return (
    <section className="evidence-frame">
      <header className="evidence-frame__title-rail">
        <div>
          <span className="eyebrow">Model evidence</span>
          <h2>Time-frequency inspection</h2>
        </div>
        <div className="evidence-frame__controls" aria-label="Attribution view">
          {MODES.map((item) => (
            <button
              className={`console-button console-button--small ${
                mode === item.id ? "is-active" : ""
              }`}
              key={item.id}
              onClick={() => setMode(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </header>

      {loading || !gradcam || !spectrogram ? (
        <div className="evidence-frame__viewport evidence-frame__viewport--loading">
          <div className="instrument-state">
            <span className="state-meter" />
            <strong>Calculating evidence map</strong>
            <small>Resolving discriminative time-frequency regions</small>
          </div>
        </div>
      ) : (
        <>
          <div className="evidence-frame__viewport">
            <span className="evidence-frame__corners" aria-hidden="true" />
            <AttributionCanvas
              gradcam={gradcam}
              spectrogram={spectrogram}
              mode={mode}
            />
          </div>
          <div className="evidence-frame__metadata-rail">
            <div>
              <span>Peak position</span>
              <strong>
                {Math.round(
                  (gradcam.peak_time_index / Math.max(1, gradcam.width - 1)) *
                    100,
                )}
                %
              </strong>
            </div>
            <div>
              <span>Frequency band</span>
              <strong>
                {Math.round(
                  (gradcam.peak_frequency_index /
                    Math.max(1, gradcam.height - 1)) *
                    6000,
                )}{" "}
                Hz
              </strong>
            </div>
            <div>
              <span>Evidence coverage</span>
              <strong>{(gradcam.focus_fraction * 100).toFixed(1)}%</strong>
            </div>
          </div>
        </>
      )}
      <footer className="evidence-frame__caption">
        Contours mark increasing model response. The reticle identifies the
        strongest time-frequency region; the traces summarize evidence across
        time and frequency.
      </footer>
    </section>
  );
}
