import type { PredictionData } from "../api";
import type { ReplayStatus } from "./LiveReplayConsole";

interface ModelReadoutRackProps {
  prediction: PredictionData | null;
  loading: boolean;
  replayStatus: ReplayStatus;
}

export function ModelReadoutRack({
  prediction,
  loading,
  replayStatus,
}: ModelReadoutRackProps) {
  const locked = replayStatus === "WINDOW COMPLETE";
  const armed = replayStatus === "READY";

  return (
    <aside className="readout-rack">
      <header className="readout-rack__title-rail">
        <div>
          <span className="eyebrow">Classification result</span>
          <h2>Bearing diagnosis</h2>
        </div>
        <span className={`status-tag ${locked ? "status-tag--locked" : ""}`}>
          {locked ? "LOCKED" : armed ? "MODEL READY" : "ANALYZING"}
        </span>
      </header>

      {loading || !prediction ? (
        <div className="readout-rack__loading">
          <div className="instrument-state">
            <span className="state-meter" />
            <strong>Loading model readout</strong>
            <small>Evaluating the selected measurement record</small>
          </div>
        </div>
      ) : (
        <>
          <div className="readout-rack__row readout-rack__row--state">
            <span>Bearing state</span>
            <strong>{prediction.predicted_class.replace("_", " ")}</strong>
          </div>

          <div className="readout-rack__row readout-rack__row--gauge">
            <div className="readout-rack__row-heading">
              <span>Confidence gauge</span>
              <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
            </div>
            <div className="probability-ruler probability-ruler--confidence">
              <span
                className="probability-ruler__fill"
                style={{ width: `${prediction.confidence * 100}%` }}
              />
              <span className="probability-ruler__ticks" aria-hidden="true" />
            </div>
            <div className="probability-ruler__scale" aria-hidden="true">
              <span>0</span>
              <span>25</span>
              <span>50</span>
              <span>75</span>
              <span>100</span>
            </div>
          </div>

          <div className="readout-rack__row">
            <span>Reference label</span>
            <strong>{prediction.true_class.replace("_", " ")}</strong>
          </div>

          <div className="readout-rack__distribution">
            <span>Fault class distribution</span>
            {Object.entries(prediction.probabilities).map(([name, value]) => (
              <div className="probability-ruler-row" key={name}>
                <div className="probability-ruler-row__label">
                  <span>{name.replace("_", " ")}</span>
                  <strong>{(value * 100).toFixed(1)}%</strong>
                </div>
                <div className="probability-ruler">
                  <span
                    className="probability-ruler__fill"
                    style={{ width: `${value * 100}%` }}
                  />
                  <span
                    className="probability-ruler__ticks"
                    aria-hidden="true"
                  />
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}
