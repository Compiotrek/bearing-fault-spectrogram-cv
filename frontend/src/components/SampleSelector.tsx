import { useEffect, useMemo, useRef, useState } from "react";
import type { SampleMetadata } from "../api";

interface Props {
  samples: SampleMetadata[];
  selectedId: string | null;
  onSelect: (sample: SampleMetadata) => void;
}

function unique(values: string[]): string[] {
  return [...new Set(values)].sort();
}

function numericUnique(values: number[]): number[] {
  return [...new Set(values)].sort((left, right) => left - right);
}

export function SampleSelector({ samples, selectedId, onSelect }: Props) {
  const [split, setSplit] = useState("test");
  const [variant, setVariant] = useState("clean");
  const [label, setLabel] = useState("all");
  const [load, setLoad] = useState("all");
  const initializedRef = useRef(false);

  useEffect(() => {
    if (initializedRef.current || !selectedId) return;
    const selected = samples.find((sample) => sample.sample_id === selectedId);
    if (!selected) return;
    setSplit(selected.split);
    setVariant(selected.variant);
    setLabel(selected.label);
    setLoad(String(selected.load));
    initializedRef.current = true;
  }, [samples, selectedId]);

  const splitOptions = useMemo(
    () => unique(samples.map((sample) => sample.split)),
    [samples],
  );
  const splitSamples = samples.filter((sample) => sample.split === split);
  const variantOptions = unique(splitSamples.map((sample) => sample.variant));
  const variantSamples = splitSamples.filter(
    (sample) => sample.variant === variant,
  );
  const labelOptions = unique(variantSamples.map((sample) => sample.label));
  const labelSamples = variantSamples.filter(
    (sample) => label === "all" || sample.label === label,
  );
  const loadOptions = numericUnique(labelSamples.map((sample) => sample.load));
  const filtered = labelSamples
    .filter((sample) => load === "all" || sample.load === Number(load))
    .sort((left, right) => {
      const labelOrder = left.label.localeCompare(right.label);
      return labelOrder || left.window_start - right.window_start;
    });

  useEffect(() => {
    if (filtered.length === 0) return;
    if (!filtered.some((sample) => sample.sample_id === selectedId)) {
      const previous = samples.find((sample) => sample.sample_id === selectedId);
      const matchingWindow = previous
        ? filtered.find(
            (sample) =>
              sample.label === previous.label &&
              sample.load === previous.load &&
              sample.window_start === previous.window_start,
          )
        : undefined;
      onSelect(matchingWindow ?? filtered[0]);
    }
  }, [filtered, onSelect, samples, selectedId]);

  const selectedSample =
    filtered.find((sample) => sample.sample_id === selectedId) ?? filtered[0];

  return (
    <section className="record-deck">
      <header className="record-deck__identity">
        <div>
          <span className="eyebrow">Record setup</span>
          <h2>Measurement source</h2>
        </div>
        <span className="record-count">{filtered.length} available windows</span>
      </header>

      <div className="record-deck__controls">
        <label className="control-field">
          <span>Split</span>
          <select
            value={split}
            onChange={(event) => {
              const next = event.target.value;
              const variants = unique(
                samples
                  .filter((sample) => sample.split === next)
                  .map((sample) => sample.variant),
              );
              setSplit(next);
              setVariant(variants.includes("clean") ? "clean" : variants[0]);
              setLabel("all");
              setLoad("all");
            }}
          >
            {splitOptions.map((option) => (
              <option key={option}>{option}</option>
            ))}
          </select>
        </label>

        <label className="control-field">
          <span>Condition</span>
          <select
            value={variant}
            onChange={(event) => {
              setVariant(event.target.value);
            }}
          >
            {variantOptions.map((option) => (
              <option key={option}>{option}</option>
            ))}
          </select>
        </label>

        <label className="control-field">
          <span>Bearing state</span>
          <select
            value={label}
            onChange={(event) => setLabel(event.target.value)}
          >
            <option value="all">All classes</option>
            {labelOptions.map((option) => (
              <option key={option} value={option}>
                {option.replace("_", " ")}
              </option>
            ))}
          </select>
        </label>

        <label className="control-field">
          <span>Motor load</span>
          <select value={load} onChange={(event) => setLoad(event.target.value)}>
            <option value="all">All loads</option>
            {loadOptions.map((option) => (
              <option key={option} value={option}>
                Load {option}
              </option>
            ))}
          </select>
        </label>

        <label className="control-field control-field--record">
          <span>Signal window</span>
          <select
            value={selectedSample?.sample_id ?? ""}
            onChange={(event) => {
              const sample = filtered.find(
                (candidate) => candidate.sample_id === event.target.value,
              );
              if (sample) onSelect(sample);
            }}
          >
            {filtered.map((sample) => (
              <option key={sample.sample_id} value={sample.sample_id}>
                {sample.label.replace("_", " ")} · load {sample.load} · start{" "}
                {sample.window_start.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
      </div>

      {selectedSample && (
        <section className="record-plate">
          <header className="record-plate__code">
            <span>Active record</span>
            <b>DE / {selectedSample.sample_rate / 1000}K</b>
          </header>
          <dl className="record-plate__rows">
            <div>
              <dt>State</dt>
              <dd>{selectedSample.label.replace("_", " ")}</dd>
            </div>
            <div>
              <dt>Start</dt>
              <dd>{selectedSample.window_start.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Load</dt>
              <dd>{selectedSample.load}</dd>
            </div>
          </dl>
        </section>
      )}
    </section>
  );
}
