import type { VariantComparisonData } from "../api";

function conditionCode(variant: string): string {
  if (variant === "clean") return "CLN";
  if (variant.startsWith("denoised")) return "DNS";
  return "NSY";
}

export function VariantComparison({
  comparison,
  loading,
}: {
  comparison: VariantComparisonData | null;
  loading: boolean;
}) {
  return (
    <section className="condition-strip">
      <header className="condition-strip__title-rail">
        <div>
          <span className="eyebrow">Noise bench</span>
          <h2>Condition response</h2>
        </div>
        <span className="method-tag">MATCHED RECORD</span>
      </header>
      {loading || !comparison ? (
        <div className="condition-strip__loading">
          <div className="instrument-state">
            <span className="state-meter" />
            <strong>Comparing signal conditions</strong>
            <small>Holding bearing state, load and window constant</small>
          </div>
        </div>
      ) : (
        <div className="condition-strip__table">
          <div className="condition-strip__columns" aria-hidden="true">
            <span>Type</span>
            <span>Condition / class</span>
            <span>Confidence ruler</span>
            <span>Value</span>
          </div>
          {comparison.variants.map((variant) => (
            <div className="condition-strip__row" key={variant.sample_id}>
              <span className="condition-code">{conditionCode(variant.variant)}</span>
              <div className="condition-strip__identity">
                <strong>{variant.variant.replace("_", " ")}</strong>
                <small>{variant.predicted_class.replace("_", " ")}</small>
              </div>
              <div className="probability-ruler probability-ruler--condition">
                <span
                  className="probability-ruler__fill"
                  style={{ width: `${variant.confidence * 100}%` }}
                />
                <span className="probability-ruler__ticks" aria-hidden="true" />
              </div>
              <b>{(variant.confidence * 100).toFixed(1)}%</b>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
