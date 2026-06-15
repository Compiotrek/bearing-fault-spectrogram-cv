const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface SampleMetadata {
  sample_id: string;
  label: string;
  split: string;
  variant: string;
  load: number;
  window_start: number;
  sample_rate: number;
}

export interface SpectrogramData {
  sample_id: string;
  height: number;
  width: number;
  values: number[][];
}

export interface PredictionData {
  sample_id: string;
  predicted_class: string;
  true_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface GradCamData {
  sample_id: string;
  media_type: string;
  image_base64: string;
  height: number;
  width: number;
  values: number[][];
  time_profile: number[];
  frequency_profile: number[];
  peak_time_index: number;
  peak_frequency_index: number;
  focus_fraction: number;
}

export interface VariantPrediction extends PredictionData {
  variant: string;
}

export interface VariantComparisonData {
  source_sample_id: string;
  variants: VariantPrediction[];
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.detail ?? `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function getSamples(signal?: AbortSignal): Promise<SampleMetadata[]> {
  return request<SampleMetadata[]>("/samples", signal);
}

export function getSpectrogram(
  sampleId: string,
  signal?: AbortSignal,
): Promise<SpectrogramData> {
  return request<SpectrogramData>(
    `/spectrogram/${encodeURIComponent(sampleId)}`,
    signal,
  );
}

export function getPrediction(
  sampleId: string,
  signal?: AbortSignal,
): Promise<PredictionData> {
  return request<PredictionData>(
    `/predict/${encodeURIComponent(sampleId)}`,
    signal,
  );
}

export function getGradCam(
  sampleId: string,
  signal?: AbortSignal,
): Promise<GradCamData> {
  return request<GradCamData>(
    `/gradcam/${encodeURIComponent(sampleId)}`,
    signal,
  );
}

export function getVariants(
  sampleId: string,
  signal?: AbortSignal,
): Promise<VariantComparisonData> {
  return request<VariantComparisonData>(
    `/variants/${encodeURIComponent(sampleId)}`,
    signal,
  );
}
