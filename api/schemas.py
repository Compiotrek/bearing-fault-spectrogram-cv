"""Response schemas for the diagnostic demo API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    manifest_ready: bool
    checkpoint_ready: bool


class SampleMetadata(BaseModel):
    sample_id: str
    label: str
    split: str
    variant: str
    load: int
    window_start: int
    sample_rate: int


class SampleDetail(SampleMetadata):
    recording_path: str
    spectrogram_path: str
    signal_key: str


class SpectrogramResponse(BaseModel):
    sample_id: str
    height: int
    width: int
    values: list[list[float]]


class PredictionResponse(BaseModel):
    sample_id: str
    predicted_class: str
    true_class: str
    confidence: float
    probabilities: dict[str, float]


class GradCamResponse(BaseModel):
    sample_id: str
    media_type: str
    image_base64: str
    height: int
    width: int
    values: list[list[float]]
    time_profile: list[float]
    frequency_profile: list[float]
    peak_time_index: int
    peak_frequency_index: int
    focus_fraction: float


class VariantPrediction(BaseModel):
    sample_id: str
    variant: str
    predicted_class: str
    true_class: str
    confidence: float
    probabilities: dict[str, float]


class VariantComparisonResponse(BaseModel):
    source_sample_id: str
    variants: list[VariantPrediction]
