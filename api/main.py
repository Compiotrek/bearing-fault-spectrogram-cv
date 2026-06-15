"""FastAPI backend for the Interactive Spectrogram Fault Explorer."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (
    GradCamResponse,
    HealthResponse,
    PredictionResponse,
    SampleDetail,
    SampleMetadata,
    SpectrogramResponse,
    VariantComparisonResponse,
    VariantPrediction,
)
from src.app_utils import (
    PredictionResult,
    load_app_manifest,
    load_checkpoint_model,
    load_spectrogram,
    normalize_spectrogram,
    predict_spectrogram,
    spectrogram_to_tensor,
)
from src.dataset import ProcessedSample
from src.gradcam import generate_gradcam, overlay_heatmap_on_spectrogram

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "resnet18_clean" / "best_model.pt"


def _sample_metadata(sample: ProcessedSample) -> SampleMetadata:
    return SampleMetadata(
        sample_id=sample.sample_id,
        label=sample.label,
        split=sample.split,
        variant=sample.variant,
        load=sample.load,
        window_start=sample.window_start,
        sample_rate=sample.sample_rate,
    )


def _sample_detail(sample: ProcessedSample) -> SampleDetail:
    return SampleDetail(
        **_sample_metadata(sample).model_dump(),
        recording_path=str(sample.recording_path),
        spectrogram_path=str(sample.spectrogram_path),
        signal_key=sample.signal_key,
    )


def _variant_order(variant: str) -> tuple[int, int, str]:
    if variant == "clean":
        return (0, 0, variant)
    family = 1 if variant.startswith("noisy_") else 2
    try:
        snr = int(variant.rsplit("_", 1)[1].removesuffix("db"))
    except (IndexError, ValueError):
        snr = -999
    return (family, -snr, variant)


class DemoService:
    """Lazy, cached access to demo artifacts and model inference."""

    def __init__(
        self,
        manifest_path: str | Path,
        checkpoint_path: str | Path,
        project_root: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.project_root = Path(project_root)
        self._samples: list[ProcessedSample] | None = None
        self._sample_by_id: dict[str, ProcessedSample] = {}
        self._model = None
        self._class_mapping: dict[str, int] | None = None
        self._device = None
        self._prediction_cache: dict[str, PredictionResult] = {}
        self._model_lock = Lock()

    def samples(self) -> list[ProcessedSample]:
        if self._samples is None:
            try:
                samples = load_app_manifest(
                    self.manifest_path,
                    project_root=self.project_root,
                )
            except (FileNotFoundError, OSError, ValueError) as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"manifest unavailable: {error}",
                ) from error
            self._samples = samples
            self._sample_by_id = {sample.sample_id: sample for sample in samples}
        return self._samples

    def sample(self, sample_id: str) -> ProcessedSample:
        self.samples()
        try:
            return self._sample_by_id[sample_id]
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=f"unknown sample_id: {sample_id}",
            ) from error

    def spectrogram(self, sample: ProcessedSample) -> np.ndarray:
        try:
            return load_spectrogram(sample.spectrogram_path)
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail=f"spectrogram unavailable: {error}",
            ) from error

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is None:
                try:
                    model, mapping, device = load_checkpoint_model(
                        self.checkpoint_path,
                        device="auto",
                    )
                except (
                    FileNotFoundError,
                    OSError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise HTTPException(
                        status_code=503,
                        detail=f"checkpoint unavailable: {error}",
                    ) from error
                self._model = model
                self._class_mapping = mapping
                self._device = device

    def predict(self, sample: ProcessedSample) -> PredictionResult:
        cached = self._prediction_cache.get(sample.sample_id)
        if cached is not None:
            return cached
        self._ensure_model()
        spectrogram = self.spectrogram(sample)
        with self._model_lock:
            result = predict_spectrogram(
                self._model,
                spectrogram,
                self._class_mapping,
                self._device,
            )
        self._prediction_cache[sample.sample_id] = result
        return result

    def gradcam_analysis(
        self,
        sample: ProcessedSample,
    ) -> tuple[bytes, np.ndarray]:
        self._ensure_model()
        spectrogram = self.spectrogram(sample)
        input_tensor = spectrogram_to_tensor(spectrogram, self._device)
        prediction = self.predict(sample)
        with self._model_lock:
            heatmap = generate_gradcam(
                self._model,
                input_tensor,
                target_class=prediction.predicted_class_id,
            )
        overlay = overlay_heatmap_on_spectrogram(spectrogram, heatmap)
        image = Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB")
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), heatmap

    def matching_variants(
        self,
        sample: ProcessedSample,
    ) -> list[ProcessedSample]:
        matches = [
            candidate
            for candidate in self.samples()
            if candidate.recording_path == sample.recording_path
            and candidate.window_start == sample.window_start
            and candidate.label == sample.label
            and candidate.load == sample.load
        ]
        return sorted(matches, key=lambda item: _variant_order(item.variant))


def create_app(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    project_root: str | Path = PROJECT_ROOT,
) -> FastAPI:
    """Create a configured demo API application."""
    application = FastAPI(
        title="Interactive Spectrogram Fault Explorer API",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    service = DemoService(manifest_path, checkpoint_path, project_root)
    application.state.demo_service = service

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            manifest_ready=service.manifest_path.is_file(),
            checkpoint_ready=service.checkpoint_path.is_file(),
        )

    @application.get("/samples", response_model=list[SampleMetadata])
    def samples(
        split: str | None = Query(default=None),
        variant: str | None = Query(default=None),
        label: str | None = Query(default=None),
    ) -> list[SampleMetadata]:
        selected = [
            sample
            for sample in service.samples()
            if (split is None or sample.split == split)
            and (variant is None or sample.variant == variant)
            and (label is None or sample.label == label)
        ]
        return [_sample_metadata(sample) for sample in selected]

    @application.get("/samples/{sample_id}", response_model=SampleDetail)
    def sample_detail(sample_id: str) -> SampleDetail:
        return _sample_detail(service.sample(sample_id))

    @application.get(
        "/spectrogram/{sample_id}",
        response_model=SpectrogramResponse,
    )
    def spectrogram(sample_id: str) -> SpectrogramResponse:
        sample = service.sample(sample_id)
        values = normalize_spectrogram(service.spectrogram(sample))
        return SpectrogramResponse(
            sample_id=sample_id,
            height=int(values.shape[0]),
            width=int(values.shape[1]),
            values=values.tolist(),
        )

    @application.get("/predict/{sample_id}", response_model=PredictionResponse)
    def predict(sample_id: str) -> PredictionResponse:
        sample = service.sample(sample_id)
        result = service.predict(sample)
        return PredictionResponse(
            sample_id=sample_id,
            predicted_class=result.predicted_class,
            true_class=sample.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
        )

    @application.get("/gradcam/{sample_id}", response_model=GradCamResponse)
    def gradcam(sample_id: str) -> GradCamResponse:
        sample = service.sample(sample_id)
        png, heatmap = service.gradcam_analysis(sample)
        encoded = base64.b64encode(png).decode("ascii")
        peak_frequency, peak_time = np.unravel_index(
            int(np.argmax(heatmap)),
            heatmap.shape,
        )
        threshold = float(np.quantile(heatmap, 0.8))
        return GradCamResponse(
            sample_id=sample_id,
            media_type="image/png",
            image_base64=encoded,
            height=int(heatmap.shape[0]),
            width=int(heatmap.shape[1]),
            values=heatmap.tolist(),
            time_profile=heatmap.mean(axis=0).tolist(),
            frequency_profile=heatmap.mean(axis=1).tolist(),
            peak_time_index=int(peak_time),
            peak_frequency_index=int(peak_frequency),
            focus_fraction=float(np.mean(heatmap >= threshold)),
        )

    @application.get(
        "/variants/{sample_id}",
        response_model=VariantComparisonResponse,
    )
    def variants(sample_id: str) -> VariantComparisonResponse:
        sample = service.sample(sample_id)
        results = []
        for variant_sample in service.matching_variants(sample):
            prediction = service.predict(variant_sample)
            results.append(
                VariantPrediction(
                    sample_id=variant_sample.sample_id,
                    variant=variant_sample.variant,
                    predicted_class=prediction.predicted_class,
                    true_class=variant_sample.label,
                    confidence=prediction.confidence,
                    probabilities=prediction.probabilities,
                )
            )
        return VariantComparisonResponse(
            source_sample_id=sample_id,
            variants=results,
        )

    return application


app = create_app()
