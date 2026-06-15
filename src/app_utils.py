"""Reusable helpers for the Streamlit spectrogram demo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from torch import nn

from src.dataset import ProcessedSample, load_manifest
from src.model import build_model


@dataclass(frozen=True)
class PredictionResult:
    """Classification output for one spectrogram."""

    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: dict[str, float]


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a supported inference device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


def load_app_manifest(
    path: str | Path,
    project_root: str | Path | None = None,
) -> list[ProcessedSample]:
    """Load a manifest and resolve relative artifact paths."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    resolved_samples: list[ProcessedSample] = []
    for sample in load_manifest(manifest_path):
        spectrogram_path = sample.spectrogram_path
        recording_path = sample.recording_path
        if not spectrogram_path.is_absolute():
            spectrogram_path = root / spectrogram_path
        if not recording_path.is_absolute():
            recording_path = root / recording_path
        resolved_samples.append(
            replace(
                sample,
                spectrogram_path=spectrogram_path,
                recording_path=recording_path,
            )
        )
    if not resolved_samples:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    return resolved_samples


def filter_samples(
    samples: list[ProcessedSample],
    *,
    split: str | None = None,
    variant: str | None = None,
    label: str | None = None,
) -> list[ProcessedSample]:
    """Filter samples for the demo selectors in deterministic order."""
    return sorted(
        (
            sample
            for sample in samples
            if (split is None or sample.split == split)
            and (variant is None or sample.variant == variant)
            and (label is None or sample.label == label)
        ),
        key=lambda sample: sample.sample_id,
    )


def selector_values(
    samples: list[ProcessedSample],
    field: str,
) -> list[str]:
    """Return sorted unique values for a supported sample field."""
    if field not in {"split", "variant", "label"}:
        raise ValueError("field must be 'split', 'variant', or 'label'")
    return sorted({str(getattr(sample, field)) for sample in samples})


def load_spectrogram(path: str | Path) -> np.ndarray:
    """Load and validate one cached spectrogram."""
    spectrogram_path = Path(path)
    if not spectrogram_path.is_file():
        raise FileNotFoundError(f"spectrogram file not found: {spectrogram_path}")
    spectrogram = np.load(spectrogram_path, allow_pickle=False)
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be a two-dimensional array")
    if not np.issubdtype(spectrogram.dtype, np.number):
        raise TypeError("spectrogram must be numeric")
    if not np.all(np.isfinite(spectrogram)):
        raise ValueError("spectrogram must contain only finite values")
    return spectrogram.astype(np.float32, copy=False)


def load_checkpoint_model(
    path: str | Path,
    device: str = "auto",
) -> tuple[nn.Module, dict[str, int], torch.device]:
    """Load a trained model and its class mapping for inference."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    resolved_device = resolve_device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )
    required_keys = {
        "model_state_dict",
        "model_name",
        "num_classes",
        "class_mapping",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing required fields: {missing}")

    class_mapping = checkpoint["class_mapping"]
    if not isinstance(class_mapping, dict):
        raise ValueError("checkpoint class_mapping must be a dictionary")
    normalized_mapping = {
        str(class_name): int(class_id) for class_name, class_id in class_mapping.items()
    }
    num_classes = int(checkpoint["num_classes"])
    if len(normalized_mapping) != num_classes:
        raise ValueError("checkpoint class_mapping does not match num_classes")

    model = build_model(
        str(checkpoint["model_name"]),
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, normalized_mapping, resolved_device


def spectrogram_to_tensor(
    spectrogram: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a validated 2D spectrogram to a model input tensor."""
    values = np.asarray(spectrogram)
    if values.ndim != 2:
        raise ValueError("spectrogram must be a two-dimensional array")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("spectrogram must be numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrogram must contain only finite values")
    return (
        torch.from_numpy(values.astype(np.float32, copy=False))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


def predict_spectrogram(
    model: nn.Module,
    spectrogram: np.ndarray,
    class_mapping: dict[str, int],
    device: torch.device | str = "cpu",
) -> PredictionResult:
    """Run model inference for one spectrogram."""
    if not class_mapping:
        raise ValueError("class_mapping must not be empty")
    class_names = {
        int(class_id): class_name for class_name, class_id in class_mapping.items()
    }
    input_tensor = spectrogram_to_tensor(spectrogram, device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    if probabilities.size != len(class_names):
        raise ValueError("model output does not match class_mapping")

    predicted_class_id = int(np.argmax(probabilities))
    if predicted_class_id not in class_names:
        raise ValueError("predicted class is missing from class_mapping")
    sorted_probabilities = {
        class_names[class_id]: float(probabilities[class_id])
        for class_id in np.argsort(probabilities)[::-1]
    }
    return PredictionResult(
        predicted_class=class_names[predicted_class_id],
        predicted_class_id=predicted_class_id,
        confidence=float(probabilities[predicted_class_id]),
        probabilities=sorted_probabilities,
    )


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """Normalize a finite 2D spectrogram to the interval [0, 1]."""
    values = np.asarray(spectrogram)
    if values.ndim != 2:
        raise ValueError("spectrogram must be a two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrogram must contain only finite values")

    normalized = values.astype(np.float32, copy=True)
    normalized -= float(normalized.min())
    maximum = float(normalized.max())
    if maximum > 0.0:
        normalized /= maximum
    return normalized


def reveal_spectrogram(
    spectrogram: np.ndarray,
    progress: float,
    hidden_value: float | None = None,
) -> np.ndarray:
    """Reveal spectrogram columns from left to right for replay."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")
    values = np.asarray(spectrogram)
    if values.ndim != 2:
        raise ValueError("spectrogram must be a two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrogram must contain only finite values")

    frame = values.astype(np.float32, copy=True)
    revealed_columns = int(np.ceil(frame.shape[1] * progress))
    fill_value = float(frame.min()) if hidden_value is None else float(hidden_value)
    frame[:, revealed_columns:] = fill_value
    return frame


def rolling_spectrogram_window(
    spectrogram: np.ndarray,
    progress: float,
    viewport_columns: int = 96,
) -> np.ndarray:
    """Build a fixed-width history buffer with newest data on the right."""
    normalized = normalize_spectrogram(spectrogram)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")
    if viewport_columns <= 1:
        raise ValueError("viewport_columns must be greater than 1")

    source_width = normalized.shape[1]
    current_column = min(
        source_width - 1,
        int(round(progress * (source_width - 1))),
    )
    history_start = max(0, current_column - viewport_columns + 1)
    history = normalized[:, history_start : current_column + 1]
    window = np.zeros(
        (normalized.shape[0], viewport_columns),
        dtype=np.float32,
    )
    window[:, -history.shape[1] :] = history
    return window


def build_live_replay_frame(
    spectrogram: np.ndarray,
    progress: float,
    viewport_columns: int = 96,
) -> np.ndarray:
    """Render a rolling RGB buffer whose newest data enters on the right."""
    window = rolling_spectrogram_window(
        spectrogram,
        progress,
        viewport_columns=viewport_columns,
    )

    height, width = window.shape
    frame = colormaps["magma"](window)[..., :3].astype(np.float32)
    empty_columns = max(
        0,
        viewport_columns
        - min(
            int(round(progress * (spectrogram.shape[1] - 1))) + 1,
            viewport_columns,
        ),
    )
    background = np.array([0.018, 0.028, 0.050], dtype=np.float32)
    frame[:, :empty_columns] = background

    grid_color = np.array([0.055, 0.105, 0.155], dtype=np.float32)
    for column in range(0, width, max(1, width // 8)):
        frame[:, column] = np.maximum(frame[:, column], grid_color)
    for row in range(0, height, max(1, height // 4)):
        frame[row, :] = np.maximum(frame[row, :], grid_color)

    live_column = width - 2
    glow_radius = max(3, width // 24)
    cyan = np.array([0.10, 0.90, 1.00], dtype=np.float32)
    for distance in range(glow_radius, 0, -1):
        column = live_column - distance
        strength = 0.24 * (1.0 - distance / glow_radius)
        frame[:, column] = (1.0 - strength) * frame[:, column] + strength * cyan
    frame[:, live_column] = cyan

    border = np.array([0.12, 0.18, 0.27], dtype=np.float32)
    frame[[0, -1], :, :] = border
    frame[:, [0, -1], :] = border
    return np.clip(frame, 0.0, 1.0)


def spectral_activity(spectrogram: np.ndarray, progress: float) -> float:
    """Return normalized RMS energy for the newest incoming time frame."""
    normalized = normalize_spectrogram(spectrogram)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")
    column = min(
        normalized.shape[1] - 1,
        int(round(progress * (normalized.shape[1] - 1))),
    )
    return float(np.sqrt(np.mean(np.square(normalized[:, column]))))


def replay_elapsed_seconds(
    progress: float,
    window_size: int,
    sample_rate: int,
) -> tuple[float, float]:
    """Return elapsed and total replay duration in seconds."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    total = window_size / sample_rate
    return progress * total, total
