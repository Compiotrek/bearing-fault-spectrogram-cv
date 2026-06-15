"""Build and load cached spectrogram datasets."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_loader import discover_recordings, load_mat_signal
from src.denoise import butterworth_bandpass_filter
from src.noise import add_gaussian_noise
from src.segment import segment_signal
from src.spectrogram import compute_log_power_spectrogram, resize_spectrogram

MANIFEST_FIELDS = (
    "sample_id",
    "recording_path",
    "spectrogram_path",
    "label",
    "label_id",
    "load",
    "split",
    "variant",
    "window_start",
    "signal_key",
    "sample_rate",
)


@dataclass(frozen=True)
class ProcessedSample:
    """Metadata for one cached spectrogram."""

    sample_id: str
    recording_path: Path
    spectrogram_path: Path
    label: str
    label_id: int
    load: int
    split: str
    variant: str
    window_start: int
    signal_key: str
    sample_rate: int


class SpectrogramDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load cached spectrograms as PyTorch tensors."""

    def __init__(
        self,
        samples: list[ProcessedSample],
        split: str | None = None,
        variant: str | None = None,
        repeat_channels: bool = False,
    ) -> None:
        filtered_samples = [
            sample
            for sample in samples
            if (split is None or sample.split == split)
            and (variant is None or sample.variant == variant)
        ]
        if not filtered_samples:
            raise ValueError("no samples remain after applying dataset filters")
        self.samples = filtered_samples
        self.repeat_channels = repeat_channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        spectrogram = np.load(sample.spectrogram_path, allow_pickle=False)
        if spectrogram.ndim != 2:
            raise ValueError(
                f"spectrogram at {sample.spectrogram_path} must be two-dimensional"
            )
        if not np.issubdtype(spectrogram.dtype, np.number):
            raise TypeError(f"spectrogram at {sample.spectrogram_path} must be numeric")
        if not np.all(np.isfinite(spectrogram)):
            raise ValueError(
                f"spectrogram at {sample.spectrogram_path} must contain finite values"
            )

        image = torch.from_numpy(spectrogram.astype(np.float32, copy=False)).unsqueeze(
            0
        )
        if self.repeat_channels:
            image = image.repeat(3, 1, 1)
        label = torch.tensor(sample.label_id, dtype=torch.long)
        return image, label


def split_from_load(load: int) -> str:
    """Map a CWRU motor load to its dataset split."""
    if isinstance(load, (bool, np.bool_)) or not isinstance(load, (int, np.integer)):
        raise ValueError("load must be one of 0, 1, 2, or 3")
    if load in (0, 1):
        return "train"
    if load == 2:
        return "val"
    if load == 3:
        return "test"
    raise ValueError("load must be one of 0, 1, 2, or 3")


def class_mapping() -> dict[str, int]:
    """Return the fixed class-to-index mapping."""
    return {
        "normal": 0,
        "ball": 1,
        "inner_race": 2,
        "outer_race": 3,
    }


def _stable_noise_seed(
    global_seed: int,
    recording_path: Path,
    window_start: int,
    snr_db: int,
) -> int:
    seed_material = f"{global_seed}|{recording_path.as_posix()}|{window_start}|{snr_db}"
    digest = hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _sample_id(
    *,
    load: int,
    label: str,
    recording_stem: str,
    window_start: int,
    variant: str,
) -> str:
    return f"load_{load}_{label}_{recording_stem}_start_{window_start:08d}_{variant}"


def _save_sample(
    *,
    spectrogram: np.ndarray,
    output_root: Path,
    recording_path: Path,
    recording_stem: str,
    label: str,
    label_id: int,
    load: int,
    split: str,
    variant: str,
    window_start: int,
    signal_key: str,
    sample_rate: int,
    spectrogram_shape: tuple[int, int],
) -> ProcessedSample:
    sample_id = _sample_id(
        load=load,
        label=label,
        recording_stem=recording_stem,
        window_start=window_start,
        variant=variant,
    )
    resized = resize_spectrogram(spectrogram, output_shape=spectrogram_shape)
    spectrogram_path = (
        output_root / "spectrograms" / split / variant / f"{sample_id}.npy"
    )
    spectrogram_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spectrogram_path, resized, allow_pickle=False)
    return ProcessedSample(
        sample_id=sample_id,
        recording_path=recording_path,
        spectrogram_path=spectrogram_path,
        label=label,
        label_id=label_id,
        load=load,
        split=split,
        variant=variant,
        window_start=window_start,
        signal_key=signal_key,
        sample_rate=sample_rate,
    )


def build_processed_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    window_size: int = 2048,
    stride: int = 1024,
    sample_rate: int = 12000,
    snr_levels: tuple[int, ...] = (10, 5, 0),
    denoise_lowcut: float = 100.0,
    denoise_highcut: float = 5000.0,
    spectrogram_shape: tuple[int, int] = (224, 224),
    seed: int = 42,
) -> list[ProcessedSample]:
    """Build cached clean, noisy, and denoised spectrogram variants."""
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not isinstance(snr_levels, tuple) or any(
        isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer))
        for level in snr_levels
    ):
        raise TypeError("snr_levels must be a tuple of integers")
    if len(set(snr_levels)) != len(snr_levels):
        raise ValueError("snr_levels must not contain duplicates")

    labels = class_mapping()
    processed_samples: list[ProcessedSample] = []

    for recording in discover_recordings(raw_root_path):
        signal, signal_key = load_mat_signal(
            recording.path,
            signal_key=recording.signal_key,
        )
        windows, starts = segment_signal(
            signal,
            window_size=window_size,
            stride=stride,
        )
        split = split_from_load(recording.load)
        try:
            relative_recording_path = recording.path.relative_to(raw_root_path)
        except ValueError:
            relative_recording_path = recording.path

        for window, start in zip(windows, starts, strict=True):
            window_start = int(start)
            clean_spectrogram = compute_log_power_spectrogram(
                window,
                sample_rate=sample_rate,
            )
            processed_samples.append(
                _save_sample(
                    spectrogram=clean_spectrogram,
                    output_root=output_root_path,
                    recording_path=recording.path,
                    recording_stem=recording.path.stem,
                    label=recording.label,
                    label_id=labels[recording.label],
                    load=recording.load,
                    split=split,
                    variant="clean",
                    window_start=window_start,
                    signal_key=signal_key,
                    sample_rate=sample_rate,
                    spectrogram_shape=spectrogram_shape,
                )
            )

            for snr_db in snr_levels:
                noise_seed = _stable_noise_seed(
                    int(seed),
                    relative_recording_path,
                    window_start,
                    int(snr_db),
                )
                noisy_signal = add_gaussian_noise(
                    window,
                    snr_db=float(snr_db),
                    seed=noise_seed,
                )
                noisy_variant = f"noisy_{snr_db}db"
                noisy_spectrogram = compute_log_power_spectrogram(
                    noisy_signal,
                    sample_rate=sample_rate,
                )
                processed_samples.append(
                    _save_sample(
                        spectrogram=noisy_spectrogram,
                        output_root=output_root_path,
                        recording_path=recording.path,
                        recording_stem=recording.path.stem,
                        label=recording.label,
                        label_id=labels[recording.label],
                        load=recording.load,
                        split=split,
                        variant=noisy_variant,
                        window_start=window_start,
                        signal_key=signal_key,
                        sample_rate=sample_rate,
                        spectrogram_shape=spectrogram_shape,
                    )
                )

                denoised_signal = butterworth_bandpass_filter(
                    noisy_signal,
                    sample_rate=sample_rate,
                    lowcut=denoise_lowcut,
                    highcut=denoise_highcut,
                )
                denoised_variant = f"denoised_{snr_db}db"
                denoised_spectrogram = compute_log_power_spectrogram(
                    denoised_signal,
                    sample_rate=sample_rate,
                )
                processed_samples.append(
                    _save_sample(
                        spectrogram=denoised_spectrogram,
                        output_root=output_root_path,
                        recording_path=recording.path,
                        recording_stem=recording.path.stem,
                        label=recording.label,
                        label_id=labels[recording.label],
                        load=recording.load,
                        split=split,
                        variant=denoised_variant,
                        window_start=window_start,
                        signal_key=signal_key,
                        sample_rate=sample_rate,
                        spectrogram_shape=spectrogram_shape,
                    )
                )

    output_root_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for sample in processed_samples:
            row = asdict(sample)
            row["recording_path"] = str(sample.recording_path)
            row["spectrogram_path"] = str(sample.spectrogram_path)
            writer.writerow(row)

    return processed_samples


def load_manifest(path: str | Path) -> list[ProcessedSample]:
    """Load processed sample metadata from a CSV manifest."""
    manifest_path = Path(path)
    samples: list[ProcessedSample] = []
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or set(reader.fieldnames) != set(MANIFEST_FIELDS):
            raise ValueError("manifest has missing or unexpected columns")
        for row in reader:
            samples.append(
                ProcessedSample(
                    sample_id=row["sample_id"],
                    recording_path=Path(row["recording_path"]),
                    spectrogram_path=Path(row["spectrogram_path"]),
                    label=row["label"],
                    label_id=int(row["label_id"]),
                    load=int(row["load"]),
                    split=row["split"],
                    variant=row["variant"],
                    window_start=int(row["window_start"]),
                    signal_key=row["signal_key"],
                    sample_rate=int(row["sample_rate"]),
                )
            )
    return samples
