"""Loading and discovery utilities for CWRU vibration recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

VALID_LABELS = frozenset({"normal", "ball", "inner_race", "outer_race"})
VALID_LOADS = frozenset(range(4))


@dataclass(frozen=True)
class RecordingMetadata:
    """Metadata inferred from a recording's CWRU directory layout."""

    path: Path
    label: str
    load: int
    signal_key: str
    sample_rate: int = 12000


def find_drive_end_key(mat_dict: dict[str, object]) -> str:
    """Return the first non-metadata MATLAB key containing ``DE_time``."""
    for key in mat_dict:
        if not key.startswith("__") and "DE_time" in key:
            return key
    raise ValueError("no drive-end signal key containing 'DE_time' was found")


def load_mat_signal(
    path: str | Path,
    signal_key: str | None = None,
) -> tuple[NDArray[np.float32], str]:
    """Load one vibration signal from a MATLAB file."""
    mat_path = Path(path)
    mat_dict = loadmat(mat_path)
    selected_key = (
        signal_key if signal_key is not None else find_drive_end_key(mat_dict)
    )

    if selected_key not in mat_dict:
        raise KeyError(
            f"signal key {selected_key!r} was not found in MATLAB file {mat_path}"
        )

    selected_array = np.asarray(mat_dict[selected_key])
    if not np.issubdtype(selected_array.dtype, np.number):
        raise TypeError(
            f"signal key {selected_key!r} in {mat_path} does not contain "
            "a numeric array"
        )
    if np.issubdtype(selected_array.dtype, np.complexfloating):
        raise ValueError(
            f"signal key {selected_key!r} in {mat_path} must contain real values"
        )

    signal = selected_array.astype(np.float32, copy=False).reshape(-1).copy()
    return signal, selected_key


def infer_metadata_from_path(
    path: str | Path,
    sample_rate: int = 12000,
) -> RecordingMetadata:
    """Infer recording metadata from ``load_<n>/<class>/<file>.mat``."""
    mat_path = Path(path)
    label = mat_path.parent.name
    load_directory = mat_path.parent.parent.name

    if label not in VALID_LABELS:
        valid = ", ".join(sorted(VALID_LABELS))
        raise ValueError(f"invalid CWRU label {label!r}; expected one of: {valid}")

    if not load_directory.startswith("load_"):
        raise ValueError(
            f"invalid load directory {load_directory!r}; expected load_0 through load_3"
        )

    load_text = load_directory.removeprefix("load_")
    try:
        load_number = int(load_text)
    except ValueError as error:
        raise ValueError(
            f"invalid load directory {load_directory!r}; expected load_0 through load_3"
        ) from error

    if load_number not in VALID_LOADS or load_directory != f"load_{load_number}":
        raise ValueError(
            f"invalid load directory {load_directory!r}; expected load_0 through load_3"
        )

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    mat_dict = loadmat(mat_path)
    signal_key = find_drive_end_key(mat_dict)
    return RecordingMetadata(
        path=mat_path,
        label=label,
        load=load_number,
        signal_key=signal_key,
        sample_rate=int(sample_rate),
    )


def discover_recordings(root: str | Path) -> list[RecordingMetadata]:
    """Discover CWRU MATLAB recordings in deterministic metadata order."""
    root_path = Path(root)
    recordings = [
        infer_metadata_from_path(path)
        for path in root_path.rglob("*.mat")
        if path.is_file()
    ]
    return sorted(
        recordings,
        key=lambda metadata: (
            metadata.load,
            metadata.label,
            metadata.path.name,
        ),
    )
