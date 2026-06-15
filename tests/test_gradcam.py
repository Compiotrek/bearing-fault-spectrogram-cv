from pathlib import Path

import numpy as np
import pytest
import torch

from src.gradcam import (
    generate_gradcam,
    overlay_heatmap_on_spectrogram,
    save_gradcam_example,
)
from src.model import SmallSpectrogramCNN


def test_generate_gradcam_returns_normalized_input_shape() -> None:
    model = SmallSpectrogramCNN(num_classes=4)
    model.eval()
    input_tensor = torch.randn(1, 1, 32, 48)

    heatmap = generate_gradcam(model, input_tensor)

    assert heatmap.shape == (32, 48)
    assert heatmap.dtype == np.float32
    assert np.all(np.isfinite(heatmap))
    assert np.all((0.0 <= heatmap) & (heatmap <= 1.0))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_generate_gradcam_accepts_explicit_target_class() -> None:
    model = SmallSpectrogramCNN(num_classes=4)
    model.eval()

    heatmap = generate_gradcam(
        model,
        torch.randn(1, 1, 32, 32),
        target_class=2,
    )

    assert heatmap.shape == (32, 32)


def test_overlay_returns_rgb_image_without_mutating_inputs() -> None:
    spectrogram = np.arange(12, dtype=np.float32).reshape(3, 4)
    heatmap = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    original_spectrogram = spectrogram.copy()
    original_heatmap = heatmap.copy()

    overlay = overlay_heatmap_on_spectrogram(spectrogram, heatmap)

    assert overlay.shape == (3, 4, 3)
    assert overlay.dtype == np.float32
    assert np.all((0.0 <= overlay) & (overlay <= 1.0))
    np.testing.assert_array_equal(spectrogram, original_spectrogram)
    np.testing.assert_array_equal(heatmap, original_heatmap)


def test_overlay_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        overlay_heatmap_on_spectrogram(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_overlay_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        overlay_heatmap_on_spectrogram(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            alpha=alpha,
        )


def test_save_gradcam_example_creates_png(tmp_path: Path) -> None:
    output_path = tmp_path / "figures" / "example.png"

    save_gradcam_example(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.eye(4, dtype=np.float32),
        output_path,
        title="Grad-CAM example",
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
