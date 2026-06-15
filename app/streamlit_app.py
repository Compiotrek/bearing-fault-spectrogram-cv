"""Interactive portfolio demo for bearing fault spectrogram classification."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_utils import (  # noqa: E402
    build_live_replay_frame,
    filter_samples,
    load_app_manifest,
    load_checkpoint_model,
    load_spectrogram,
    predict_spectrogram,
    replay_elapsed_seconds,
    selector_values,
    spectral_activity,
    spectrogram_to_tensor,
)
from src.gradcam import generate_gradcam, overlay_heatmap_on_spectrogram  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "resnet18_clean" / "best_model.pt"


@st.cache_data(show_spinner=False)
def cached_manifest(path: str, modified_time: float):
    """Load manifest data once per file version."""
    del modified_time
    return load_app_manifest(path, project_root=PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def cached_spectrogram(path: str, modified_time: float) -> np.ndarray:
    """Load one spectrogram once per file version."""
    del modified_time
    return load_spectrogram(path)


@st.cache_resource(show_spinner=False)
def cached_model(path: str, modified_time: float):
    """Load the trained checkpoint once per file version."""
    del modified_time
    return load_checkpoint_model(path, device="auto")


def file_modified_time(path: Path) -> float:
    """Return a cache key while preserving a clear missing-file error."""
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return path.stat().st_mtime


def spectrogram_figure(spectrogram: np.ndarray, title: str):
    """Create a compact time-frequency plot."""
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = axis.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axis.set(
        title=title,
        xlabel="Time frame",
        ylabel="Frequency bin",
    )
    figure.colorbar(image, ax=axis, label="Log power")
    return figure


def stop_with_error(message: str, error: Exception | None = None) -> None:
    """Show an actionable error and stop the current app run."""
    st.error(message)
    if error is not None:
        st.caption(str(error))
    st.stop()


st.set_page_config(
    page_title="Bearing Fault Spectrogram CV",
    layout="wide",
)

st.html(
    """
    <style>
    .replay-kicker {
        color: #58e6f2;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }
    .replay-status {
        align-items: center;
        color: #d8e4f2;
        display: flex;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem;
        gap: 0.55rem;
        padding: 0.2rem 0 0.5rem;
    }
    .replay-dot {
        background: #58e6f2;
        border-radius: 50%;
        box-shadow: 0 0 12px #58e6f2;
        height: 0.5rem;
        width: 0.5rem;
    }
    </style>
    """
)

st.title("Noise-Robust Bearing Fault Detection")
st.caption(
    "Explore cached CWRU spectrograms, model predictions, and Grad-CAM "
    "explanations. Replay is simulated from stored data; no live sensor is used."
)

try:
    manifest_mtime = file_modified_time(DEFAULT_MANIFEST)
    samples = cached_manifest(str(DEFAULT_MANIFEST), manifest_mtime)
except (FileNotFoundError, OSError, ValueError) as error:
    stop_with_error(
        "The processed dataset manifest could not be loaded. "
        "Build it with `python -m src.build_dataset` first.",
        error,
    )

with st.sidebar:
    st.header("Sample selection")
    split_options = selector_values(samples, "split")
    default_split = split_options.index("test") if "test" in split_options else 0
    selected_split = st.selectbox(
        "Split",
        split_options,
        index=default_split,
    )

    split_samples = filter_samples(samples, split=selected_split)
    variant_options = selector_values(split_samples, "variant")
    default_variant = (
        variant_options.index("clean") if "clean" in variant_options else 0
    )
    selected_variant = st.selectbox(
        "Variant",
        variant_options,
        index=default_variant,
    )

    variant_samples = filter_samples(
        split_samples,
        variant=selected_variant,
    )
    label_options = selector_values(variant_samples, "label")
    selected_label = st.selectbox("True label", label_options)

    filtered_samples = filter_samples(
        variant_samples,
        label=selected_label,
    )
    if not filtered_samples:
        stop_with_error("No samples match the selected filters.")
    selected_sample = st.selectbox(
        "Sample",
        filtered_samples,
        format_func=lambda sample: (
            f"{sample.sample_id} | start {sample.window_start:,}"
        ),
    )
    st.divider()
    st.caption(f"Manifest: `{DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)}`")
    st.caption(f"Checkpoint: `{DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)}`")

try:
    spectrogram_mtime = file_modified_time(selected_sample.spectrogram_path)
    spectrogram = cached_spectrogram(
        str(selected_sample.spectrogram_path),
        spectrogram_mtime,
    )
except (FileNotFoundError, OSError, TypeError, ValueError) as error:
    stop_with_error("The selected spectrogram could not be loaded.", error)

try:
    checkpoint_mtime = file_modified_time(DEFAULT_CHECKPOINT)
    model, class_mapping, device = cached_model(
        str(DEFAULT_CHECKPOINT),
        checkpoint_mtime,
    )
    prediction = predict_spectrogram(
        model,
        spectrogram,
        class_mapping,
        device,
    )
except (
    FileNotFoundError,
    OSError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
) as error:
    stop_with_error("The trained model could not be loaded or executed.", error)

prediction_column, spectrogram_column = st.columns([1, 2])
with prediction_column:
    st.subheader("Prediction")
    st.metric("Predicted class", prediction.predicted_class)
    st.metric("True class", selected_sample.label)
    st.metric("Confidence", f"{prediction.confidence:.1%}")
    if prediction.predicted_class == selected_sample.label:
        st.success("Correct classification")
    else:
        st.warning("Misclassification")

    st.markdown("**Class probabilities**")
    for class_name, probability in prediction.probabilities.items():
        st.progress(
            probability,
            text=f"{class_name}: {probability:.1%}",
        )

with spectrogram_column:
    st.subheader("Selected spectrogram")
    figure = spectrogram_figure(
        spectrogram,
        f"{selected_variant} | {selected_sample.sample_id}",
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

st.divider()
st.subheader("Grad-CAM explanation")
st.caption("Warm regions contributed most strongly to the model's predicted class.")
try:
    with st.spinner("Computing Grad-CAM..."):
        input_tensor = spectrogram_to_tensor(spectrogram, device)
        heatmap = generate_gradcam(
            model,
            input_tensor,
            target_class=prediction.predicted_class_id,
        )
        overlay = overlay_heatmap_on_spectrogram(spectrogram, heatmap)
    st.image(
        overlay,
        caption=(
            f"Grad-CAM for {prediction.predicted_class} ({prediction.confidence:.1%})"
        ),
        width="stretch",
        clamp=True,
    )
except (RuntimeError, TypeError, ValueError) as error:
    st.warning(f"Grad-CAM could not be generated: {error}")

st.divider()
st.html('<div class="replay-kicker">Signal Analysis Console</div>')
st.subheader("Spectrogram replay")
st.caption(
    "A simulated rolling monitor fed by the stored 2,048-sample window. "
    "No live sensor or background stream is connected."
)

with st.container(border=True):
    control_column, speed_column, play_column = st.columns([4, 1.4, 1])
    with control_column:
        replay_position = st.slider(
            "Window position",
            min_value=0,
            max_value=100,
            value=100,
            format="%d%%",
            label_visibility="collapsed",
        )
    with speed_column:
        replay_speed = st.segmented_control(
            "Playback speed",
            options=("0.5×", "1×", "2×"),
            default="1×",
            label_visibility="collapsed",
            width="stretch",
        )
    with play_column:
        play_replay = st.button(
            "Run scan",
            type="primary",
            width="stretch",
        )

    replay_view = st.empty()
    replay_status = st.empty()
    replay_metrics = st.empty()

    def render_replay(progress: float, status: str) -> None:
        elapsed, total = replay_elapsed_seconds(
            progress,
            window_size=2048,
            sample_rate=selected_sample.sample_rate,
        )
        replay_view.image(
            build_live_replay_frame(spectrogram, progress),
            width="stretch",
            clamp=True,
        )
        replay_status.html(
            '<div class="replay-status">'
            '<span class="replay-dot"></span>'
            f"<span>{status}</span>"
            "</div>"
        )
        with replay_metrics.container():
            time_column, frame_column, activity_column, rate_column = st.columns(4)
            time_column.metric(
                "Signal time",
                f"{elapsed * 1000:06.1f} ms",
                border=True,
            )
            frame_column.metric(
                "Time frame",
                f"{int(round(progress * (spectrogram.shape[1] - 1))):03d}"
                f" / {spectrogram.shape[1] - 1:03d}",
                border=True,
            )
            activity_column.metric(
                "Activity",
                f"{spectral_activity(spectrogram, progress):.0%}",
                help="Normalized RMS energy in the newest spectrogram column.",
                border=True,
            )
            rate_column.metric(
                "Sample rate",
                f"{selected_sample.sample_rate / 1000:g} kHz",
                help=f"Window duration: {total * 1000:.1f} ms",
                border=True,
            )

    if play_replay:
        delay = {"0.5×": 0.07, "1×": 0.035, "2×": 0.018}[replay_speed]
        for frame_index, progress in enumerate(np.linspace(0.0, 1.0, 73)):
            render_replay(float(progress), "LIVE BUFFER · RECEIVING NEXT FRAME")
            if frame_index < 72:
                time.sleep(delay)
        render_replay(1.0, "BUFFER COMPLETE · REPLAY READY")
    else:
        render_replay(
            replay_position / 100.0,
            "PAUSED · DRAG TIMELINE OR START BUFFER",
        )

with st.expander("Sample metadata"):
    st.json(
        {
            "sample_id": selected_sample.sample_id,
            "split": selected_sample.split,
            "variant": selected_sample.variant,
            "label": selected_sample.label,
            "load": selected_sample.load,
            "window_start": selected_sample.window_start,
            "sample_rate": selected_sample.sample_rate,
            "signal_key": selected_sample.signal_key,
            "device": str(device),
        }
    )
