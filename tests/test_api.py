import base64
import csv
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from src.dataset import MANIFEST_FIELDS, ProcessedSample


def make_api_client(tmp_path: Path) -> tuple[TestClient, ProcessedSample]:
    spectrogram_path = tmp_path / "spectrogram.npy"
    np.save(
        spectrogram_path,
        np.arange(64, dtype=np.float32).reshape(8, 8),
        allow_pickle=False,
    )
    sample = ProcessedSample(
        sample_id="test_sample_clean",
        recording_path=tmp_path / "recording.mat",
        spectrogram_path=spectrogram_path,
        label="normal",
        label_id=0,
        load=3,
        split="test",
        variant="clean",
        window_start=0,
        signal_key="X001_DE_time",
        sample_rate=12000,
    )
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                **sample.__dict__,
                "recording_path": str(sample.recording_path),
                "spectrogram_path": str(sample.spectrogram_path),
            }
        )
    app = create_app(
        manifest_path=manifest_path,
        checkpoint_path=tmp_path / "missing-model.pt",
        project_root=tmp_path,
    )
    return TestClient(app), sample


def test_health_reports_artifact_readiness(tmp_path: Path) -> None:
    client, _ = make_api_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "manifest_ready": True,
        "checkpoint_ready": False,
    }


def test_samples_returns_manifest_metadata(tmp_path: Path) -> None:
    client, sample = make_api_client(tmp_path)

    response = client.get("/samples", params={"split": "test"})

    assert response.status_code == 200
    assert response.json() == [
        {
            "sample_id": sample.sample_id,
            "label": "normal",
            "split": "test",
            "variant": "clean",
            "load": 3,
            "window_start": 0,
            "sample_rate": 12000,
        }
    ]


def test_spectrogram_returns_normalized_values(tmp_path: Path) -> None:
    client, sample = make_api_client(tmp_path)

    response = client.get(f"/spectrogram/{sample.sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["height"] == 8
    assert body["width"] == 8
    assert np.asarray(body["values"]).min() == 0.0
    assert np.asarray(body["values"]).max() == 1.0


def test_unknown_sample_returns_404(tmp_path: Path) -> None:
    client, _ = make_api_client(tmp_path)

    response = client.get("/samples/unknown")

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown sample_id: unknown"


def test_gradcam_returns_raw_attribution_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, sample = make_api_client(tmp_path)
    heatmap = np.asarray(
        [
            [0.0, 0.2, 0.4],
            [0.1, 0.8, 1.0],
        ],
        dtype=np.float32,
    )
    service = client.app.state.demo_service
    monkeypatch.setattr(
        service,
        "gradcam_analysis",
        lambda _sample: (b"png-bytes", heatmap),
    )

    response = client.get(f"/gradcam/{sample.sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["height"] == 2
    assert body["width"] == 3
    np.testing.assert_allclose(body["values"], heatmap)
    np.testing.assert_allclose(body["time_profile"], heatmap.mean(axis=0))
    np.testing.assert_allclose(body["frequency_profile"], heatmap.mean(axis=1))
    assert body["peak_time_index"] == 2
    assert body["peak_frequency_index"] == 1
    assert base64.b64decode(body["image_base64"]) == b"png-bytes"
