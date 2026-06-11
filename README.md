# Bearing Fault Spectrogram CV

Noise-robust bearing fault detection from CWRU vibration signals. The current
foundation provides deterministic signal segmentation, MATLAB data loading,
Gaussian noise injection, and zero-phase Butterworth denoising.

## Development

Install the project with its development dependencies:

```bash
python -m pip install --editable ".[dev]"
```

Run the quality checks:

```bash
ruff format --check src tests
ruff check src tests
python -m pytest
```

Raw CWRU `.mat` files remain local under `data/raw/cwru/` and are excluded from
version control.
