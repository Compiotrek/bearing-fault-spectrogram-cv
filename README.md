# Bearing Fault Spectrogram CV

Noise-robust bearing fault detection from CWRU vibration signals with
spectrogram classification, robustness evaluation, Grad-CAM explanations, and
a live-like diagnostic replay.

## Interactive Demo

The primary portfolio demo is a FastAPI backend with a React/Vite/TypeScript
frontend. Its canvas monitor simulates a live stream from stored spectrogram
columns: new data enters on the right, history rolls left, and older columns
fade like a diagnostic display. It does not connect to real sensor hardware.

Start the backend from the repository root:

```bash
uvicorn api.main:app --reload
```

Start the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The API uses these artifacts by default:

- `data/processed/manifest.csv`
- `models/resnet18_clean/best_model.pt`

The previous Streamlit demo remains available as a fallback:

```bash
streamlit run app/streamlit_app.py
```

## Development

Install the project with its development dependencies:

```bash
python -m pip install --editable ".[dev]"
```

On Intel-based macOS, use Python 3.12. PyTorch does not publish Python 3.13
wheels for macOS `x86_64`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --editable ".[dev]"
```

Run the quality checks:

```bash
ruff format --check api src tests
ruff check api src tests
python -m pytest
cd frontend && npm run build
```

Raw CWRU `.mat` files remain local under `data/raw/cwru/` and are excluded from
version control.
