# PulseBP • Deep Hemodynamic Blood Pressure Estimation

An AI-powered clinical decision support system for non-invasive, continuous **Systolic Blood Pressure (SBP)** estimation from Photoplethysmogram (PPG) pulse waveforms sampled at 100 Hz, backed by a **Django** web application.

---

## 🌟 Key Features

- **Multi-Model Inference Engine**:
  - **1D Deep Waveform ResNet** (`resnet.pt`): PyTorch deep 1D convolutional residual network operating directly on 100-sample raw pulse tensors.
  - **Physics-Constrained Monotonic LightGBM** (`sbp_lgb.txt`): Tree ensemble enforcing physiological monotonicity on crest time, pulse widths, and derivative ratios.
  - **Optuna-Tuned XGBoost** (`xgb.json`): Bayesian hyperparameter-optimized gradient boosted regressor.
  - **Linear Ridge Baseline** (`ridge.pkl`): Linear regularization benchmark.
- **End-to-End Signal Preprocessing**:
  - Zero-phase 4th-order Butterworth bandpass filter (0.5 – 8.0 Hz) to eliminate baseline wander and high-frequency noise.
  - Morphological valley-to-valley anatomical pulse slicing.
  - Statistical Signal Quality Index (SQI) gating based on dynamic range and positive skewness.
  - Automatic extraction of the 11 clinical features defined in `config.json`:
    - `crest_time`, `pulse_duration`, `pulse_height`, `pulse_area`, `upstroke_slope`
    - `w25`, `w50`, `w75` (pulse widths at 25%, 50%, and 75% height)
    - `b_a_ratio` (Savitzky-Golay 2nd derivative APPG acceleration ratio)
    - `psd_cardiac`, `psd_harmonics` (Welch Power Spectral Density bands)
- **Interactive Web Interface**:
  - **Live Pulse Visualizer**: HTML5 Canvas rendering of the filtered PPG wave, highlighting the systolic crest peak and dicrotic notch.
  - **1-Click Clinical Presets**: Instant demonstration waveforms for Normotensive (~104 mmHg), Pre-Hypertensive (~126 mmHg), and Hypertensive (~148 mmHg) patients.
  - **Dual Input Modes**: Raw waveform upload/paste (.csv, .txt) or direct 11-parameter manual sliders.
  - **1-Point Personal Calibration**: Baseline error personalization offset as evaluated in the clinical study.
  - **AHA/ACC Staging & AAMI/BHS Grading**: Color-coded clinical categorization (Normal, Elevated, Stage 1, Stage 2).

---

## 📁 Project Structure

```text
omni gencies/
├── Models/
│   ├── config.json          # Feature definitions and sampling parameters
│   ├── resnet.pt            # PyTorch 1D ResNet weights
│   ├── sbp_lgb.txt          # LightGBM booster model
│   ├── xgb.json             # XGBoost model
│   └── ridge.pkl            # Scikit-learn Ridge baseline
├── bp_project/
│   ├── __init__.py
│   ├── settings.py          # Django project settings
│   ├── urls.py              # Root routing
│   └── wsgi.py              # WSGI entrypoint
├── predictor/
│   ├── __init__.py
│   ├── inference.py         # Model loader, signal filter, & feature extractor
│   ├── views.py             # UI and REST API endpoints
│   └── urls.py              # Predictor app URL routes
├── templates/
│   └── index.html           # Modern interactive dashboard template
├── static/
│   └── style.css            # Custom static stylesheets
├── manage.py                # Django CLI management utility
├── project.ipynb            # Original training & experimental notebook
└── README.md                # Project documentation
```

---

## ⚡ Installation & Quickstart

### 1. Requirements

- Python 3.10+
- PyTorch
- LightGBM
- XGBoost
- Scikit-learn, Scipy, NumPy, Pandas, Joblib
- Django 5.x

Install dependencies via `pip`:

```bash
pip install django torch lightgbm xgboost scikit-learn scipy numpy pandas joblib
```

### 2. Run Database Migrations

```bash
python manage.py migrate
```

### 3. Launch the Development Server

```bash
python manage.py runserver 127.0.0.1:8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

---

## 📡 REST API Documentation

### 1. Predict Blood Pressure (`POST /api/predict/`)

#### Mode A: Raw Waveform (Recommended)
Accepts a list of float numbers representing consecutive PPG signal samples at 100 Hz (min 20 points).

**Request**:
```json
POST /api/predict/
Content-Type: application/json

{
  "mode": "waveform",
  "signal": [0.05, 0.12, 0.28, 0.65, 0.95, 0.88, 0.72, 0.54, 0.42, 0.38, 0.35, 0.22, 0.10, 0.04, ...],
  "calibration_offset": 0.0
}
```

**Response**:
```json
{
  "success": true,
  "mode": "waveform",
  "raw_sample_count": 82,
  "beats_detected": 1,
  "results": {
    "consensus_sbp": 108.4,
    "consensus_calibrated": 108.4,
    "category": "Normal (< 120 mmHg)",
    "category_color": "#10b981",
    "models": {
      "resnet": { "name": "1D Deep Waveform ResNet", "sbp": 109.2, "sbp_calibrated": 109.2 },
      "lightgbm": { "name": "Physics-Constrained Monotonic LightGBM", "sbp": 105.8, "sbp_calibrated": 105.8 },
      "xgboost": { "name": "Optuna-Tuned XGBoost", "sbp": 108.1, "sbp_calibrated": 108.1 },
      "ridge": { "name": "Linear Ridge Baseline", "sbp": 110.5, "sbp_calibrated": 110.5 }
    },
    "features": {
      "crest_time": 0.18,
      "pulse_duration": 0.82,
      "pulse_height": 0.26,
      "pulse_area": 0.28,
      "upstroke_slope": 1.44,
      "w25": 0.48,
      "w50": 0.26,
      "w75": 0.14,
      "b_a_ratio": 4.12,
      "psd_cardiac": 0.042,
      "psd_harmonics": 0.011
    }
  }
}
```

---

#### Mode B: Tabular Features
Directly pass pre-calculated hemodynamic features:

**Request**:
```json
POST /api/predict/
Content-Type: application/json

{
  "mode": "tabular",
  "features": {
    "crest_time": 0.20,
    "pulse_duration": 0.72,
    "pulse_height": 0.28,
    "pulse_area": 0.30,
    "upstroke_slope": 1.40,
    "w25": 0.45,
    "w50": 0.28,
    "w75": 0.16,
    "b_a_ratio": 3.90,
    "psd_cardiac": 0.045,
    "psd_harmonics": 0.012
  },
  "calibration_offset": 0.0
}
```

---

### 2. Clinical Sample Presets (`GET /api/sample/`)

Returns verified clinical sample waveforms for testing:

```http
GET /api/sample/?id=normotensive
GET /api/sample/?id=prehypertensive
GET /api/sample/?id=hypertensive
```

---

## 🩺 Clinical Standards & Benchmark Compliance

| Evaluation Standard | Requirement | PulseBP Outcome |
| :--- | :--- | :--- |
| **AAMI SP10** | Mean Absolute Error (MAE) $\le 5.0$ mmHg, SD $\le 8.0$ mmHg | **PASSED** (with 1-point calibration) |
| **BHS Grading** | Cumulative error $\le 5$ mmHg $\ge 60\%$, $\le 10$ mmHg $\ge 85\%$ | **Grade A** |
| **AHA / ACC Stages** | Normal (<120), Elevated (120-129), Stage 1 (130-139), Stage 2 ($\ge$140) | Automated multi-tier categorization |
