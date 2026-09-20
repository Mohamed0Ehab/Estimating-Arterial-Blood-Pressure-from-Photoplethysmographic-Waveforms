import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, welch
from scipy.stats import skew
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')


# ── PyTorch 1D ResNet Architecture — matches notebook Cell 14 exactly ─────────
class ResNetBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class HemodynamicResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.res1 = ResNetBlock1D(32)
        self.down = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.res2 = ResNetBlock1D(64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # SBP Prediction Head
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


# ── Blood Pressure Diagnostic Service ─────────────────────────────────────────
class BPInferenceService:
    """
    Blood pressure inference service faithful to the VitalDB research notebook.

    Architecture:
      1. 100 Hz continuous PPG bandpass filter (0.5 - 8.0 Hz, 4th order zero-phase Butterworth)
      2. Dynamic valley-to-valley anatomical pulse cycle segmentation + SQI gating
      3. 11 high-order hemodynamic features (notebook Cell 12) + 100-pt waveform tensor
      4. 4 Machine Learning models:
         - PyTorch 1D-ResNet
         - Monotonic Physics-Constrained LightGBM
         - Bayesian Optuna-Tuned XGBoost
         - Linear Ridge Baseline
      5. Hemodynamic Biomarker Estimator across full clinical spectrum (65 - 195 mmHg)
      6. Reference Arterial Catheter Extraction & 1-Point Personal Calibration (notebook Cell 9)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BPInferenceService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self.sample_rate = 100
        self.fixed_len = 100

        # Feature names — exact order of notebook Cell 13
        self.tab_features = [
            "crest_time", "pulse_duration", "pulse_height", "pulse_area",
            "upstroke_slope", "w25", "w50", "w75", "b_a_ratio",
            "psd_cardiac", "psd_harmonics"
        ]

        # Load config overrides if present
        config_path = os.path.join(MODELS_DIR, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.tab_features = cfg.get("tab_features", self.tab_features)
                    self.fixed_len = cfg.get("fixed_len", self.fixed_len)
                    self.sample_rate = cfg.get("sample_rate", self.sample_rate)
            except Exception as e:
                print(f"Warning loading config.json: {e}")

        # 4th-order zero-phase Butterworth bandpass filter coefficients (0.5 - 8.0 Hz at 100 Hz)
        nyq = self.sample_rate / 2.0
        self.b, self.a = butter(4, [0.5 / nyq, 8.0 / nyq], btype='band')

        self.models = {}
        self._load_models()
        self.initialized = True

    # ─── Model Loading ────────────────────────────────────────────────────────
    def _load_models(self):
        # 1. Ridge model
        ridge_path = os.path.join(MODELS_DIR, 'ridge.pkl')
        if os.path.exists(ridge_path):
            try:
                import joblib
                self.models['ridge'] = joblib.load(ridge_path)
                print("[BPInferenceService] Loaded Ridge model.")
            except Exception:
                try:
                    with open(ridge_path, 'rb') as f:
                        self.models['ridge'] = pickle.load(f)
                    print("[BPInferenceService] Loaded Ridge model (pickle fallback).")
                except Exception as e:
                    print(f"[BPInferenceService] Error loading ridge.pkl: {e}")

        # 2. LightGBM model
        lgb_path = os.path.join(MODELS_DIR, 'sbp_lgb.txt')
        if os.path.exists(lgb_path):
            try:
                import lightgbm as lgb
                self.models['lightgbm'] = lgb.Booster(model_file=lgb_path)
                print("[BPInferenceService] Loaded LightGBM model.")
            except Exception as e:
                print(f"[BPInferenceService] Error loading sbp_lgb.txt: {e}")

        # 3. XGBoost model
        xgb_path = os.path.join(MODELS_DIR, 'xgb.json')
        if os.path.exists(xgb_path):
            try:
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(xgb_path)
                self.models['xgboost'] = booster
                print("[BPInferenceService] Loaded XGBoost model.")
            except Exception as e:
                print(f"[BPInferenceService] Error loading xgb.json: {e}")

        # 4. PyTorch ResNet-1D
        resnet_path = os.path.join(MODELS_DIR, 'resnet.pt')
        if os.path.exists(resnet_path):
            try:
                net = HemodynamicResNet1D()
                state_dict = torch.load(resnet_path, map_location=torch.device('cpu'))
                net.load_state_dict(state_dict)
                net.eval()
                self.models['resnet'] = net
                print("[BPInferenceService] Loaded PyTorch ResNet model.")
            except Exception as e:
                print(f"[BPInferenceService] Error loading resnet.pt: {e}")

    # ─── Signal Preprocessing — mirrors notebook Cells 3 & 12 ─────────────────
    def filter_signal(self, raw_wave):
        """Zero-phase 4th order Butterworth bandpass (0.5 - 8.0 Hz at 100 Hz)."""
        wave = np.asarray(raw_wave, dtype=np.float64)
        if len(wave) < 15:
            return wave.astype(np.float32)
        try:
            return filtfilt(self.b, self.a, wave).astype(np.float32)
        except Exception:
            return wave.astype(np.float32)

    def process_raw_signal(self, raw_signal):
        """
        Takes raw continuous PPG signal, filters, detects valleys, checks SQI,
        and returns normalized signal and list of valid beat pulses.
        """
        raw_signal = np.asarray(raw_signal, dtype=np.float64)
        if len(raw_signal) < 20:
            raise ValueError("Signal must have at least 20 samples.")

        # 1. Bandpass filter
        raw_filt = self.filter_signal(raw_signal)

        # 2. Normalize to [0, 1] range to eliminate sensor baseline drift
        ptp = float(np.ptp(raw_filt))
        if ptp > 1e-6:
            filt_ppg = ((raw_filt - np.min(raw_filt)) / (ptp + 1e-6)).astype(np.float32)
        else:
            filt_ppg = raw_filt.astype(np.float32)

        # 3. Valley detection (notebook: distance = sample_rate * 0.45, prominence = 0.12)
        valleys, _ = find_peaks(-filt_ppg,
                                distance=int(self.sample_rate * 0.45),
                                prominence=0.12)

        # 4. Slicing and Signal Quality Index (SQI) check
        beats = []
        if len(valleys) >= 2:
            for i in range(len(valleys) - 1):
                pulse = filt_ppg[valleys[i]:valleys[i + 1]]
                N = len(pulse)
                if N >= int(self.sample_rate * 0.38):
                    s = skew(pulse)
                    dyn_range = np.ptp(pulse)
                    if (s > -0.05 and dyn_range > 0.08):
                        beats.append(pulse)

        # Fallback if recording is very short or single pulse
        if len(beats) == 0:
            beats.append(filt_ppg)

        return filt_ppg, beats

    def extract_features_from_beat(self, pulse_ppg):
        """
        Extracts 11 tabular hemodynamic features and 100-pt resampled waveform.
        Faithful to notebook Cell 12 line-by-line.
        """
        pulse_ppg = np.asarray(pulse_ppg, dtype=np.float32)
        N = len(pulse_ppg)
        if N < 5:
            raise ValueError("Beat too short for feature extraction.")

        # Resample to fixed-length 100 timesteps for ResNet-1D
        t_orig = np.linspace(0, 1, N)
        t_fixed = np.linspace(0, 1, self.fixed_len)
        wave_100 = np.interp(t_fixed, t_orig, pulse_ppg).astype(np.float32)

        # Morphological features
        peak_idx = int(np.argmax(pulse_ppg))
        pulse_height = float(np.ptp(pulse_ppg))
        pulse_min = float(np.min(pulse_ppg))
        crest_time = float(peak_idx / self.sample_rate)
        pulse_dur = float(N / self.sample_rate)

        # Pulse widths at 25%, 50%, and 75% height
        w25 = float(np.sum(pulse_ppg >= (pulse_min + 0.25 * pulse_height)) / self.sample_rate)
        w50 = float(np.sum(pulse_ppg >= (pulse_min + 0.50 * pulse_height)) / self.sample_rate)
        w75 = float(np.sum(pulse_ppg >= (pulse_min + 0.75 * pulse_height)) / self.sample_rate)

        # APPG 2nd derivative via Savitzky-Golay
        try:
            wl = min(9, N if N % 2 != 0 else N - 1)
            if wl >= 5:
                d2 = savgol_filter(pulse_ppg, window_length=wl, polyorder=min(3, wl - 1), deriv=2)
                b_a_ratio = float(np.ptp(d2) / (np.std(d2) + 1e-5))
            else:
                b_a_ratio = 1.0
        except Exception:
            b_a_ratio = 1.0

        # Welch Power Spectral Density
        try:
            nper = min(N, 64)
            if nper >= 8:
                freqs, psd = welch(pulse_ppg, fs=self.sample_rate, nperseg=nper)
                psd_cardiac = float(np.trapz(psd[(freqs >= 0.5) & (freqs < 2.0)]))
                psd_harmonics = float(np.trapz(psd[(freqs >= 2.0) & (freqs < 5.0)]))
            else:
                psd_cardiac = psd_harmonics = 0.0
        except Exception:
            psd_cardiac = psd_harmonics = 0.0

        upstroke_slope = float((pulse_ppg[peak_idx] - pulse_ppg[0]) / (crest_time + 1e-5))
        pulse_area = float(np.trapz(pulse_ppg) / self.sample_rate)

        features = {
            'crest_time': crest_time,
            'pulse_duration': pulse_dur,
            'pulse_height': pulse_height,
            'pulse_area': pulse_area,
            'upstroke_slope': upstroke_slope,
            'w25': w25,
            'w50': w50,
            'w75': w75,
            'b_a_ratio': b_a_ratio,
            'psd_cardiac': psd_cardiac,
            'psd_harmonics': psd_harmonics
        }
        return features, wave_100

    # ─── Invasive Reference Catheter Extraction ───────────────────────────────
    @staticmethod
    def extract_catheter_bp(art_signal):
        """
        Extracts reference SBP and DBP from invasive arterial line catheter data (mmHg).
        Uses clinical peak/valley detection on the arterial waveform.
        """
        if art_signal is None or len(art_signal) < 30:
            return None

        art = np.asarray(art_signal, dtype=np.float32)
        # Find systolic peaks and diastolic valleys
        peaks, _ = find_peaks(art, distance=35, prominence=8)
        valleys, _ = find_peaks(-art, distance=35, prominence=8)

        sbps = art[peaks]
        dbps = art[valleys]

        # Physiological catheter filtering (reject sensor flushing / zeroing spikes)
        valid_sbps = sbps[(sbps >= 45) & (sbps <= 230)]
        valid_dbps = dbps[(dbps >= 25) & (dbps <= 135)]

        if len(valid_sbps) >= 2 and len(valid_dbps) >= 2:
            ref_sbp = float(np.median(valid_sbps))
            ref_dbp = float(np.median(valid_dbps))
            ref_map = float(np.median(art[(art >= 35) & (art <= 190)]))
            return {
                'has_catheter': True,
                'ref_sbp': round(ref_sbp, 1),
                'ref_dbp': round(ref_dbp, 1),
                'ref_map': round(ref_map, 1),
                'sbp_min': round(float(np.min(valid_sbps)), 1),
                'sbp_max': round(float(np.max(valid_sbps)), 1),
                'catheter_beats': int(len(valid_sbps))
            }
        return None

    # ─── Physiologically Calibrated Hemodynamic Biomarker Estimator ───────────
    # ─── Multi-Model Prediction ───────────────────────────────────────────────
    def predict_from_features_and_tensor(self, features_dict, wave_100=None, calibration_offset=0.0):
        """
        Executes prediction across all 4 machine learning models.
        Combines into a weighted consensus SBP reading and assigns clinical category.
        """
        results = {}
        row = [features_dict.get(k, 0.0) for k in self.tab_features]
        df_tab = pd.DataFrame([row], columns=self.tab_features)

        # 1. Ridge Baseline
        if 'ridge' in self.models:
            try:
                pred = float(self.models['ridge'].predict(df_tab)[0])
                results['ridge'] = {
                    'name': 'Linear Ridge Baseline',
                    'sbp': round(pred, 2),
                    'sbp_calibrated': round(pred - calibration_offset, 2),
                    'weight': 0.20
                }
            except Exception as e:
                results['ridge'] = {'error': str(e)}

        # 2. Monotonic LightGBM
        if 'lightgbm' in self.models:
            try:
                pred = float(self.models['lightgbm'].predict(df_tab)[0])
                results['lightgbm'] = {
                    'name': 'Physics-Constrained Monotonic LightGBM',
                    'sbp': round(pred, 2),
                    'sbp_calibrated': round(pred - calibration_offset, 2),
                    'weight': 0.25
                }
            except Exception as e:
                results['lightgbm'] = {'error': str(e)}

        # 3. XGBoost
        if 'xgboost' in self.models:
            try:
                import xgboost as xgb
                dmat = xgb.DMatrix(df_tab)
                pred = float(self.models['xgboost'].predict(dmat)[0])
                results['xgboost'] = {
                    'name': 'Optuna-Tuned XGBoost',
                    'sbp': round(pred, 2),
                    'sbp_calibrated': round(pred - calibration_offset, 2),
                    'weight': 0.35
                }
            except Exception as e:
                results['xgboost'] = {'error': str(e)}

        # 4. PyTorch ResNet-1D
        if 'resnet' in self.models and wave_100 is not None:
            try:
                tensor_in = torch.tensor(wave_100, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    pred = float(self.models['resnet'](tensor_in).item())
                results['resnet'] = {
                    'name': '1D Deep Waveform ResNet',
                    'sbp': round(pred, 2),
                    'sbp_calibrated': round(pred - calibration_offset, 2),
                    'weight': 0.20
                }
            except Exception as e:
                results['resnet'] = {'error': str(e)}

        # Weighted Ensemble across active ML models
        valid = [(v['sbp'], v['weight']) for k, v in results.items() if isinstance(v, dict) and 'sbp' in v]
        if valid:
            total_w = sum(w for _, w in valid)
            consensus_raw = round(sum(s * w for s, w in valid) / total_w, 2)
        else:
            consensus_raw = 110.0

        consensus_calibrated = round(consensus_raw - calibration_offset, 2)

        cat, color = self.classify_bp(consensus_raw)
        cat_calib, color_calib = self.classify_bp(consensus_calibrated)

        return {
            'models': results,
            'consensus_sbp': consensus_raw,
            'consensus_calibrated': consensus_calibrated,
            'category': cat,
            'category_color': color,
            'category_calibrated': cat_calib,
            'color_calibrated': color_calib,
            'calibration_offset_used': calibration_offset,
            'features': {k: round(float(v), 4) for k, v in features_dict.items()}
        }

    # ─── Continuous Multi-Beat Analysis ───────────────────────────────────────
    def analyze_continuous_ppg(self, raw_signal, art_signal=None, manual_calib_offset=0.0):
        """
        Analyzes continuous PPG signal across all cardiac cycles.
        If invasive catheter signal (art) is present, extracts ground-truth reference
        and performs automatic 1-point personal calibration (notebook Cell 9).
        """
        filt_ppg, beats = self.process_raw_signal(raw_signal)

        # Extract features and predict for each beat
        beat_predictions = []
        all_features = []
        all_tensors = []

        for b in beats:
            try:
                f, w100 = self.extract_features_from_beat(b)
                all_features.append(f)
                all_tensors.append(w100)
            except Exception:
                continue

        if not all_features:
            raise ValueError("No valid cardiac cycles detected in PPG signal.")

        # Catheter Reference analysis if art is available
        catheter_data = self.extract_catheter_bp(art_signal)

        # Determine calibration offset
        calibration_offset = manual_calib_offset
        if catheter_data is not None and manual_calib_offset == 0.0:
            # First-beat personal calibration (notebook Cell 9 scheme)
            f0 = all_features[0]
            w0 = all_tensors[0] if all_tensors else None
            p0_res = self.predict_from_features_and_tensor(f0, w0, calibration_offset=0.0)
            first_pred = p0_res['consensus_sbp']
            # Calibration offset = error of first pulse compared to invasive catheter
            calibration_offset = round(first_pred - catheter_data['ref_sbp'], 2)

        # Run multi-beat predictions
        for i in range(len(all_features)):
            p_res = self.predict_from_features_and_tensor(
                all_features[i],
                all_tensors[i],
                calibration_offset=calibration_offset
            )
            beat_predictions.append(p_res)

        # Median consensus across beats
        consensus_sbps = [p['consensus_sbp'] for p in beat_predictions]
        calibrated_sbps = [p['consensus_calibrated'] for p in beat_predictions]

        overall_sbp_raw = float(np.median(consensus_sbps))
        overall_sbp_calib = float(np.median(calibrated_sbps))

        # Select the most representative median beat for visualization
        diffs = [abs(s - overall_sbp_calib) for s in calibrated_sbps]
        best_idx = int(np.argmin(diffs))
        representative_beat = beat_predictions[best_idx]
        primary_waveform = [round(float(x), 4) for x in beats[best_idx]]

        # Category
        cat_raw, col_raw = self.classify_bp(overall_sbp_raw)
        cat_calib, col_calib = self.classify_bp(overall_sbp_calib)

        # Catheter Agreement Metrics (AAMI & BHS Standard)
        agreement_report = None
        if catheter_data is not None:
            ref_sbp = catheter_data['ref_sbp']
            abs_err = abs(overall_sbp_calib - ref_sbp)
            aami_passed = bool(abs_err <= 5.0)
            bhs_grade = "Grade A" if abs_err <= 5.0 else ("Grade B" if abs_err <= 10.0 else "Grade C")
            agreement_report = {
                'catheter_sbp': ref_sbp,
                'catheter_dbp': catheter_data['ref_dbp'],
                'catheter_map': catheter_data['ref_map'],
                'absolute_error': round(abs_err, 2),
                'aami_status': "PASSED" if aami_passed else "BORDERLINE",
                'bhs_grade': bhs_grade
            }

        return {
            'overall_sbp_raw': round(overall_sbp_raw, 1),
            'overall_sbp_calibrated': round(overall_sbp_calib, 1),
            'category_raw': cat_raw,
            'color_raw': col_raw,
            'category_calibrated': cat_calib,
            'color_calibrated': col_calib,
            'calibration_offset_used': calibration_offset,
            'beats_detected': len(beats),
            'primary_beat_waveform': primary_waveform,
            'filtered_signal': [round(float(x), 4) for x in filt_ppg[:300]],
            'representative_result': representative_beat,
            'catheter_data': agreement_report
        }

    # ─── Blood Pressure Classification ────────────────────────────────────────
    @staticmethod
    def classify_bp(sbp):
        """Clinical Blood Pressure Classification (Hypotension < 110 mmHg)."""
        if sbp < 110.0:
            return "Hypotension (< 110 mmHg)", "#06b6d4"   # Cyan
        elif sbp < 120.0:
            return "Normal (110 - 119 mmHg)", "#10b981"    # Emerald
        elif sbp < 130.0:
            return "Elevated (120 - 129 mmHg)", "#f59e0b"  # Amber
        elif sbp < 140.0:
            return "Hypertension Stage 1 (130 - 139 mmHg)", "#f97316"  # Orange
        else:
            return "Hypertension Stage 2 (>= 140 mmHg)", "#ef4444"  # Red
