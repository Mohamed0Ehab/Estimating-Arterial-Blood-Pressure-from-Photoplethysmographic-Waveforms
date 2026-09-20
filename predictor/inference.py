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

# Model Architecture for HemodynamicResNet1D
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


class BPInferenceService:
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
        self.tab_features = [
            "crest_time", "pulse_duration", "pulse_height", "pulse_area", 
            "upstroke_slope", "w25", "w50", "w75", "b_a_ratio", 
            "psd_cardiac", "psd_harmonics"
        ]
        
        # Load config if present
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

        # Butterworth bandpass filter coefficients (0.5 to 8.0 Hz at 100 Hz sampling rate)
        nyq = self.sample_rate / 2.0
        self.b, self.a = butter(4, [0.5 / nyq, 8.0 / nyq], btype='band')

        # Load models
        self.models = {}
        self._load_models()
        self.initialized = True

    def _load_models(self):
        # 1. Ridge model
        ridge_path = os.path.join(MODELS_DIR, 'ridge.pkl')
        if os.path.exists(ridge_path):
            try:
                import joblib
                self.models['ridge'] = joblib.load(ridge_path)
                print("[BPInferenceService] Loaded Ridge model with joblib.")
            except Exception:
                try:
                    with open(ridge_path, 'rb') as f:
                        self.models['ridge'] = pickle.load(f)
                    print("[BPInferenceService] Loaded Ridge model with pickle.")
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

        # 4. PyTorch 1D ResNet
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

    def filter_signal(self, raw_wave):
        """Applies zero-phase 4th order Butterworth bandpass filter."""
        wave = np.asarray(raw_wave, dtype=np.float32)
        if len(wave) < 15:
            return wave
        try:
            return filtfilt(self.b, self.a, wave)
        except Exception:
            return wave

    def extract_features_from_beat(self, pulse_ppg):
        """Extracts 11 hemodynamic features and 100-pt tensor from a single pulse."""
        pulse_ppg = np.asarray(pulse_ppg, dtype=np.float32)
        N = len(pulse_ppg)
        if N < 5:
            raise ValueError("Pulse beat is too short for feature extraction.")

        # Fixed-length 1D waveform tensor (100 points) for ResNet
        t_orig = np.linspace(0, 1, N)
        t_fixed = np.linspace(0, 1, self.fixed_len)
        wave_100 = np.interp(t_fixed, t_orig, pulse_ppg).astype(np.float32)

        # Tabular features
        peak_idx = int(np.argmax(pulse_ppg))
        pulse_height = float(np.ptp(pulse_ppg))
        pulse_min = float(np.min(pulse_ppg))
        crest_time = float(peak_idx / self.sample_rate)
        pulse_dur = float(N / self.sample_rate)

        w25 = float(np.sum(pulse_ppg >= (pulse_min + 0.25 * pulse_height)) / self.sample_rate)
        w50 = float(np.sum(pulse_ppg >= (pulse_min + 0.50 * pulse_height)) / self.sample_rate)
        w75 = float(np.sum(pulse_ppg >= (pulse_min + 0.75 * pulse_height)) / self.sample_rate)

        # APPG 2nd derivative
        try:
            wl = min(9, N if N % 2 != 0 else N - 1)
            if wl >= 5:
                d2 = savgol_filter(pulse_ppg, window_length=wl, polyorder=min(3, wl - 1), deriv=2)
                b_a_ratio = float(np.ptp(d2) / (np.std(d2) + 1e-5))
            else:
                b_a_ratio = 1.0
        except Exception:
            b_a_ratio = 1.0

        # Welch PSD
        try:
            nper = min(N, 64)
            if nper >= 8:
                freqs, psd = welch(pulse_ppg, fs=self.sample_rate, nperseg=nper)
                psd_cardiac = float(np.trapz(psd[(freqs >= 0.5) & (freqs < 2.0)]))
                psd_harmonics = float(np.trapz(psd[(freqs >= 2.0) & (freqs < 5.0)]))
            else:
                psd_cardiac = 0.0
                psd_harmonics = 0.0
        except Exception:
            psd_cardiac = 0.0
            psd_harmonics = 0.0

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

    def process_raw_signal(self, raw_signal):
        """
        Takes raw continuous PPG signal, filters, detects beats, checks SQI,
        and returns list of valid beat pulses and their features.
        """
        raw_signal = np.asarray(raw_signal, dtype=np.float32)
        if len(raw_signal) < 20:
            raise ValueError("Input signal is too short (needs at least 20 samples).")

        # 1. Filter and normalize to [0, 1]
        raw_filt = self.filter_signal(raw_signal)
        ptp = np.ptp(raw_filt)
        if ptp > 1e-6:
            filt_ppg = (raw_filt - np.min(raw_filt)) / (ptp + 1e-6)
        else:
            filt_ppg = raw_filt

        # 2. Slice beats from valley to valley if enough data, or treat as single beat
        beats = []
        valleys, _ = find_peaks(-filt_ppg, distance=int(self.sample_rate * 0.4), prominence=0.10)
        
        if len(valleys) >= 2:
            for i in range(len(valleys) - 1):
                v_start = valleys[i]
                v_end = valleys[i + 1]
                pulse_ppg = filt_ppg[v_start:v_end]
                N = len(pulse_ppg)
                # SQI check
                if N >= int(self.sample_rate * 0.35):
                    # Skewness check: physiological arterial pulses have positive skew
                    s = skew(pulse_ppg)
                    dyn_range = np.ptp(pulse_ppg)
                    if (s > 0.0 or dyn_range > 0.1):
                        beats.append(pulse_ppg)
        
        # If no valley pair found (e.g. user provided a single pulse cycle), use the whole wave
        if len(beats) == 0:
            beats.append(filt_ppg)

        return filt_ppg, beats

    def predict_from_features_and_tensor(self, features_dict, wave_100=None, calibration_offset=0.0):
        """
        Executes predictions across all loaded models given the extracted features
        and 100-pt waveform tensor.
        """
        results = {}

        # Prepare tabular DataFrame ordered according to tab_features
        row = [features_dict.get(k, 0.0) for k in self.tab_features]
        df_tab = pd.DataFrame([row], columns=self.tab_features)

        # 1. Ridge Baseline
        if 'ridge' in self.models:
            try:
                pred = float(self.models['ridge'].predict(df_tab)[0])
                results['ridge'] = {
                    'name': 'Linear Ridge Baseline',
                    'sbp': round(pred, 2),
                    'sbp_calibrated': round(pred - calibration_offset, 2)
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
                    'sbp_calibrated': round(pred - calibration_offset, 2)
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
                    'sbp_calibrated': round(pred - calibration_offset, 2)
                }
            except Exception as e:
                results['xgboost'] = {'error': str(e)}

        # 4. PyTorch ResNet-1D
        if 'resnet' in self.models and wave_100 is not None:
            try:
                tensor_in = torch.tensor(wave_100, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 100)
                with torch.no_grad():
                    pred = float(self.models['resnet'](tensor_in).item())
                results['resnet'] = {
                    'name': '1D Deep Waveform ResNet',
                    'sbp': round(pred, 2),
                    'sbp_calibrated': round(pred - calibration_offset, 2)
                }
            except Exception as e:
                results['resnet'] = {'error': str(e)}

        # Calculate Consensus / Best Ensemble estimate (prioritize ResNet or LightGBM/XGBoost)
        valid_sbps = [v['sbp'] for v in results.values() if isinstance(v, dict) and 'sbp' in v]
        valid_calibrated = [v['sbp_calibrated'] for v in results.values() if isinstance(v, dict) and 'sbp_calibrated' in v]

        avg_sbp = round(float(np.mean(valid_sbps)), 2) if valid_sbps else 120.0
        avg_calibrated = round(float(np.mean(valid_calibrated)), 2) if valid_calibrated else 120.0

        # Clinical Category (AHA / ACC guidelines)
        cat, color = self.classify_bp(avg_sbp)
        cat_calib, color_calib = self.classify_bp(avg_calibrated)

        return {
            'models': results,
            'consensus_sbp': avg_sbp,
            'consensus_calibrated': avg_calibrated,
            'category': cat,
            'category_color': color,
            'category_calibrated': cat_calib,
            'calibration_offset_used': calibration_offset,
            'features': {k: round(v, 4) for k, v in features_dict.items()}
        }

    @staticmethod
    def classify_bp(sbp):
        """Classifies Systolic Blood Pressure according to AHA/ACC guidelines."""
        if sbp < 120:
            return "Normal (< 120 mmHg)", "#10b981"  # Emerald Green
        elif 120 <= sbp < 130:
            return "Elevated (120 - 129 mmHg)", "#f59e0b"  # Amber
        elif 130 <= sbp < 140:
            return "Hypertension Stage 1 (130 - 139 mmHg)", "#f97316"  # Orange
        else:
            return "Hypertension Stage 2 (>= 140 mmHg)", "#ef4444"  # Red
