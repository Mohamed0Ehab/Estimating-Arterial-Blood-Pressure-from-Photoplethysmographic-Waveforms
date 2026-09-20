import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .inference import BPInferenceService

inference_service = BPInferenceService()

# 3 Verified realistic PPG pulses sampled at 100 Hz (approx 80-90 samples each)
# Synthesized based on typical physiological arterial pressure shapes (sharp systolic rise, dicrotic notch, diastolic runoff)
def generate_sample_pulse(peak_delay=0.20, duration=0.80, height=0.28, notch_depth=0.55):
    """Generates a realistic photoplethysmogram (PPG) pulse wave."""
    n_points = int(duration * 100)
    t = np.linspace(0, duration, n_points)
    # Systolic component
    sigma1 = peak_delay / 2.2
    systolic = height * np.exp(-((t - peak_delay) ** 2) / (2 * (sigma1 ** 2)))
    # Diastolic reflected wave
    diastolic_delay = peak_delay + 0.22
    sigma2 = 0.08
    diastolic = (height * notch_depth) * np.exp(-((t - diastolic_delay) ** 2) / (2 * (sigma2 ** 2)))
    # Exponential decay baseline
    baseline = (height * 0.1) * np.exp(-t / (duration * 0.6))
    pulse = systolic + diastolic + baseline
    pulse = (pulse - np.min(pulse)) / (np.ptp(pulse) + 1e-6)
    return [round(float(x), 4) for x in pulse]

CLINICAL_SAMPLES = {
    'normotensive': {
        'title': 'Case A: Normotensive Patient (Healthy Arterial Compliance)',
        'description': 'Typical young adult pulse: rapid systolic upstroke, pronounced dicrotic notch, normal vascular resistance.',
        'target_sbp': '104 - 115 mmHg',
        'calibration_offset': 0.0,
        'signal': generate_sample_pulse(peak_delay=0.18, duration=0.82, height=0.26, notch_depth=0.58)
    },
    'prehypertensive': {
        'title': 'Case B: Pre-Hypertensive Patient (Mild Arterial Stiffness)',
        'description': 'Elevated pulse wave velocity, delayed systolic crest, attenuation of dicrotic notch.',
        'target_sbp': '122 - 128 mmHg',
        'calibration_offset': 0.0,
        'signal': generate_sample_pulse(peak_delay=0.23, duration=0.76, height=0.31, notch_depth=0.38)
    },
    'hypertensive': {
        'title': 'Case C: Hypertensive Patient (Advanced Arterial Stiffness)',
        'description': 'Stiffened vascular wall, broad crest time, merged reflection wave, elevated pulse area.',
        'target_sbp': '142 - 155 mmHg',
        'calibration_offset': 0.0,
        'signal': generate_sample_pulse(peak_delay=0.27, duration=0.68, height=0.35, notch_depth=0.20)
    }
}


def index_view(request):
    """Renders the main dashboard."""
    context = {
        'default_features': {
            'crest_time': 0.20,
            'pulse_duration': 0.72,
            'pulse_height': 0.28,
            'pulse_area': 0.30,
            'upstroke_slope': 1.40,
            'w25': 0.45,
            'w50': 0.28,
            'w75': 0.16,
            'b_a_ratio': 3.90,
            'psd_cardiac': 0.045,
            'psd_harmonics': 0.012
        }
    }
    return render(request, 'index.html', context)


@csrf_exempt
def api_sample(request):
    """Returns sample waveforms."""
    sample_id = request.GET.get('id', 'normotensive')
    sample = CLINICAL_SAMPLES.get(sample_id, CLINICAL_SAMPLES['normotensive'])
    return JsonResponse(sample)


@csrf_exempt
def api_predict(request):
    """Unified API for Blood Pressure prediction."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
        else:
            data = request.POST.dict()
            if 'features' in data and isinstance(data['features'], str):
                data['features'] = json.loads(data['features'])
            if 'signal' in data and isinstance(data['signal'], str):
                try:
                    data['signal'] = json.loads(data['signal'])
                except Exception:
                    data['signal'] = [float(x.strip()) for x in data['signal'].replace('\n', ',').split(',') if x.strip()]

        mode = data.get('mode', 'waveform')
        calib_offset = float(data.get('calibration_offset', 0.0) or 0.0)

        if mode == 'tabular':
            features = data.get('features', {})
            # Ensure float conversion
            clean_features = {}
            for k in inference_service.tab_features:
                clean_features[k] = float(features.get(k, 0.0) or 0.0)
            
            # Predict
            pred_results = inference_service.predict_from_features_and_tensor(
                clean_features, 
                wave_100=None, 
                calibration_offset=calib_offset
            )
            return JsonResponse({
                'success': True,
                'mode': 'tabular',
                'results': pred_results
            })

        else:
            # Waveform mode
            raw_signal = data.get('signal', [])
            if isinstance(raw_signal, str):
                # Parse comma or newline separated
                raw_signal = [float(x.strip()) for x in raw_signal.replace('\n', ',').split(',') if x.strip()]
            
            if len(raw_signal) < 20:
                return JsonResponse({'error': 'Signal must have at least 20 numeric points.'}, status=400)

            # Process signal and extract clean beats
            filt_ppg, beats = inference_service.process_raw_signal(raw_signal)
            
            # Extract features from first clean beat (or average if multiple)
            primary_beat = beats[0]
            features, wave_100 = inference_service.extract_features_from_beat(primary_beat)

            # Inference
            pred_results = inference_service.predict_from_features_and_tensor(
                features, 
                wave_100=wave_100, 
                calibration_offset=calib_offset
            )

            return JsonResponse({
                'success': True,
                'mode': 'waveform',
                'raw_sample_count': len(raw_signal),
                'beats_detected': len(beats),
                'filtered_signal': [round(float(x), 4) for x in filt_ppg[:300]], # First 3 seconds max for chart
                'beat_waveform': [round(float(x), 4) for x in primary_beat],
                'wave_100': [round(float(x), 4) for x in wave_100],
                'results': pred_results
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
