import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .inference import BPInferenceService

inference_service = BPInferenceService()


def index_view(request):
    """Renders the main diagnostic dashboard."""
    return render(request, 'index.html', {})


@csrf_exempt
def api_sample(request):
    """Health check and status API."""
    return JsonResponse({
        'status': 'online',
        'models_active': list(inference_service.models.keys()),
        'sampling_rate_hz': inference_service.sample_rate
    })


@csrf_exempt
def api_predict(request):
    """
    Unified API for Blood Pressure prediction and Invasive Catheter validation.
    Accepts:
      signal: List[float] (100 Hz continuous PPG samples)
      art: Optional[List[float]] (Invasive arterial catheter samples from VitalDB)
      calibration_offset: Optional[float] (manual baseline offset in mmHg)
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
        else:
            data = request.POST.dict()
            if 'signal' in data and isinstance(data['signal'], str):
                try:
                    data['signal'] = json.loads(data['signal'])
                except Exception:
                    data['signal'] = [float(x.strip()) for x in data['signal'].replace('\n', ',').split(',') if x.strip()]
            if 'art' in data and isinstance(data['art'], str):
                try:
                    data['art'] = json.loads(data['art'])
                except Exception:
                    data['art'] = [float(x.strip()) for x in data['art'].replace('\n', ',').split(',') if x.strip()]

        raw_signal = data.get('signal', [])
        art_signal = data.get('art', None)
        calib_offset = float(data.get('calibration_offset', 0.0) or 0.0)

        if isinstance(raw_signal, str):
            raw_signal = [float(x.strip()) for x in raw_signal.replace('\n', ',').split(',') if x.strip()]
        if isinstance(art_signal, str):
            art_signal = [float(x.strip()) for x in art_signal.replace('\n', ',').split(',') if x.strip()]

        if not raw_signal or len(raw_signal) < 20:
            return JsonResponse({'error': 'Input signal must contain at least 20 numeric samples.'}, status=400)

        # Run continuous multi-beat analysis
        analysis = inference_service.analyze_continuous_ppg(
            raw_signal=raw_signal,
            art_signal=art_signal,
            manual_calib_offset=calib_offset
        )

        rep = analysis['representative_result']
        has_calib = (abs(analysis['calibration_offset_used']) > 0.01)

        # Structure response to seamlessly power the frontend report
        response_payload = {
            'success': True,
            'raw_sample_count': len(raw_signal),
            'beats_detected': analysis['beats_detected'],
            'filtered_signal': analysis['filtered_signal'],
            'beat_waveform': analysis['primary_beat_waveform'],
            'catheter_data': analysis['catheter_data'],
            'results': {
                'consensus_sbp': analysis['overall_sbp_raw'],
                'consensus_calibrated': analysis['overall_sbp_calibrated'],
                'category': analysis['category_raw'],
                'category_color': analysis['color_raw'],
                'category_calibrated': analysis['category_calibrated'],
                'color_calibrated': analysis['color_calibrated'],
                'calibration_offset_used': analysis['calibration_offset_used'],
                'models': rep['models'],
                'features': rep['features'],
                'catheter_report': analysis['catheter_data']
            }
        }

        return JsonResponse(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
