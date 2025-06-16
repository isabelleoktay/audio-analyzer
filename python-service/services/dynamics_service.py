from utils.audio_loader import get_cached_or_loaded_audio
from utils.smoothing import smooth_data
# from feature_extraction.variability import calculate_high_variability_sections
from feature_extraction.dynamics import get_cached_or_calculated_dynamics
# import librosa
from config import *

def process_dynamics(audio_bytes):
    audio, sr, audio_url, error = get_cached_or_loaded_audio(audio_bytes)

    if error:
        return None, error
    
    _, audio_url, sr, _, smoothed_rms, highlighted_section, _ = get_cached_or_calculated_dynamics(audio, sr, audio_url)

    # rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    # data_len = rms.shape[1]
    # time = [round(i * HOP_LENGTH / sr, 2) for i in range(data_len)]

    # window_size = max(1, int(data_len * WINDOW_PERCENTAGE))     
    # hop_size = max(1, int(data_len * HOP_PERCENTAGE))         
    # window_size = min(window_size, data_len)       

    # audio_duration_sec = len(audio) / sr
    # highlighted_section = calculate_high_variability_sections(
    #     rms[0], window_size=window_size, hop_size=hop_size, 
    #     hop_duration_sec=HOP_LENGTH / sr, audio_duration_sec=audio_duration_sec, is_pitch=False
    # )

    # rms = smooth_data(rms[0], filter_type='mean', window_percentage=0.1)

    result = {
        'data': [
            {
                'data': smoothed_rms.tolist(),
                'highlighted': highlighted_section,
                'label': 'dynamics'
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url
    }

    return result, None