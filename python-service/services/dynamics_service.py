
from utils.audio_loader import get_cached_or_loaded_audio
from utils.smoothing import smooth_data
from variability import calculate_variability, get_high_variability_section
import librosa
from config import *
def process_dynamics(audio_bytes):
    audio, sr, audio_url, error = get_cached_or_loaded_audio(audio_bytes)

    if error:
        return None, error

    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    data_len = rms.shape[1]
    time = [round(i * HOP_LENGTH / sr, 2) for i in range(data_len)]

    window_size = max(1, int(data_len * WINDOW_PERCENTAGE))     
    hop_size = max(1, int(data_len * HOP_PERCENTAGE))         
    window_size = min(window_size, data_len)       

    audio_duration_sec = len(audio) / sr
    variability = calculate_variability(rms[0], window_size=window_size, hop_size=hop_size)
    highlighted_section = get_high_variability_section(variability, HOP_LENGTH, sr, percentage=SEGMENT_PERCENTAGE, audio_duration_sec=audio_duration_sec)

    rms = smooth_data(rms[0], filter_type='mean', window_percentage=0.1)

    result = {
        'x_axis': time,
        'dynamics': rms.tolist(),
        'sample_rate': sr,
        'highlighted_section': highlighted_section,
        'audio_url': audio_url
    }

    return result, None