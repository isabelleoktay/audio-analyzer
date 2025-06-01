
from utils.audio_loader import get_cached_or_loaded_audio
from tempo import calculate_dynamic_tempo_essentia
from utils.data_utils import round_array_to_nearest
from config import *
def process_tempo(audio_bytes):
    audio, sr, audio_url, error = get_cached_or_loaded_audio(audio_bytes)

    if error:
        return None, error

    audio_duration_sec = len(audio) / sr
    dynamic_tempo, global_tempo, time_axis = calculate_dynamic_tempo_essentia(audio)
    data_len = len(dynamic_tempo)

    window_size = max(1, int(data_len * WINDOW_PERCENTAGE))     
    hop_size = max(1, int(data_len * HOP_PERCENTAGE))     
    window_size = min(window_size, data_len)    

    # variability = calculate_variability(dynamic_tempo, window_size=window_size, hop_size=hop_size)
    # highlighted_section = get_high_variability_section_with_time_axis(variability, time_axis, percentage=SEGMENT_PERCENTAGE)

    time_axis = round_array_to_nearest(time_axis, 2)

    result = {
        'x_axis': time_axis,
        'tempo': dynamic_tempo,
        'sample_rate': sr,
        'audio_url': audio_url,
        # 'highlighted_section': highlighted_section
    }

    return result, None