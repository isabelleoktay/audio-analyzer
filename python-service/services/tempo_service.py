
from utils.audio_loader import get_cached_or_loaded_audio
from feature_extraction.tempo import calculate_dynamic_tempo_essentia
from utils.data_utils import round_array_to_nearest
from config import *
def process_tempo(audio_bytes):
    audio, sr, audio_url, error = get_cached_or_loaded_audio(audio_bytes)

    if error:
        return None, error

    audio_duration_sec = len(audio) / sr
    dynamic_tempo, _, time_axis = calculate_dynamic_tempo_essentia(audio, sr, audio_duration_sec)
    time_axis = round_array_to_nearest(time_axis, 2)

    result = {
        'data': [
            {
                'data': dynamic_tempo,
                'label': 'tempo'
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url
    }

    return result, None