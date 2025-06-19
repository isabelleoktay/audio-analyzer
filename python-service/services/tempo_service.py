
from utils.audio_loader import get_user_cache, get_user_id_from_token, update_user_cache, load_and_process_audio, clear_cache_if_new_file
from feature_extraction.tempo import calculate_dynamic_tempo_essentia
from utils.data_utils import round_array_to_nearest
from config import *
def process_tempo(audio_bytes):
    clear_cache_if_new_file(audio_bytes)
    audio, sr, audio_url, error = load_and_process_audio(audio_bytes, sample_rate=44100)

    if error:
        return None, None, None, error

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
        'audio_url': audio_url,
        'duration': audio_duration_sec
    }

    return result, None