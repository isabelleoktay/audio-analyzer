
# from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
# from feature_extraction.tempo import calculate_dynamic_tempo_essentia, calculate_dynamic_tempo_beattracker
# from utils.data_utils import round_array_to_nearest
# from config import *

# def process_tempo(audio_bytes):
#     clear_cache_if_new_file(audio_bytes)
#     audio, sr, audio_url, __, error = load_and_process_audio(audio_bytes, sample_rate=44100)

#     if error:
#         return None, None, None, error

#     audio_duration_sec = len(audio) / sr
#     dynamic_tempo, _, time_axis = calculate_dynamic_tempo_beattracker(audio, sr, audio_duration_sec)
#     time_axis = round_array_to_nearest(time_axis, 2)

#     result = {
#         'data': [
#             {
#                 'data': dynamic_tempo,
#                 'label': 'tempo'
#             }
#         ],
#         'sample_rate': sr,
#         'audio_url': audio_url,
#         'duration': audio_duration_sec
#     }

#     return result, None

from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
from feature_extraction.tempo import calculate_dynamic_tempo_beattracker
from utils.data_utils import round_array_to_nearest
from config import *
from flask import current_app
import numpy as np

def process_tempo(audio_bytes, session_id=None, file_key="input"):
    """
    Process tempo for the provided audio bytes.
    Uses session_id and file_key to scope cache to a specific file within a session.
    """
    # clear cache only for the specific session/file
    clear_cache_if_new_file(audio_bytes)

    audio, sr, audio_url, __, error = load_and_process_audio(audio_bytes, sample_rate=44100)
    if error:
        return None, error

    audio_duration_sec = len(audio) / sr
    dynamic_tempo, beats, time_axis = calculate_dynamic_tempo_beattracker(audio, sr, audio_duration_sec)
    time_axis = round_array_to_nearest(time_axis, 2)

    result = {
        'data': [
            {
                'data': dynamic_tempo,
                'label': 'tempo',
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url,
        'duration': audio_duration_sec
    }

    return result, None