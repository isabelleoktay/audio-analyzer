import numpy as np
from flask import current_app
from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
from feature_extraction.tempo import calculate_dynamic_tempo_beattracker
from utils.data_utils import round_array_to_nearest
from config import *
from utils.resource_monitoring import ResourceMonitor, get_resource_logger


def process_tempo(audio_bytes, session_id=None, file_key="input"):
    """
    Process tempo for the provided audio bytes.
    Uses session_id and file_key to scope cache to a specific file within a session.
    """
    file_logger = get_resource_logger()
    
    # clear cache only for the specific session/file
    clear_cache_if_new_file(audio_bytes)

    audio, sr, audio_url, __, error = load_and_process_audio(audio_bytes, sample_rate=44100)
    if error:
        return None, error

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    audio_duration_sec = len(audio) / sr
    dynamic_tempo, beats, time_axis = calculate_dynamic_tempo_beattracker(audio, sr, audio_duration_sec)
    time_axis = round_array_to_nearest(time_axis, 2)

    monitor.stop()
    stats = monitor.summary(feature_type="tempo")
    print(f"Tempo inference metrics: {stats}")
    file_logger.info(f"Tempo inference metrics: {stats}")

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