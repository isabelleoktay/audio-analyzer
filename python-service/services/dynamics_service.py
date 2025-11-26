# from utils.audio_loader import clear_cache_if_new_file
# from feature_extraction.dynamics import get_cached_or_calculated_dynamics

# def process_dynamics(audio_bytes):
#     clear_cache_if_new_file(audio_bytes)

#     audio, audio_url, sr, _, smoothed_rms, highlighted_section, _ = get_cached_or_calculated_dynamics(audio_bytes, sample_rate=44100, return_path=True)
#     audio_duration = len(audio) / sr

#     result = {
#         'data': [
#             {
#                 'data': smoothed_rms.tolist(),
#                 'highlighted': highlighted_section,
#                 'label': 'dynamics'
#             }
#         ],
#         'sample_rate': sr,
#         'audio_url': audio_url,
#         'duration': audio_duration 
#     }

#     return result, None

from utils.audio_loader import clear_cache_if_new_file
from feature_extraction.dynamics import get_cached_or_calculated_dynamics

def process_dynamics(audio_bytes, session_id=None, file_key="input"):
    """
    Process dynamics for the provided audio bytes.

    session_id and file_key are optional and will be passed to cache helpers so
    the correct per-session / per-file cache is used.
    """
    # ensure we only clear the cache for the specific session/file (not global)
    clear_cache_if_new_file(audio_bytes, session_id=session_id, file_key=file_key)

    audio, audio_url, sr, _, smoothed_rms, highlighted_section, _ = get_cached_or_calculated_dynamics(
        audio_bytes,
        sample_rate=44100,
        return_path=True,
        session_id=session_id,
        file_key=file_key,
    )
    audio_duration = len(audio) / sr

    result = {
        'data': [
            {
                'data': smoothed_rms.tolist(),
                'highlighted': highlighted_section,
                'label': 'dynamics'
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url,
        'duration': audio_duration 
    }

    return result, None