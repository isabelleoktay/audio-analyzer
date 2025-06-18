from utils.audio_loader import clear_cache_if_new_file
from feature_extraction.dynamics import get_cached_or_calculated_dynamics

def process_dynamics(audio_bytes):
    clear_cache_if_new_file(audio_bytes)

    _, audio_url, sr, _, smoothed_rms, highlighted_section, _ = get_cached_or_calculated_dynamics(audio_bytes, sample_rate=44100, return_path=True)

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