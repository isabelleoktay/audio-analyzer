from utils.audio_loader import clear_cache_if_new_file
from feature_extraction.dynamics import get_cached_or_calculated_dynamics
from utils.resource_monitoring import ResourceMonitor, get_resource_logger

file_logger = get_resource_logger()

def process_dynamics(audio_bytes):
    clear_cache_if_new_file(audio_bytes)

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    audio, audio_url, sr, _, smoothed_rms, highlighted_section, _ = get_cached_or_calculated_dynamics(audio_bytes, sample_rate=44100, return_path=True)
    audio_duration = len(audio) / sr

    monitor.stop()
    stats = monitor.summary(feature_type="dynamics")
    print(f"Dynamics inference metrics: {stats}")
    file_logger.info(f"Dynamics inference metrics: {stats}")

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