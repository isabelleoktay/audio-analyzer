from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
from feature_extraction.phonation import extract_phonation
from utils.resource_monitoring import ResourceMonitor, get_resource_logger

file_logger = get_resource_logger()

def process_phonation(audio_bytes):
    clear_cache_if_new_file(audio_bytes)

    # 1) Load & trim audio + get URL
    audio, sr, audio_url, __, error = load_and_process_audio(audio_bytes, sample_rate=16000)
    if error:
        return None, error

    audio_duration = len(audio) / sr

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    phonation_classes = extract_phonation(audio)
    transposed_predictions = phonation_classes.T

    monitor.stop()
    stats = monitor.summary(feature_type="phonation")
    print(f"Phonation inference metrics: {stats}")
    file_logger.info(f"Phonation inference metrics: {stats}")

    data = []
    index = 0
    class_names = ['breathy', 'flow', 'neutral', 'pressed']

    for prediction in transposed_predictions:
        data.append({
            'data': prediction.tolist(),
            'label': class_names[index]
        })

        index += 1

    result = {
        'data': data,
        'sample_rate': sr,
        'audio_url': audio_url,
        'duration': audio_duration
    }

    return result, None