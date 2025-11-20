import numpy as np 
from utils.smoothing import smooth_data
from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
from feature_extraction.vocal_tone import extract_vocal_tone

def process_vocal_tone(audio_bytes, gender):
    clear_cache_if_new_file(audio_bytes)

    # Load & preprocess audio
    audio, sr, audio_url, audio_path, error = load_and_process_audio(audio_bytes, sample_rate=16000)
    if error:
        return None, error

    audio_duration = len(audio) / sr

    # Run both models (Whisper + CLAP)
    whisper_class_names, whisper_predictions, clap_class_names, clap_predictions = extract_vocal_tone(audio_path, gender)

    smoothed_clap = np.zeros_like(clap_predictions)
    smoothed_whisper = np.zeros_like(whisper_predictions)

    for c in range(len(clap_class_names)):
        smoothed_clap[:, c] = smooth_data(
            clap_predictions[:, c],
            window_size=3,             
            filter_type="median"
        )

    for c in range(len(whisper_class_names)):
        smoothed_whisper[:, c] = smooth_data(
            whisper_predictions[:, c],
            window_size=3,             
            filter_type="median"
        )

    # Transpose predictions so each class has a vector of probabilities over time
    transposed_whisper = smoothed_whisper.T
    transposed_clap = smoothed_clap.T

    result = {
        'data': {
            'CLAP': [
                {
                    'label': label,
                    'data': transposed_clap[i].tolist(),
                }
                for i, label in enumerate(clap_class_names)
            ],
            'Whisper': [
                {
                    'label': label,
                    'data': transposed_whisper[i].tolist(),
                }
                for i, label in enumerate(whisper_class_names)
            ],
        },
        'sample_rate': sr,
        'audio_url': audio_url,
        'duration': audio_duration,
    }

    return result, None
