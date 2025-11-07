from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
from feature_extraction.vocal_tone import extract_vocal_tone

def process_vocal_tone(audio_bytes, gender):
    clear_cache_if_new_file(audio_bytes)

    # Load & preprocess audio
    audio, sr, audio_url, error = load_and_process_audio(audio_bytes, sample_rate=16000)
    if error:
        return None, error

    audio_duration = len(audio) / sr

    # Run both models (Whisper + CLAP)
    whisper_class_names, whisper_predictions, clap_class_names, clap_predictions = extract_vocal_tone(audio_url, gender)

    # Transpose predictions so each class has a vector of probabilities over time
    transposed_whisper = whisper_predictions.T
    transposed_clap = clap_predictions.T

    result = {
        "vocalTone": {
            "CLAP": {
                "audioUrl": audio_url,
                "duration": audio_duration,
                "data": [
                    {
                        "label": label,
                        "data": transposed_clap[i].tolist(),
                    }
                    for i, label in enumerate(clap_class_names)
                ],
            },
            "Whisper": {
                "audioUrl": audio_url,
                "duration": audio_duration,
                "data": [
                    {
                        "label": label,
                        "data": transposed_whisper[i].tolist(),
                    }
                    for i, label in enumerate(whisper_class_names)
                ],
            },
        }
    }

    return result, None
