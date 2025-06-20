from utils.audio_loader import load_and_process_audio, clear_cache_if_new_file
from feature_extraction.phonation import extract_phonation

def process_phonation(audio_bytes):
    clear_cache_if_new_file(audio_bytes)

    # 1) Load & trim audio + get URL
    audio, sr, audio_url, error = load_and_process_audio(audio_bytes, sample_rate=16000)
    if error:
        return None, error
    
    audio_duration = len(audio) / sr
        
    phonation_classes = extract_phonation(audio)
    transposed_predictions = phonation_classes.T

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