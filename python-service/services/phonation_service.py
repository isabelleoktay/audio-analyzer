from utils.audio_loader import get_cached_or_loaded_audio
from feature_extraction.phonation import extract_phonation

def process_phonation(audio_bytes):

    # 1) Load & trim audio + get URL
    audio, sr, audio_url, error = get_cached_or_loaded_audio(audio_bytes, sample_rate=16000, return_path=True)
    if error:
        return None, error
        
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
        'audio_url': audio_url
    }

    return result, None