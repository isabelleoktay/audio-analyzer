from utils.audio_loader import get_cached_or_loaded_audio
from services.pitch_service import get_cached_or_calculated_pitch
from feature_extraction.vibrato_utils import frequencies_to_piano_notes
from feature_extraction.vibrato import (
    extract_vibrato, 
    calculate_vibrato_parameters, 
    get_sections_with_vibrato
)

def process_vibrato(audio_bytes, method="crepe"):

    # 1) Load & trim audio + get URL
    audio, sr, audio_url, error = get_cached_or_loaded_audio(audio_bytes, sample_rate=16000, return_path=True)
    if error:
        return None, error
    
    audio, audio_url, sr, pitches, smoothed_pitches, _, x_axis, hop_sec_duration = get_cached_or_calculated_pitch(audio, sr, audio_url, method=method)

    pitch_piano_notes = frequencies_to_piano_notes(smoothed_pitches)
    vibrato_data = extract_vibrato(pitches, smoothed_pitches, pitch_piano_notes)
    vibrato_extents, vibrato_rates = calculate_vibrato_parameters(vibrato_data, x_axis, sr)

    highlighted = get_sections_with_vibrato(vibrato_extents, vibrato_rates, x_axis, hop_sec_duration)

    result = {
        'data': [
            {
                'data': smoothed_pitches.tolist(),
                'highlighted': highlighted,
                'label': 'vibrato'
            },
            {
                'data': vibrato_extents.tolist(),
                'label': 'extents'
            },
            {
                'data': vibrato_rates.tolist(),
                'label': 'rates'
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url
    }

    return result, None



