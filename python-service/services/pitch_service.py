from utils.audio_loader import get_cached_or_loaded_audio
from utils.extraction_utils import filter_low_confidence
from feature_extraction.pitch import (
    compute_dynamic_params_crepe,
    extract_raw_pitch_crepe,
    smooth_and_segment_crepe,
)

def process_pitch(audio_bytes):

    # 1) Load & (already) trim audio + get URL
    audio, sr, audio_url, error  = get_cached_or_loaded_audio(audio_bytes, sample_rate=16000, return_path=True)
    if error:
        return None, error
    
    # 2) Compute dynamic parameters
    dur_sec = len(audio) / sr
    hop_ms, hop_sec, batch = compute_dynamic_params_crepe(dur_sec)

    # 3) Extract CREPE pitch
    times, pitch, conf = extract_raw_pitch_crepe(audio, hop_ms, batch)

    # 4) Filter out low-confidence frames
    pitch = filter_low_confidence(pitch, conf)
    if pitch is None:
        return None, "No high-confidence pitch found"
    
    # 5) Smooth, replace zeros, find highlight segment
    pitch, highlighted = smooth_and_segment_crepe(pitch, hop_sec)

    result = {
        'x_axis': [round(float(t), 2) for t in times],
        'pitch': pitch.tolist(),
        'highlighted_section': highlighted,
        'sample_rate': sr,
        'audio_url': audio_url
    }

    return result, None