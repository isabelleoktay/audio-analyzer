import numpy as np

from feature_extraction.dynamics import get_cached_or_calculated_dynamics
from utils.audio_loader import get_user_cache, get_cached_or_loaded_audio
from utils.audio_loader import get_cached_or_loaded_audio
from utils.extraction_utils import filter_low_confidence
from feature_extraction.pitch import CrepePitchExtractor, LibrosaPitchExtractor
from feature_extraction.pitch_utils import (
    compute_dynamic_params_crepe,
    smooth_and_segment_crepe,
    filter_pitch_by_rms
)
from flask import session

def get_cached_or_calculated_pitch(audio, sr, audio_url, method="crepe"):
    audio_cache = get_user_cache()

    if audio_cache["pitch"]["pitch"] is not None and audio_cache["pitch"]["smoothed_pitch"] is not None:
        # Convert cached lists back to numpy arrays when returning
        cached_audio = np.array(audio_cache["pitch"]["audio"]) if audio_cache["pitch"]["audio"] is not None else None
        cached_pitch = np.array(audio_cache["pitch"]["pitch"]) if audio_cache["pitch"]["pitch"] is not None else None
        cached_smoothed = np.array(audio_cache["pitch"]["smoothed_pitch"]) if audio_cache["pitch"]["smoothed_pitch"] is not None else None
        
        return (cached_audio, audio_cache["pitch"]["audio_url"], audio_cache["pitch"]["sr"], 
                cached_pitch, cached_smoothed,
                audio_cache["pitch"]["highlighted_section"], audio_cache["pitch"]["x_axis"], 
                audio_cache["pitch"]["hop_sec_duration"])
    
    # Get dynamics to filter out low-energy segments
    _, _, _, rms, _, _, _ = get_cached_or_calculated_dynamics(audio, sr, audio_url)

    # 2) Compute dynamic parameters
    dur_sec = len(audio) / sr
    hop_ms, hop_sec, batch = compute_dynamic_params_crepe(dur_sec)

    # 3) Select pitch extraction method
    if method == "crepe":
        extractor = CrepePitchExtractor()
        times, pitch, conf = extractor.extract_pitch(audio, hop_ms=hop_ms, batch=batch)
    elif method == "librosa":
        hop_length = int(hop_sec * sr)
        extractor = LibrosaPitchExtractor()
        times, pitch, conf = extractor.extract_pitch(audio, sr=sr, hop_length=hop_length)
    else:
        return None, f"Unknown method: {method}"

    # 4) Filter out low-confidence frames
    thresh = (0.1 * np.max(conf))
    pitch = filter_low_confidence(pitch, conf, thresh=thresh)
    if pitch is None:
        return None, "No high-confidence pitch found"
    
    pitch = filter_pitch_by_rms(pitch, rms)
    
    # 5) Smooth, replace zeros, find highlight segment
    smoothed_pitch, highlighted = smooth_and_segment_crepe(pitch, hop_sec)
    x_axis = [round(float(t), 2) for t in times]

    audio_cache["pitch"]["audio"] = audio.tolist()
    audio_cache["pitch"]["sr"] = sr
    audio_cache["pitch"]["pitch"] = pitch.tolist()
    audio_cache["pitch"]["smoothed_pitch"] = smoothed_pitch.tolist()
    audio_cache["pitch"]["highlighted_section"] = highlighted
    audio_cache["pitch"]["x_axis"] = x_axis
    audio_cache["pitch"]["audio_url"] = audio_url
    audio_cache["pitch"]["hop_sec_duration"] = hop_sec

    # Update the session cache
    session['audio_cache'] = audio_cache

    return audio, audio_url, sr, pitch, smoothed_pitch, highlighted, x_axis, hop_sec

def process_pitch(audio_bytes, method="crepe"):

    # 1) Load & (already) trim audio + get URL
    audio, sr, audio_url, error  = get_cached_or_loaded_audio(audio_bytes, sample_rate=16000, return_path=True)
    if error:
        return None, error
    
    _, audio_url, sr, _, smoothed_pitch, highlighted, _, _ = get_cached_or_calculated_pitch(audio, sr, audio_url, method=method)

    result = {
        'data': [
            {
                'data': smoothed_pitch.tolist(),
                'highlighted': highlighted,
                'label': 'pitch'
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url
    }

    return result, None
    
