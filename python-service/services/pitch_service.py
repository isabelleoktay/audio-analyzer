import numpy as np
from flask import current_app
import os

from feature_extraction.dynamics import get_cached_or_calculated_dynamics
from utils.audio_loader import get_user_cache, get_user_id_from_token, update_user_cache, load_and_process_audio, clear_cache_if_new_file
from utils.extraction_utils import filter_low_confidence
from feature_extraction.pitch import CrepePitchExtractor, LibrosaPitchExtractor
from feature_extraction.pitch_utils import (
    compute_dynamic_params_crepe,
    smooth_and_segment_crepe,
    filter_pitch_by_rms
)

def get_cached_or_calculated_pitch(audio_bytes, sample_rate=16000, method="crepe"):
    # Clear cache if new file
    clear_cache_if_new_file(audio_bytes)
    
    # Check cache first
    audio_cache = get_user_cache()
    user_id = get_user_id_from_token()
    
    if (audio_cache and audio_cache["pitch"]["pitch"] is not None and 
        audio_cache["pitch"]["sr"] == sample_rate and audio_cache["pitch"]["audio_url"] is not None):

        audio_url = audio_cache["pitch"]["audio_url"]
        if '/audio/' in audio_url:
            filename = audio_url.split('/audio/')[-1]
            file_path = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
            
            if not os.path.exists(file_path):
                print(f"Cached audio file {file_path} no longer exists, recalculating...")
            else:
                print("Cache hit for pitch")
                # Convert cached lists back to numpy arrays when returning
                cached_audio = np.array(audio_cache["pitch"]["audio"]) if audio_cache["pitch"]["audio"] is not None else None
                cached_pitch = np.array(audio_cache["pitch"]["pitch"]) if audio_cache["pitch"]["pitch"] is not None else None
                cached_smoothed = np.array(audio_cache["pitch"]["smoothed_pitch"]) if audio_cache["pitch"]["smoothed_pitch"] is not None else None
                
                return (cached_audio, audio_cache["pitch"]["audio_url"], audio_cache["pitch"]["sr"], 
                        cached_pitch, cached_smoothed,
                        audio_cache["pitch"]["highlighted_section"], audio_cache["pitch"]["x_axis"], 
                        audio_cache["pitch"]["hop_sec_duration"], None)  # No error
            
    # Need to calculate - get dynamics first (this will cache dynamics if not already cached)
    _, _, _, rms, _, _, _ = get_cached_or_calculated_dynamics(audio_bytes, sample_rate=44100)

    audio_cache = get_user_cache()
    
    print(f"Loading audio for pitch at {sample_rate}Hz")
    # Load audio at 16000 Hz for pitch extraction
    audio, sr, audio_url, error = load_and_process_audio(audio_bytes, sample_rate=16000)
    if error:
        return None, None, None, None, None, None, None, None, error

    # Compute dynamic parameters
    dur_sec = len(audio) / sr
    hop_ms, hop_sec, batch = compute_dynamic_params_crepe(dur_sec)

    # Select pitch extraction method
    if method == "crepe":
        extractor = CrepePitchExtractor()
        times, pitch, conf = extractor.extract_pitch(audio, hop_ms=hop_ms, batch=batch)
    elif method == "librosa":
        hop_length = int(hop_sec * sr)
        extractor = LibrosaPitchExtractor()
        times, pitch, conf = extractor.extract_pitch(audio, sr=sr, hop_length=hop_length)
    else:
        return None, None, None, None, None, None, None, None, f"Unknown method: {method}"

    # Filter out low-confidence frames
    thresh = (0.1 * np.max(conf))
    pitch = filter_low_confidence(pitch, conf, thresh=thresh)
    if pitch is None:
        return None, None, None, None, None, None, None, None, "No high-confidence pitch found"
    
    pitch = filter_pitch_by_rms(pitch, rms)
    
    # Smooth, replace zeros, find highlight segment
    smoothed_pitch, highlighted = smooth_and_segment_crepe(pitch, hop_sec)
    x_axis = [round(float(t), 2) for t in times]

    # Store in cache if user is authenticated
    if user_id and audio_cache:
        audio_cache["pitch"]["audio"] = audio.tolist()
        audio_cache["pitch"]["sr"] = sr
        audio_cache["pitch"]["pitch"] = pitch.tolist()
        audio_cache["pitch"]["smoothed_pitch"] = smoothed_pitch.tolist()
        audio_cache["pitch"]["highlighted_section"] = highlighted
        audio_cache["pitch"]["x_axis"] = x_axis
        audio_cache["pitch"]["audio_url"] = audio_url
        audio_cache["pitch"]["hop_sec_duration"] = hop_sec

        update_user_cache(user_id, audio_cache)

    return audio, audio_url, sr, pitch, smoothed_pitch, highlighted, x_axis, hop_sec, None  

def process_pitch(audio_bytes, method="crepe"):
    """Simple wrapper around get_cached_or_calculated_pitch that formats the response"""
    
    _, audio_url, sr, _, smoothed_pitch, highlighted, _, _, error = get_cached_or_calculated_pitch(audio_bytes, method)
    
    # Check if there was an error
    if error:
        return None, error
    
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