import numpy as np
from flask import current_app
import os

from feature_extraction.dynamics import get_cached_or_calculated_dynamics
from utils.audio_loader import (
    get_file_cache,
    get_user_cache,
    get_user_id_from_token,
    update_user_cache,
    load_and_process_audio,
    clear_cache_if_new_file,
)
from utils.extraction_utils import filter_low_confidence
from feature_extraction.pitch import CrepePitchExtractor, LibrosaPitchExtractor
from feature_extraction.pitch_utils import (
    compute_dynamic_params_crepe,
    smooth_and_segment_crepe,
    filter_pitch_by_rms
)
from utils.resource_monitoring import ResourceMonitor, get_resource_logger


def get_cached_or_calculated_pitch(audio_bytes, sample_rate=16000, method="crepe", session_id=None, file_key="input"):
    
    file_logger = get_resource_logger()

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()
    
    # Clear cache only for the specific session/file
    clear_cache_if_new_file(audio_bytes, session_id=session_id, file_key=file_key)
    
    # Try to get file-scoped cache (returns file_cache, user_cache, user_id)
    file_cache, user_cache, user_id = get_file_cache(session_id, file_key, create_if_missing=True)
    
    # If no file-scoped cache (unauthenticated), fall back to legacy user cache
    if file_cache is None:
        audio_cache = get_user_cache()
        user_id = get_user_id_from_token()
    else:
        audio_cache = file_cache

    # Check cache first
    if (audio_cache 
        and audio_cache.get("pitch", {}).get("pitch") is not None 
        and audio_cache.get("pitch", {}).get("sr") == sample_rate 
        and audio_cache.get("pitch", {}).get("audio_url") is not None):

        audio_url = audio_cache["pitch"]["audio_url"]
        if '/audio/' in audio_url:
            filename = audio_url.split('/audio/')[-1]
            file_path = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
            
            if not os.path.exists(file_path):
                current_app.logger.info(f"Cached audio file {file_path} no longer exists, recalculating...")
            else:
                current_app.logger.info("Cache hit for pitch")
                cached_audio = np.array(audio_cache["pitch"]["audio"]) if audio_cache["pitch"]["audio"] is not None else None
                cached_pitch = np.array(audio_cache["pitch"]["pitch"]) if audio_cache["pitch"]["pitch"] is not None else None
                cached_smoothed = np.array(audio_cache["pitch"]["smoothed_pitch"]) if audio_cache["pitch"]["smoothed_pitch"] is not None else None
                
                return (cached_audio, audio_cache["pitch"]["audio_url"], audio_cache["pitch"]["sr"], 
                        cached_pitch, cached_smoothed,
                        audio_cache["pitch"]["highlighted_section"], audio_cache["pitch"]["x_axis"], 
                        audio_cache["pitch"]["hop_sec_duration"], None)  # No error
            
    # Ensure dynamics for this same session/file are available (dynamics uses same session/file cache)
    _, _, _, rms, _, _, _ = get_cached_or_calculated_dynamics(audio_bytes, sample_rate=44100, session_id=session_id, file_key=file_key)

    print(f"Loading audio for pitch at {sample_rate}Hz (session={session_id} file_key={file_key})")
    # Load audio at requested sample rate for pitch extraction
    audio, sr, audio_url, __, error = load_and_process_audio(audio_bytes, sample_rate=sample_rate)
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

    monitor.stop()
    stats = monitor.summary(feature_type="pitch")
    print(f"Pitch inference metrics: {stats}")
    file_logger.info(f"Pitch inference metrics: {stats}")

    # Store in session+file cache if user is authenticated
    # user_cache is the full user cache dict (contains sessions -> files -> file_cache)
    if user_id and (user_cache is not None):
        # ensure file_cache exists within user_cache for this session/file
        # get_file_cache returned file_cache variable earlier; update that
        file_cache["pitch"]["audio"] = audio.tolist()
        file_cache["pitch"]["sr"] = sr
        file_cache["pitch"]["pitch"] = pitch.tolist()
        file_cache["pitch"]["smoothed_pitch"] = smoothed_pitch.tolist()
        file_cache["pitch"]["highlighted_section"] = highlighted
        file_cache["pitch"]["x_axis"] = x_axis
        file_cache["pitch"]["audio_url"] = audio_url
        file_cache["pitch"]["hop_sec_duration"] = hop_sec

        # persist the full user_cache back to storage
        update_user_cache(user_id, user_cache)

    return audio, audio_url, sr, pitch, smoothed_pitch, highlighted, x_axis, hop_sec, None  

def process_pitch(audio_bytes, sample_rate=16000, method="crepe", session_id=None, file_key="input"):
    """Simple wrapper around get_cached_or_calculated_pitch that formats the response"""
    audio, audio_url, sr, _, smoothed_pitch, highlighted, _, _, error = get_cached_or_calculated_pitch(
        audio_bytes, sample_rate=sample_rate, method=method, session_id=session_id, file_key=file_key
    )
    if error:
        return None, error

    audio_duration = len(audio) / sr if audio is not None else 0

    result = {
        'data': [
            {
                'data': smoothed_pitch.tolist() if hasattr(smoothed_pitch, 'tolist') else smoothed_pitch,
                'highlighted': highlighted,
                'label': 'pitch'
            }
        ],
        'sample_rate': sr,
        'audio_url': audio_url,
        'duration': audio_duration
    }

    return result, None