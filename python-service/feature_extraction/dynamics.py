from utils.audio_loader import get_user_cache
from utils.smoothing import smooth_data
from feature_extraction.variability import calculate_high_variability_sections
import librosa
from config import *
from flask import session
import numpy as np

def get_cached_or_calculated_dynamics(audio, sr, audio_url):
    """
    Perform all dynamics-related calculations (RMS, highlighted sections, etc.).

    Args:
        audio (numpy.ndarray): The audio data.
        sr (int): The sample rate of the audio.

    Returns:
        dict: A dictionary containing RMS values, highlighted sections, and time axis.
    """
    audio_cache = get_user_cache()

    if (audio_cache["dynamics"]["dynamics"] is not None and 
        audio_cache["dynamics"]["smoothed_dynamics"] is not None):
        
        # Convert cached lists back to numpy arrays
        cached_audio = np.array(audio_cache["dynamics"]["audio"]) if audio_cache["dynamics"]["audio"] is not None else None
        cached_dynamics = np.array(audio_cache["dynamics"]["dynamics"]) if audio_cache["dynamics"]["dynamics"] is not None else None
        cached_smoothed = np.array(audio_cache["dynamics"]["smoothed_dynamics"]) if audio_cache["dynamics"]["smoothed_dynamics"] is not None else None
        
        return (cached_audio, audio_cache["dynamics"]["audio_url"], 
                audio_cache["dynamics"]["sr"], cached_dynamics, cached_smoothed,
                audio_cache["dynamics"]["highlighted_section"], 
                audio_cache["dynamics"]["x_axis"])
    
    # Calculate RMS values
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    data_len = rms.shape[1]
    time = [round(i * HOP_LENGTH / sr, 2) for i in range(data_len)]

    # Calculate highlighted sections
    window_size = max(1, int(data_len * WINDOW_PERCENTAGE))
    hop_size = max(1, int(data_len * HOP_PERCENTAGE))
    window_size = min(window_size, data_len)

    audio_duration_sec = len(audio) / sr
    highlighted_section = calculate_high_variability_sections(
        rms[0], window_size=window_size, hop_size=hop_size,
        hop_duration_sec=HOP_LENGTH / sr, audio_duration_sec=audio_duration_sec, is_pitch=False
    )

    # Smooth RMS values
    smoothed_rms = smooth_data(rms[0], filter_type='mean', window_percentage=0.1)

    audio_cache["dynamics"]["audio"] = audio.tolist()
    audio_cache["dynamics"]["sr"] = sr
    audio_cache["dynamics"]["dynamics"] = rms[0].tolist()
    audio_cache["dynamics"]["smoothed_dynamics"] = smoothed_rms.tolist()
    audio_cache["dynamics"]["highlighted_section"] = highlighted_section
    audio_cache["dynamics"]["x_axis"] = time
    audio_cache["dynamics"]["audio_url"] = audio_url

    session['audio_cache'] = audio_cache

    return audio, audio_url, sr, rms[0], smoothed_rms, highlighted_section, time