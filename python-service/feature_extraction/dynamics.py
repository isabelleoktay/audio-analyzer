import os
from utils.audio_loader import get_user_cache, get_user_id_from_token, update_user_cache, load_and_process_audio
from utils.smoothing import smooth_data
from feature_extraction.variability import calculate_high_variability_sections
import librosa
from config import *
import numpy as np
from flask import current_app

def get_cached_or_calculated_dynamics(audio_bytes, sample_rate=44100, return_path=True):
    """
    Perform all dynamics-related calculations (RMS, highlighted sections, etc.).

    Args:
        audio (numpy.ndarray): The audio data.
        sr (int): The sample rate of the audio.

    Returns:
        dict: A dictionary containing RMS values, highlighted sections, and time axis.
    """
    audio_cache = get_user_cache()
    user_id = get_user_id_from_token()

    if (audio_cache and audio_cache["dynamics"]["dynamics"] is not None and 
        audio_cache["dynamics"]["sr"] == sample_rate and audio_cache["dynamics"]["audio_url"] is not None):

        audio_url = audio_cache["dynamics"]["audio_url"]
        if '/audio/' in audio_url:
            filename = audio_url.split('/audio/')[-1]
            file_path = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
            
            if not os.path.exists(file_path):
                print(f"Cached audio file {file_path} no longer exists, recalculating...")
            else:
                # Convert cached lists back to numpy arrays
                print("Cache hit for dynamics")
                cached_audio = np.array(audio_cache["dynamics"]["audio"]) if audio_cache["dynamics"]["audio"] is not None else None
                cached_dynamics = np.array(audio_cache["dynamics"]["dynamics"]) if audio_cache["dynamics"]["dynamics"] is not None else None
                cached_smoothed = np.array(audio_cache["dynamics"]["smoothed_dynamics"]) if audio_cache["dynamics"]["smoothed_dynamics"] is not None else None
                
                return (cached_audio, audio_cache["dynamics"]["audio_url"], 
                        audio_cache["dynamics"]["sr"], cached_dynamics, cached_smoothed,
                        audio_cache["dynamics"]["highlighted_section"], 
                        audio_cache["dynamics"]["x_axis"])
    
    print(f"Loading audio for dynamics at {sample_rate}Hz")
    audio, sr, audio_url, error = load_and_process_audio(audio_bytes, sample_rate=sample_rate, return_path=return_path)
    if error:
        return None, None, None, None, None, None
    
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

    # Store in cache if user is authenticated
    if user_id and audio_cache:
        audio_cache["dynamics"]["audio"] = audio.tolist()
        audio_cache["dynamics"]["sr"] = sr
        audio_cache["dynamics"]["dynamics"] = rms[0].tolist()
        audio_cache["dynamics"]["smoothed_dynamics"] = smoothed_rms.tolist()
        audio_cache["dynamics"]["highlighted_section"] = highlighted_section
        audio_cache["dynamics"]["x_axis"] = time
        audio_cache["dynamics"]["audio_url"] = audio_url

        update_user_cache(user_id, audio_cache)

    return audio, audio_url, sr, rms[0], smoothed_rms, highlighted_section, time