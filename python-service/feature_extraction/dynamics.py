import os
from flask import current_app, request

from utils.audio_loader import (
    get_user_cache,
    get_user_id_from_token,
    update_user_cache,
    load_and_process_audio,
    get_file_cache,
)
from utils.smoothing import smooth_data
from feature_extraction.variability import calculate_high_variability_sections
import librosa
from config import *
import numpy as np

from utils.resource_monitoring import ResourceMonitor, get_resource_logger

def get_cached_or_calculated_dynamics(
    audio_bytes,
    sample_rate=44100,
    return_path=True,
    session_id=None,
    file_key="input",
    ignore_cache=False,
):
    """
    Perform dynamics calculations and store/retrieve results from a session+file scoped cache.

    session_id and file_key may be provided by the caller or read from request.form.
    If ignore_cache is True, skip reading/writing cache and always compute from scratch.
    """
    # allow caller or frontend to pass sessionId/fileKey via form data

    file_logger = get_resource_logger()
    
    if session_id is None:
        session_id = request.form.get("sessionId")
    if not file_key:
        file_key = request.form.get("fileKey", "input")

    # When ignoring cache, skip cache lookup / user id resolution
    file_cache = user_cache = None
    user_id = None

    if not ignore_cache:
        # try to get the file-scoped cache; fall back to legacy user cache
        file_cache, user_cache, user_id = get_file_cache(session_id, file_key, create_if_missing=True)
        if file_cache is None:
            # legacy path: unauthenticated or missing jwt — fall back to get_user_cache
            audio_cache = get_user_cache()
            user_id = get_user_id_from_token()
        else:
            audio_cache = file_cache
    else:
        audio_cache = None

    # Check cache hit (only when not ignoring cache)
    if not ignore_cache and (
        audio_cache
        and audio_cache.get("dynamics", {}).get("dynamics") is not None
        and audio_cache.get("dynamics", {}).get("sr") == sample_rate
        and audio_cache.get("dynamics", {}).get("audio_url") is not None
    ):
        audio_url = audio_cache["dynamics"]["audio_url"]
        if "/audio/" in audio_url:
            filename = audio_url.split("/audio/")[-1]
            file_path = os.path.join(current_app.config["AUDIO_FOLDER"], filename)

            if not os.path.exists(file_path):
                print(f"Cached audio file {file_path} no longer exists, recalculating...")
            else:
                print("Cache hit for dynamics")
                cached_audio = (
                    np.array(audio_cache["dynamics"]["audio"])
                    if audio_cache["dynamics"]["audio"] is not None
                    else None
                )
                cached_dynamics = (
                    np.array(audio_cache["dynamics"]["dynamics"])
                    if audio_cache["dynamics"]["dynamics"] is not None
                    else None
                )
                cached_smoothed = (
                    np.array(audio_cache["dynamics"]["smoothed_dynamics"])
                    if audio_cache["dynamics"]["smoothed_dynamics"] is not None
                    else None
                )

                return (
                    cached_audio,
                    audio_cache["dynamics"]["audio_url"],
                    audio_cache["dynamics"]["sr"],
                    cached_dynamics,
                    cached_smoothed,
                    audio_cache["dynamics"]["highlighted_section"],
                    audio_cache["dynamics"]["x_axis"],
                )

    # Not cached or ignoring cache — load and compute
    print(f"Loading audio for dynamics at {sample_rate}Hz (session={session_id} file_key={file_key} ignore_cache={ignore_cache})")

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    audio, sr, audio_url, __, error = load_and_process_audio(
        audio_bytes, sample_rate=sample_rate, return_path=return_path
    )
    if error:
        return None, None, None, None, None, None, error

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
        rms[0],
        window_size=window_size,
        hop_size=hop_size,
        hop_duration_sec=HOP_LENGTH / sr,
        audio_duration_sec=audio_duration_sec,
        is_pitch=False,
    )

    # Smooth RMS values
    smoothed_rms = smooth_data(rms[0], filter_type="mean", window_percentage=0.1)

    monitor.stop()
    stats = monitor.summary(feature_type="dynamics")
    print(f"Dynamics inference metrics: {stats}")
    file_logger.info(f"Dynamics inference metrics: {stats}")

    # Store in session+file cache if user is authenticated and not ignoring cache
    if (not ignore_cache) and user_id and audio_cache is not None:
        audio_cache["dynamics"]["audio"] = audio.tolist()
        audio_cache["dynamics"]["sr"] = sr
        audio_cache["dynamics"]["dynamics"] = rms[0].tolist()
        audio_cache["dynamics"]["smoothed_dynamics"] = smoothed_rms.tolist()
        audio_cache["dynamics"]["highlighted_section"] = highlighted_section
        audio_cache["dynamics"]["x_axis"] = time
        audio_cache["dynamics"]["audio_url"] = audio_url

        # persist full user cache (user_cache contains session->files->file_cache)
        if user_cache is not None:
            update_user_cache(user_id, user_cache)

    return audio, audio_url, sr, rms[0], smoothed_rms, highlighted_section, time