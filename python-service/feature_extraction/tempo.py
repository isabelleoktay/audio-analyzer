import essentia.standard as es
import numpy as np
from utils.smoothing import smooth_data

def calculate_dynamic_tempo_essentia(audio, sr, duration, window_frac=0.01):
    """
    Calculates dynamic tempo using Essentia's RhythmExtractor and interpolates BPM values
    to ensure correct positioning on the time axis.

    Parameters:
        audio (array-like): The audio signal.
        sr (int): The sample rate of the audio.
        duration (float): The duration of the audio in seconds.

    Returns:
        tuple: Interpolated dynamic tempo, global tempo, and interpolated time axis.
    """
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    global_tempo, beats, _, _, _ = rhythm_extractor(audio)

    # Handle case with too few beats to calculate tempo
    if len(beats) < 2:
        interpolated_time_axis = np.linspace(0, duration, int(duration * sr / 512))
        return np.full_like(interpolated_time_axis, np.nan), 0, interpolated_time_axis

    time_axis = np.array(beats)
    dynamic_tempo = 60.0 / np.diff(time_axis)
    tempo_time_axis = (time_axis[:-1] + time_axis[1:]) / 2

    dynamic_tempo = np.insert(dynamic_tempo, 0, dynamic_tempo[0])
    dynamic_tempo = np.append(dynamic_tempo, dynamic_tempo[-1])
    tempo_time_axis = np.insert(tempo_time_axis, 0, time_axis[0])
    tempo_time_axis = np.append(tempo_time_axis, time_axis[-1])

    interpolated_time_axis = np.linspace(0, duration, int(duration * sr / 512))
    interpolated_dynamic_tempo = np.interp(interpolated_time_axis,
                                           tempo_time_axis,
                                           dynamic_tempo)

    # compute window size as a fraction of total frames, ensure odd ≥ 3
    window_size = max(3, int(window_frac * len(interpolated_dynamic_tempo)))
    if window_size % 2 == 0:
        window_size += 1

    smooth_dynamic_tempo = smooth_data(interpolated_dynamic_tempo,
                                       window_size=window_size,
                                       filter_type='median')
    return smooth_dynamic_tempo, global_tempo, interpolated_time_axis

def calculate_dynamic_tempo_beattracker(audio, sr, duration,
                                        min_tempo=40,
                                        max_tempo=208,
                                        min_confidence=1.5,
                                        window_frac=0.1):
    """
    Calculates dynamic tempo using Essentia's BeatTrackerMultiFeature,
    but returns NaN‐filled tempo if confidence < min_confidence.

    Parameters:
        audio (array-like): audio signal (44100 Hz).
        sr (int): sample rate.
        duration (float): total duration in seconds.
        min_tempo (int): slowest tempo [bpm].
        max_tempo (int): fastest tempo [bpm].
        min_confidence (float): minimum “good” confidence threshold.

    Returns:
        tuple: (smooth_dynamic_tempo, confidence, interpolated_time_axis)
    """
    beat_tracker = es.BeatTrackerMultiFeature(minTempo=min_tempo,
                                              maxTempo=max_tempo)
    ticks, confidence = beat_tracker(audio)

    interpolated_time_axis = np.linspace(0,
                                         duration,
                                         int(duration * sr / 512))
    if confidence < min_confidence:
        return np.full_like(interpolated_time_axis, np.nan), confidence, interpolated_time_axis

    time_axis = np.array(ticks)
    dynamic_tempo = 60.0 / np.diff(time_axis)
    tempo_time_axis = (time_axis[:-1] + time_axis[1:]) / 2

    dynamic_tempo = np.insert(dynamic_tempo, 0, dynamic_tempo[0])
    dynamic_tempo = np.append(dynamic_tempo, dynamic_tempo[-1])
    tempo_time_axis = np.insert(tempo_time_axis, 0, time_axis[0])
    tempo_time_axis = np.append(tempo_time_axis, time_axis[-1])

    interpolated_dynamic_tempo = np.interp(interpolated_time_axis,
                                           tempo_time_axis,
                                           dynamic_tempo)

    # dynamic smoothing window
    window_size = max(3, int(window_frac * len(interpolated_dynamic_tempo)))
    if window_size % 2 == 0:
        window_size += 1

    smooth_dynamic_tempo = smooth_data(interpolated_dynamic_tempo,
                                       window_size=window_size,
                                       filter_type='median')
    return smooth_dynamic_tempo, confidence, interpolated_time_axis
