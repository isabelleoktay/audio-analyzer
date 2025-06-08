import librosa
import essentia.standard as es
import numpy as np
from scipy.signal import medfilt
from smoothing_curves import smooth_curve_parallel
from utils.smoothing import smooth_data

def calculate_dynamic_tempo(audio, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=None, hop_length=hop_length, onset_envelope=onset_env)
    global_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    smooth_dynamic_tempo = smooth_curve_parallel(dynamic_tempo, window_size=50, filter_type='median')

    return smooth_dynamic_tempo, global_tempo[0]

# def calculate_dynamic_tempo_essentia(audio):
#     rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
#     global_tempo, beats, _, _, _ = rhythm_extractor(audio)

#     # Convert beat positions to time (in seconds)
#     time_axis = np.array(beats)

#     # Convert beats to local tempo estimates (BPM)
#     dynamic_tempo = np.diff(time_axis)  # Time intervals between beats
#     dynamic_tempo = 60.0 / dynamic_tempo  # Convert to BPM
#     tempo_time_axis = (time_axis[:-1] + time_axis[1:]) / 2

#     # Add the first and last values to the front and back of the dynamic tempo array
#     dynamic_tempo = np.insert(dynamic_tempo, 0, dynamic_tempo[0])  # Add first value to the front
#     dynamic_tempo = np.append(dynamic_tempo, dynamic_tempo[-1])    # Add last value to the back
#     tempo_time_axis = np.insert(tempo_time_axis, 0, time_axis[0])  # Add first beat time to the front
#     tempo_time_axis = np.append(tempo_time_axis, time_axis[-1])    # Add last beat time to the back

#     # Smooth the dynamic tempo array
#     smooth_dynamic_tempo = smooth_data(dynamic_tempo, window_size=5, filter_type='median')

#     return smooth_dynamic_tempo, global_tempo, tempo_time_axis

def calculate_dynamic_tempo_essentia(audio, sr, duration):
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

    # Convert beat positions to time (in seconds)
    time_axis = np.array(beats)

    # Convert beats to local tempo estimates (BPM)
    dynamic_tempo = np.diff(time_axis)  # Time intervals between beats
    dynamic_tempo = 60.0 / dynamic_tempo  # Convert to BPM
    tempo_time_axis = (time_axis[:-1] + time_axis[1:]) / 2

    # Add the first and last values to the front and back of the dynamic tempo array
    dynamic_tempo = np.insert(dynamic_tempo, 0, dynamic_tempo[0])  # Add first value to the front
    dynamic_tempo = np.append(dynamic_tempo, dynamic_tempo[-1])    # Add last value to the back
    tempo_time_axis = np.insert(tempo_time_axis, 0, time_axis[0])  # Add first beat time to the front
    tempo_time_axis = np.append(tempo_time_axis, time_axis[-1])    # Add last beat time to the back

    # Create an evenly spaced time axis for interpolation
    interpolated_time_axis = np.linspace(0, duration, int(duration * sr / 512))  # Adjust resolution as needed

    # Interpolate dynamic tempo values to match the evenly spaced time axis
    interpolated_dynamic_tempo = np.interp(interpolated_time_axis, tempo_time_axis, dynamic_tempo)

    # Smooth the interpolated dynamic tempo array
    smooth_dynamic_tempo = smooth_data(interpolated_dynamic_tempo, window_size=5, filter_type='median')

    return smooth_dynamic_tempo, global_tempo, interpolated_time_axis
