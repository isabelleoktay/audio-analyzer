import librosa
import numpy as np
import essentia.standard as es
from smoothing_curves import smooth_curve_parallel

def calculate_dynamic_tempo(audio, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=None, hop_length=hop_length, onset_envelope=onset_env)
    global_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    smooth_dynamic_tempo = smooth_curve_parallel(dynamic_tempo, window_size=50, filter_type='median')

    return smooth_dynamic_tempo, global_tempo[0]

def calculate_dynamic_tempo_essentia(audio):
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    global_tempo, beats, _, _, _ = rhythm_extractor(audio)

    # Convert beat positions to time (in seconds)
    time_axis = np.array(beats)

    # Convert beats to local tempo estimates (BPM)
    dynamic_tempo = np.diff(time_axis)  # Time intervals between beats
    dynamic_tempo = 60.0 / dynamic_tempo  # Convert to BPM

    smooth_dynamic_tempo = smooth_curve_parallel(dynamic_tempo, window_size=5, filter_type='median')

    return smooth_dynamic_tempo, global_tempo