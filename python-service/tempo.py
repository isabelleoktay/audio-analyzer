import librosa
from scipy.stats import lognorm
import numpy as np
from smoothing_curves import smooth_curve_parallel

def calculate_dynamic_tempo(audio, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    prior_lognorm = lognorm(loc=np.log(120), scale=120, s=1)
    dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=None, hop_length=hop_length, onset_envelope=onset_env, prior=prior_lognorm)

    smooth_dynamic_tempo = smooth_curve_parallel(dynamic_tempo, window_size=50)

    return smooth_dynamic_tempo