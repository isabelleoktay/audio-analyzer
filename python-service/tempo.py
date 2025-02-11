import librosa
from smoothing_curves import smooth_curve_parallel

def calculate_dynamic_tempo(audio, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=None, hop_length=hop_length, onset_envelope=onset_env)
    global_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    smooth_dynamic_tempo = smooth_curve_parallel(dynamic_tempo, window_size=50)

    return smooth_dynamic_tempo, global_tempo[0]