import librosa
import essentia.standard as es
import numpy as np
from scipy.signal import medfilt
# from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from smoothing_curves import smooth_curve_parallel
from utils.smoothing import smooth_data

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
    tempo_time_axis = (time_axis[:-1] + time_axis[1:]) / 2

    smooth_dynamic_tempo = smooth_data(dynamic_tempo, window_size=5, filter_type='median')

    return smooth_dynamic_tempo, global_tempo, tempo_time_axis

# def madmom_tempo_over_time(audio_file, window_size=10.0, hop_size=5.0):
#     # Step 1: Beat activation
#     beat_act = RNNBeatProcessor()(audio_file)

#     # Step 2: Beat tracking
#     beat_times = DBNBeatTrackingProcessor(fps=100)(beat_act)

#     # Step 3: Sliding window tempo calculation
#     max_time = beat_times[-1]
#     time_points = np.arange(0, max_time - window_size, hop_size)
#     tempos = []

#     for start in time_points:
#         end = start + window_size
#         window_beats = beat_times[(beat_times >= start) & (beat_times < end)]
#         if len(window_beats) > 1:
#             intervals = np.diff(window_beats)
#             median_interval = np.median(intervals)
#             bpm = 60.0 / median_interval
#         else:
#             bpm = 0  # Not enough beats
#         tempos.append(bpm)

#     return tempos, time_points

def calculate_high_res_tempo_essentia(audio, sr=44100, frame_size=1024, hop_size=512, window_sec=0.5):
    # Essentia algorithms
    windowing = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()
    flux_extractor = es.Flux(norm='L2', halfRectify=True)

    # Generate frames
    frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True)

    prev_spectrum = None
    onset_envelope = []
    times = []

    for i, frame in enumerate(frames):
        windowed = windowing(frame)
        spectrum_complex = fft(windowed)
        spectrum, _ = c2p(spectrum_complex)

        if prev_spectrum is not None:
            diff = spectrum - prev_spectrum
            flux_val = flux_extractor(diff)
            onset_envelope.append(flux_val)
            time = (i * hop_size + frame_size / 2) / sr
            times.append(time)

        prev_spectrum = spectrum

    onset_envelope = np.array(onset_envelope)
    window_size = int(window_sec * sr / hop_size)

    tempo_curve = []
    tempo_times = []

    for i in range(0, len(onset_envelope) - window_size, window_size // 2):
        segment = onset_envelope[i:i+window_size]
        if len(segment) < 2:
            continue

        # Autocorrelation
        ac = np.correlate(segment, segment, mode='full')[len(segment)-1:]
        ac[0] = 0  # ignore zero lag
        peak_lag = np.argmax(ac)

        if peak_lag == 0:
            continue

        lag_time_sec = peak_lag * hop_size / sr
        bpm = 60.0 / lag_time_sec if lag_time_sec > 0 else 0

        tempo_curve.append(bpm)
        tempo_times.append(times[i + window_size // 2])

    # Smooth the tempo curve
    if len(tempo_curve) >= 5:
        smoothed_tempo = smooth_curve_parallel(tempo_curve, window_size=5, filter_type='median')
    else:
        smoothed_tempo = np.array(tempo_curve)

    return smoothed_tempo, tempo_times
