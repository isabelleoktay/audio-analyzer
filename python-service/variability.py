import librosa
import numpy as np

def detect_variable_sections(mfccs, rms, pitches, sr, hop_length, hop_size=25, window_size=100, segment_length_factor=2.32):
    """Detect variable sections in the audio based on MFCCs, RMS, and pitches."""
    timbre_variability = calculate_variability(mfccs, window_size, hop_size)
    loudness_variability = calculate_variability(rms[0], window_size, hop_size)
    pitch_diff_variability = calculate_variability(pitches, window_size, hop_size, is_pitch=True)
    segment_length = int(segment_length_factor * sr)
    timbre_section = (np.argmax(timbre_variability) * hop_size * hop_length) // segment_length * segment_length
    loudness_section = (np.argmax(loudness_variability) * hop_size * hop_length) // segment_length * segment_length
    pitch_diff_section = (np.argmax(pitch_diff_variability) * hop_size * hop_length) // segment_length * segment_length
    variable_sections = [
        [timbre_section, timbre_section + segment_length],
        [loudness_section, loudness_section + segment_length],
        [pitch_diff_section, pitch_diff_section + segment_length]
    ]
    return np.array(variable_sections), timbre_variability, loudness_variability, pitch_diff_variability

def calculate_variability(data, window_size, hop_size, is_pitch=False):
    """Calculate variability of data using a sliding window."""
    variability = []
    num_frames = data.shape[1] if data.ndim > 1 else len(data)
    for start in range(0, num_frames - window_size + 1, hop_size):
        window = data[:, start:start + window_size] if data.ndim > 1 else data[start:start + window_size]
        if is_pitch:
            window_diffs = [abs(pitch - librosa.midi_to_hz(round(librosa.hz_to_midi(pitch)))) for pitch in window]
            avg_std = np.mean(window_diffs)
        else:
            window_std = np.std(window, axis=1) if data.ndim > 1 else np.std(window)
            avg_std = np.mean(window_std) if data.ndim > 1 else window_std
        variability.append(avg_std)
    return np.array(variability)