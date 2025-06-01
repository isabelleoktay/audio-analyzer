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

def calculate_variable_section(data, hop_length, sample_rate, is_pitch=False, hop_size=25, window_size=100, segment_length_factor=2.32):
    variability = calculate_variability(data, window_size, hop_size, is_pitch)

    variable_start_frame = np.argmax(variability)
    segment_length_frames = int(segment_length_factor * sample_rate / hop_length)
    variable_end_frame = variable_start_frame + segment_length_frames

    # Convert frame indices to sample indices
    variable_start_sample = variable_start_frame * hop_length
    variable_end_sample = variable_end_frame * hop_length

    variable_segment = {
        'frame': {
            'start': variable_start_frame,
            'end': variable_end_frame
        },
        'sample': {
            'start': variable_start_sample / sample_rate,
            'end': variable_end_sample / sample_rate
        }
    }

    return variable_segment

def get_high_variability_section(variability, hop_length, sample_rate, percentage=0.25, audio_duration_sec=None):

    variability = np.asarray(variability)
    num_windows = len(variability)
    if num_windows == 0:
        raise ValueError("Variability array is empty.")

    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    # Use provided audio duration, otherwise estimate from variability windows
    if audio_duration_sec is None:
        audio_duration_sec = (num_windows * hop_length) / sample_rate

    highlight_duration_sec = audio_duration_sec * percentage
    half_duration_sec = highlight_duration_sec / 2

    # Get peak variability time (in window units)
    peak_index = int(np.argmax(variability))
    # Map window index to time
    peak_time_sec = (peak_index + 0.5) * (audio_duration_sec / num_windows)

    # Determine highlight start and end times (clamped to audio bounds)
    start_time_sec = max(0, peak_time_sec - half_duration_sec)
    end_time_sec = min(audio_duration_sec, peak_time_sec + half_duration_sec)

    # Convert times back to indices (in frames)
    start_index = int(start_time_sec * sample_rate / hop_length)
    end_index = int(end_time_sec * sample_rate / hop_length)

    return {
        'frame': {
            'start': start_index,
            'end': end_index,
        },
        'sample': {
            'start': start_time_sec,
            'end': end_time_sec
        }
    }

def get_high_variability_section_with_time_axis(variability, time_axis, percentage=0.25):

    variability = np.asarray(variability)
    time_axis = np.asarray(time_axis)

    if len(variability) == 0 or len(time_axis) == 0:
        raise ValueError("Variability and time_axis must not be empty.")
    if len(variability) != len(time_axis):
        raise ValueError("Variability and time_axis must be the same length.")
    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    total_duration = time_axis[-1] - time_axis[0]
    highlight_duration = total_duration * percentage
    half_duration = highlight_duration / 2

    peak_index = int(np.argmax(variability))
    print(peak_index)
    peak_time = time_axis[peak_index]

    start_time = max(time_axis[0], peak_time - half_duration)
    end_time = min(time_axis[-1], peak_time + half_duration)

    # Find closest indices in time_axis to start and end times
    start_index = int(np.searchsorted(time_axis, start_time, side='left'))
    end_index = int(np.searchsorted(time_axis, end_time, side='right'))

    return {
        'frame': {
            'start': start_index,
            'end': end_index,
        },
        'sample': {
            'start': float(time_axis[start_index]),
            'end': float(time_axis[min(end_index, len(time_axis) - 1)])
        }
    }

def get_high_variability_pitch_section(variability, percentage, hop_duration_sec, audio_duration_sec):

    variability = np.asarray(variability)
    num_frames = len(variability)
    if num_frames == 0:
        raise ValueError("Variability array is empty.")

    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    highlight_duration_sec = audio_duration_sec * percentage
    half_duration_sec = highlight_duration_sec / 2

    peak_index = int(np.argmax(variability))
    peak_time_sec = peak_index * (audio_duration_sec / num_frames)

    start_time_sec = max(0, peak_time_sec - half_duration_sec)
    end_time_sec = min(audio_duration_sec, peak_time_sec + half_duration_sec)

    start_index = int(start_time_sec / hop_duration_sec)
    end_index = int(end_time_sec / hop_duration_sec)

    return {
        'frame': {
            'start': start_index,
            'end': end_index,
        },
        'sample': {
            'start': start_time_sec,
            'end': end_time_sec
        }
    }


def calculate_variability(data, window_size, hop_size, is_pitch=False):
    """
    Calculates the variability of audio feature data using a sliding window.

    Parameters:
        feature_data (array-like): The 1D array of feature values (e.g., RMS, pitch, etc.).
        window_size (int): The number of samples in each sliding window.
        hop_size (int): The number of samples to move the window at each step.

    Returns:
        list of float: Variability (standard deviation) for each window.
    """
    variability = []
    num_frames = data.shape[1] if data.ndim > 1 else len(data)
    for start in range(0, num_frames - window_size + 1, hop_size):
        window = data[:, start:start + window_size] if data.ndim > 1 else data[start:start + window_size]
        if is_pitch:
            window_diffs = [
                abs(p - librosa.midi_to_hz(round(librosa.hz_to_midi(p)))) if p > 0 and np.isfinite(p) else 0
                for p in window
            ]
            avg_std = np.mean(window_diffs)
        else:
            window_std = np.std(window, axis=1) if data.ndim > 1 else np.std(window)
            avg_std = np.mean(window_std) if data.ndim > 1 else window_std
        variability.append(avg_std)
    return np.array(variability)