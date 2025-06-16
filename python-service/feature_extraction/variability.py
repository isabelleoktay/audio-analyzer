import librosa
import numpy as np

def convert_variability_indices_to_time(start_index, end_index, audio_duration_sec, num_frames):

    start_time_sec = start_index * (audio_duration_sec / num_frames)
    end_time_sec = end_index * (audio_duration_sec / num_frames)

    return start_time_sec, end_time_sec

def append_variable_section(variable_sections, start_index, end_index, audio_duration_sec, hop_duration_sec, num_frames):
    start_time_sec, end_time_sec = convert_variability_indices_to_time(start_index, end_index, audio_duration_sec, num_frames)

    variable_sections['data'].append({
        'start': int(start_time_sec / hop_duration_sec),
        'end': int(end_time_sec / hop_duration_sec)
    })
    variable_sections['audio'].append({
        'start': start_time_sec,
        'end': end_time_sec
    })


def calculate_variability(data, window_size, hop_size, is_pitch=False):

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


def calculate_high_variability_sections(data, window_size, hop_size, hop_duration_sec=None, 
                                        audio_duration_sec=None, 
                                        thresh=0.5, is_pitch=False):
    
    # Calculate variability for the entire pitch data
    variability = calculate_variability(data, window_size, hop_size, is_pitch=is_pitch)
    thresh_value = np.max(variability) * thresh

    variable_sections = {
        'data': [],
        'audio': []
    }

    # Iterate through the variability array to find high variability frames
    num_frames = len(variability)
    start_index = None

    for i in range(num_frames):
        if variability[i] > thresh_value:
            # Start a new section if not already started
            if start_index is None:
                start_index = i
        else:
            # End the current section if variability drops below the threshold
            if start_index is not None:
                end_index = i - 1
                append_variable_section(variable_sections, start_index, end_index, audio_duration_sec, hop_duration_sec, num_frames)

                # Reset the start index for the next section
                start_index = None

    # Handle the case where the last section ends at the last frame
    if start_index is not None:
        end_index = num_frames - 1
        append_variable_section(variable_sections, start_index, end_index, audio_duration_sec, hop_duration_sec, num_frames)

    return variable_sections
    