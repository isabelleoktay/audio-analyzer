from utils import standardize_value
import numpy as np

def calculate_articulation(rms, zcr, window_size=100, hop_size=50):
    """Calculate articulation levels based on RMS and ZCR."""
    articulation_levels = []
    num_frames = rms.shape[1]
    rms_diff_values, zcr_values = [], []
    
    for start in range(0, num_frames - window_size + 1, hop_size):
        rms_diff_values.append(np.mean(np.abs(np.diff(rms[0][start:start + window_size]))))
        zcr_values.append(np.mean(zcr[0][start:start + window_size]))

    rms_diff_mean, rms_diff_std = np.mean(rms_diff_values), np.std(rms_diff_values)
    zcr_mean_overall, zcr_std_overall = np.mean(zcr_values), np.std(zcr_values)
    for start in range(0, num_frames - window_size + 1, hop_size):
        rms_diff_standardized = standardize_value(rms[0][start:start + window_size], rms_diff_mean, rms_diff_std)
        zcr_standardized = standardize_value(zcr[0][start:start + window_size], zcr_mean_overall, zcr_std_overall)
        articulation_level = 1 - (0.5 * rms_diff_standardized + 0.5 * zcr_standardized)
        articulation_levels.append(articulation_level)
    return np.array(articulation_levels)

def calculate_min_max_articulation_sections(articulation_levels, hop_length, sr, hop_size=25, segment_length_factor=2.32):
    """Calculate the sections with minimum and maximum articulation levels."""
    min_idx, max_idx = np.argmin(articulation_levels), np.argmax(articulation_levels)
    segment_length = int(segment_length_factor * sr)
    min_articulation_section = (min_idx * hop_size * hop_length) // segment_length * segment_length
    max_articulation_section = (max_idx * hop_size * hop_length) // segment_length * segment_length
    return [[min_articulation_section, min_articulation_section + segment_length], 
            [max_articulation_section, max_articulation_section + segment_length]]