from config import WINDOW_PERCENTAGE, HOP_PERCENTAGE

from utils.smoothing import smooth_data
from utils.data_utils import replace_edge_zeros
from feature_extraction.variability import calculate_high_variability_sections

def compute_dynamic_params_crepe(duration_sec):
    max_hop_ms, max_batch = 50, 256
    hop_ms = min(max_hop_ms, 10 + int(duration_sec / 10))
    hop_sec = hop_ms / 1000.0
    batch = min(max_batch, 64 + int(duration_sec / 5))
    return hop_ms, hop_sec, batch

def smooth_and_segment_crepe(pitch, hop_sec):
    n = pitch.size
    # compute smoothing windows
    b = max(1, int(n * WINDOW_PERCENTAGE))
    h = max(1, int(n * HOP_PERCENTAGE))

    seg = calculate_high_variability_sections(pitch, b, h, hop_duration_sec=hop_sec, audio_duration_sec=n * hop_sec, is_pitch=True)

    # smooth + clean edges
    p2 = smooth_data(pitch, filter_type='adaptive', threshold=0.05, base_window=b, max_window=b*2)
    p2 = replace_edge_zeros(p2)
    return p2, seg

import numpy as np

def filter_pitch_by_rms(pitch, rms, rms_threshold = 0.01):
    """
    Filters pitch values based on RMS threshold. If RMS and pitch arrays are not the same length,
    RMS values are interpolated to match the pitch array length.

    Args:
        pitch (numpy.ndarray): Array of pitch values.
        rms (numpy.ndarray): Array of RMS values.
        rms_threshold (float): Threshold below which pitch values are set to 0.
        pitch_length (int): Length of the pitch array.

    Returns:
        numpy.ndarray: Filtered pitch values.
    """
    # Interpolate RMS to match the length of the pitch array if needed
    pitch_length = len(pitch)

    if len(rms) != pitch_length:
        rms = np.interp(
            np.linspace(0, len(rms) - 1, pitch_length),  # Target indices
            np.arange(len(rms)),  # Original indices
            rms  # Original RMS values
        )

    # Apply RMS threshold to filter pitch values
    pitch = np.where(rms >= rms_threshold, pitch, 0)

    return pitch