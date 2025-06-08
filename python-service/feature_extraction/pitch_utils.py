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