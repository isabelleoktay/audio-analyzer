import numpy as np
from essentia.standard import PitchCREPE
from config import CREPE_MODEL_PATH, WINDOW_PERCENTAGE, HOP_PERCENTAGE, SEGMENT_PERCENTAGE

from utils.smoothing import smooth_data
from utils.data_utils import replace_edge_zeros
from variability import calculate_variability, get_high_variability_pitch_section

def compute_dynamic_params_crepe(duration_sec):
    max_hop_ms, max_batch = 50, 256
    hop_ms = min(max_hop_ms, 10 + int(duration_sec / 10))
    hop_sec = hop_ms / 1000.0
    batch = min(max_batch, 64 + int(duration_sec / 5))
    return hop_ms, hop_sec, batch

def extract_raw_pitch_crepe(audio, hop_ms, batch):
    extractor = PitchCREPE(
        graphFilename=CREPE_MODEL_PATH,
        hopSize=hop_ms,
        batchSize=batch
    )
    times, pitch, conf, _ = extractor(audio)
    return np.array(times), np.array(pitch), np.array(conf)

def smooth_and_segment_crepe(pitch, hop_sec):
    n = pitch.size
    # compute smoothing windows
    b = max(1, int(n * WINDOW_PERCENTAGE))
    h = max(1, int(n * HOP_PERCENTAGE))
    v = calculate_variability(pitch, window_size=b, hop_size=h, is_pitch=True)
    seg = get_high_variability_pitch_section(v, SEGMENT_PERCENTAGE, hop_sec, n * hop_sec)
    # smooth + clean edges
    p2 = smooth_data(pitch, filter_type='adaptive', threshold=0.05, base_window=b, max_window=b*2)
    p2 = replace_edge_zeros(p2)
    return p2, seg