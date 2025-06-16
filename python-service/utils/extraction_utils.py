import numpy as np

def filter_low_confidence(data, conf, thresh=0.8):
    mask = conf > thresh
    valid = np.where(mask)[0]
    if valid.size == 0:
        return None
    # nearest‐valid interpolation
    idx = np.arange(data.size)
    pos = np.searchsorted(valid, idx, side='left')
    pos = np.clip(pos, 0, valid.size - 1)
    ld = np.abs(idx - valid[np.maximum(pos - 1, 0)])
    rd = np.abs(idx - valid[np.minimum(pos, valid.size - 1)])
    use_left = (pos > 0) & (ld < rd)
    nearest = np.where(use_left, valid[pos - 1], valid[pos])
    return data[nearest]