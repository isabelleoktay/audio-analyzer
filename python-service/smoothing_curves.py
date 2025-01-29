from scipy.signal import medfilt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def smooth_chunk(index, data, start, end, window_size, filter_type, threshold=None, base_window=None, max_window=None):
    """Smooth a chunk and return its index with the result. Can handle both adaptive and regular smoothing."""
    chunk = data[start:end]

    # Handle regular smoothing (mean or median filter)
    if filter_type == 'mean':
        smoothed_chunk = np.convolve(chunk, np.ones(window_size) / window_size, mode='same')
    elif filter_type == 'median':
        smoothed_chunk = medfilt(chunk, kernel_size=window_size)
    
    # Handle adaptive smoothing for pitch
    elif filter_type == 'adaptive' and threshold is not None:
        smoothed_chunk = np.zeros(chunk.size)
        for i in range(1, len(chunk) - 1):
            pitch_change = abs(chunk[i] - chunk[i - 1])
            window_size_adaptive = base_window if pitch_change > threshold else max_window
            half_window = window_size_adaptive // 2
            start_idx = max(0, i - half_window)
            end_idx = min(len(chunk), i + half_window + 1)
            smoothed_chunk[i] = np.median(chunk[start_idx:end_idx])

    else:
        raise ValueError("Unsupported filter type. Use 'mean', 'median', or 'adaptive'.")

    return index, smoothed_chunk

def smooth_curve_parallel(data, window_size=5, filter_type='mean', n_workers=4, threshold=0.05, base_window=3, max_window=10):
    """Perform parallel smoothing with optional adaptive smoothing for pitch."""
    chunk_size = len(data) // n_workers

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i in range(n_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_workers - 1 else len(data)
            futures.append(executor.submit(smooth_chunk, i, data, start, end, window_size, filter_type, threshold, base_window, max_window))

    # Gather and sort results by chunk index
    results = sorted((future.result() for future in as_completed(futures)), key=lambda x: x[0])
    
    # Combine all smoothed chunks in order
    smoothed_data = np.hstack([chunk for _, chunk in results])

    return smoothed_data
