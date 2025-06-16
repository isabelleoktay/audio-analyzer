from scipy.signal import medfilt
import numpy as np

def smooth_data(data, window_size=None, filter_type='median', threshold=None, base_window=3, max_window=10, window_percentage=None):
    """Smooth a chunk and return its index with the result. Can handle both adaptive and regular smoothing.
    If window_percentage is given, window_size is ignored and calculated as a percentage of data length.
    """
    if window_percentage is not None:
        # Calculate window size as a percentage of data length
        window_size = max(1, int(len(data) * window_percentage))
        if filter_type == 'median' and window_size % 2 == 0:
            window_size += 1  # Ensure odd for median

    # Handle regular smoothing (mean or median filter)
    if filter_type == 'mean':
        smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    elif filter_type == 'median':
        if window_size % 2 == 0:
            window_size += 1  # Ensure window_size is odd
        smoothed_data = medfilt(data, kernel_size=window_size)
    elif filter_type == 'adaptive' and threshold is not None:
        smoothed_data = np.zeros(data.size)
        for i in range(1, len(data) - 1):
            pitch_change = abs(data[i] - data[i - 1])
            window_size_adaptive = base_window if pitch_change > threshold else max_window
            half_window = window_size_adaptive // 2
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            smoothed_data[i] = np.median(data[start_idx:end_idx])
    else:
        raise ValueError("Unsupported filter type. Use 'mean', 'median', or 'adaptive'.")

    return smoothed_data