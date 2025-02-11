import numpy as np
import librosa

def standardize_value(window, mean, std):
    """Standardize a value based on the mean and standard deviation."""
    value = np.mean(np.abs(np.diff(window)))
    return (value - mean) / std if std != 0 else value - mean

def normalize_array(array):
    """Normalize a numpy array."""
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def load_audio(file_path):
    """Load audio file using librosa."""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr