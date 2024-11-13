from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa  
import numpy as np
from scipy.signal import medfilt
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

# Constants
WINDOW_SIZE = 100
HOP_SIZE = 25
SEGMENT_LENGTH_FACTOR = 2.32

@app.route('/process-audio', methods=['POST'])
def process_audio():
    print('Processing audio file...')
    audio_file = request.files.get('audioFile')

    print('Received audio file:', audio_file)

    if audio_file is None:
        return jsonify({'error': 'No file uploaded'}), 400
    
    audio_file = BytesIO(audio_file.read())

    try:
        # Load the audio file
        audio, sr = load_audio(audio_file)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Extract features
    mfccs, rms, pitches, zcr, hop_length = extract_features(audio, sr)

    loudness_smoothed = smooth_curve(rms[0], window_size=100)
    pitches_smoothed = adaptive_smooth_pitch(pitches, base_window=15, max_window=25)

    # Calculate articulation levels
    articulation_levels = calculate_articulation_level(rms, zcr)
    min_articulation_section, max_articulation_section = calculate_min_max_articulation_sections(articulation_levels, hop_length=hop_length, sr=sr)

    # Detect variable sections
    variable_sections, timbre_variability, loudness_variability, pitch_diff_variability = detect_variable_sections(mfccs, rms, pitches, sr, hop_length)
    variable_sections = np.vstack([variable_sections, min_articulation_section, max_articulation_section])

    # Normalize the extracted features
    time_axis = np.arange(len(timbre_variability)) * (hop_length * HOP_SIZE) / sr
    timbre_variability_normalized = normalize_array(timbre_variability)
    loudness_variability_normalized = normalize_array(loudness_variability)
    pitch_diff_variability_normalized = normalize_array(pitch_diff_variability)
    articulation_levels_normalized = normalize_array(articulation_levels)

    # Convert all numpy arrays to lists
    response = {
        'mfccs': mfccs.tolist(),  
        'rms': rms.tolist(),
        'loudness_smoothed': loudness_smoothed.tolist(),
        'pitches': pitches.tolist(),
        'pitches_smoothed': pitches_smoothed.tolist(),
        'zcr': zcr.tolist(),
        'hop_length': hop_length,
        'sample_rate': sr,
        'articulation_levels': articulation_levels.tolist(),
        'variable_sections': variable_sections.tolist(),
        'timbre_variability': timbre_variability.tolist(),
        'loudness_variability': loudness_variability.tolist(),
        'pitch_diff_variability': pitch_diff_variability.tolist(),
        'normalized_time_axis': time_axis.tolist(),
        'normalized_timbre_variability': timbre_variability_normalized.tolist(),
        'normalized_loudness_variability': loudness_variability_normalized.tolist(),
        'normalized_pitch_variability': pitch_diff_variability_normalized.tolist(),
        'normalized_articulation_variability': articulation_levels_normalized.tolist(),
    }

    # Ensure all elements in the response are JSON serializable
    for key, value in response.items():
        if isinstance(value, np.ndarray):
            response[key] = value.tolist()

    return jsonify(response)

@app.route('/')
def home():
    return "Welcome to the Audio Processing API!"

def smooth_curve(data, window_size=5, filter_type='mean'):
    """Smooth the curve using mean or median filter."""
    if filter_type == 'mean':
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')
    elif filter_type == 'median':
        return medfilt(data, kernel_size=window_size)
    else:
        raise ValueError("filter_type must be 'mean' or 'median'")
    
def adaptive_smooth_pitch(pitches, base_window=3, max_window=10, threshold=0.05):
    """Smooth the pitch values adaptively based on pitch change threshold."""
    smoothed_pitches = np.copy(pitches)
    for i in range(1, len(pitches) - 1):
        pitch_change = abs(pitches[i] - pitches[i - 1])
        window_size = base_window if pitch_change > threshold else max_window
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(len(pitches), i + half_window + 1)
        smoothed_pitches[i] = np.median(pitches[start:end])
    return smoothed_pitches

def load_audio(file_path):
    """Load audio file using librosa."""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_features(audio, sr):
    """Extract MFCCs, RMS, pitches, and ZCR from the audio."""
    n_fft = 2048
    hop_length = 512
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
    pitches = librosa.yin(audio, fmin=librosa.note_to_hz('F3'), fmax=librosa.note_to_hz('B6'), sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length)
    return mfccs, rms, pitches, zcr, hop_length

def calculate_articulation_level(rms, zcr, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    """Calculate articulation levels based on RMS and ZCR."""
    articulation_levels = []
    num_frames = rms.shape[1]
    rms_diff_values, zcr_values = calculate_window_stats(rms, zcr, num_frames, window_size, hop_size)
    rms_diff_mean, rms_diff_std = np.mean(rms_diff_values), np.std(rms_diff_values)
    zcr_mean_overall, zcr_std_overall = np.mean(zcr_values), np.std(zcr_values)
    for start in range(0, num_frames - window_size + 1, hop_size):
        rms_diff_standardized = standardize_value(rms[0][start:start + window_size], rms_diff_mean, rms_diff_std)
        zcr_standardized = standardize_value(zcr[0][start:start + window_size], zcr_mean_overall, zcr_std_overall)
        articulation_level = 1 - (0.5 * rms_diff_standardized + 0.5 * zcr_standardized)
        articulation_levels.append(articulation_level)
    return np.array(articulation_levels)

def calculate_window_stats(rms, zcr, num_frames, window_size, hop_size):
    """Calculate mean and standard deviation for RMS and ZCR values within a sliding window."""
    rms_diff_values, zcr_values = [], []
    for start in range(0, num_frames - window_size + 1, hop_size):
        rms_diff_values.append(np.mean(np.abs(np.diff(rms[0][start:start + window_size]))))
        zcr_values.append(np.mean(zcr[0][start:start + window_size]))
    return rms_diff_values, zcr_values

def standardize_value(window, mean, std):
    """Standardize a value based on the mean and standard deviation."""
    value = np.mean(np.abs(np.diff(window)))
    return (value - mean) / std if std != 0 else value - mean

def calculate_min_max_articulation_sections(articulation_levels, hop_length, sr, hop_size=HOP_SIZE):
    """Calculate the sections with minimum and maximum articulation levels."""
    min_idx, max_idx = np.argmin(articulation_levels), np.argmax(articulation_levels)
    segment_length = int(SEGMENT_LENGTH_FACTOR * sr)
    min_articulation_section = (min_idx * hop_size * hop_length) // segment_length * segment_length
    max_articulation_section = (max_idx * hop_size * hop_length) // segment_length * segment_length
    return [[min_articulation_section, min_articulation_section + segment_length], 
            [max_articulation_section, max_articulation_section + segment_length]]

def detect_variable_sections(mfccs, rms, pitches, sr, hop_length):
    """Detect variable sections in the audio based on MFCCs, RMS, and pitches."""
    timbre_variability = calculate_variability(mfccs, WINDOW_SIZE, HOP_SIZE)
    loudness_variability = calculate_variability(rms[0], WINDOW_SIZE, HOP_SIZE)
    pitch_diff_variability = calculate_variability(pitches, WINDOW_SIZE, HOP_SIZE, is_pitch=True)
    segment_length = int(SEGMENT_LENGTH_FACTOR * sr)
    timbre_section = (np.argmax(timbre_variability) * HOP_SIZE * hop_length) // segment_length * segment_length
    loudness_section = (np.argmax(loudness_variability) * HOP_SIZE * hop_length) // segment_length * segment_length
    pitch_diff_section = (np.argmax(pitch_diff_variability) * HOP_SIZE * hop_length) // segment_length * segment_length
    variable_sections = [
        [timbre_section, timbre_section + segment_length],
        [loudness_section, loudness_section + segment_length],
        [pitch_diff_section, pitch_diff_section + segment_length]
    ]
    return np.array(variable_sections), timbre_variability, loudness_variability, pitch_diff_variability

def calculate_variability(data, window_size, hop_size, is_pitch=False):
    """Calculate variability of data using a sliding window."""
    variability = []
    num_frames = data.shape[1] if data.ndim > 1 else len(data)
    for start in range(0, num_frames - window_size + 1, hop_size):
        window = data[:, start:start + window_size] if data.ndim > 1 else data[start:start + window_size]
        if is_pitch:
            window_diffs = [abs(pitch - librosa.midi_to_hz(round(librosa.hz_to_midi(pitch)))) for pitch in window]
            avg_std = np.mean(window_diffs)
        else:
            window_std = np.std(window, axis=1) if data.ndim > 1 else np.std(window)
            avg_std = np.mean(window_std) if data.ndim > 1 else window_std
        variability.append(avg_std)
    return np.array(variability)

def normalize_array(array):
    """Normalize a numpy array."""
    return (array - np.min(array)) / (np.max(array) - np.min(array))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(debug=True, port=port)