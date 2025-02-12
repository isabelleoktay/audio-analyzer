from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import numpy as np
from io import BytesIO
import os
from feature_extraction import extract_features_parallel
from articulation import calculate_articulation, calculate_min_max_articulation_sections
from smoothing_curves import smooth_curve_parallel
from variability import detect_variable_sections
from tempo import calculate_dynamic_tempo
from utils import normalize_array, load_audio

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  
socketio = SocketIO(app, cors_allowed_origins="*", transports=['websocket'])

# Constants
WINDOW_SIZE = 100
HOP_SIZE = 25
HOP_LENGTH = 512
N_FFT = 2048
SEGMENT_PERCENTAGE = 0.1

def emit_progress(socket, message, percentage):
    """
    Helper function to emit progress updates to the frontend.
    """
    socket.emit('progress', {'message': message, 'percentage': percentage})
    print({'message': message, 'percentage': percentage})

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

@app.route('/python-service/process-audio', methods=['POST'])
def process_audio():
    
    audio_file = request.files.get('audioFile')
    min_note = request.form.get('minNote')
    max_note = request.form.get('maxNote')

    if audio_file is None:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = BytesIO(audio_file.read())

    try:
        # Load the audio file
        audio, sr = load_audio(audio_file)
        emit_progress(socketio, '(1/9) Audio file loaded successfully...', 5)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    segment_length_factor = (len(audio) / sr) * SEGMENT_PERCENTAGE

    # Extract features
    emit_progress(socketio, '(2/9) Starting feature extraction...', 10)
    mfccs, rms, pitches, zcr = extract_features_parallel(audio, sr, min_note, max_note, n_fft=N_FFT, hop_length=HOP_LENGTH)
    emit_progress(socketio, '(3/9) Feature extraction complete.', 45)

    dynamic_tempo, global_tempo = calculate_dynamic_tempo(audio, sr, HOP_LENGTH)
    global_tempo_array = np.full_like(dynamic_tempo, global_tempo)
    emit_progress(socketio, '(4/9) Tempo calculation complete.', 55)

    # Smoothing
    loudness_smoothed = smooth_curve_parallel(rms[0], window_size=100)
    pitches_smoothed = smooth_curve_parallel(pitches, filter_type='adaptive', base_window=15, max_window=25)
    pitches_smoothed[loudness_smoothed < 0.01] = 0
    pitches_smoothed[:10] = 0
    pitches_smoothed[-10:] = 0
    emit_progress(socketio, '(5/9) Smoothing complete.', 65)

    # Articulation Levels
    articulation_levels = calculate_articulation(rms, zcr, window_size=WINDOW_SIZE, hop_size=HOP_SIZE)
    min_articulation_section, max_articulation_section = calculate_min_max_articulation_sections(articulation_levels, HOP_LENGTH, sr, hop_size=HOP_SIZE, segment_length_factor=segment_length_factor)
    emit_progress(socketio, '(6/9) Articulation calculation complete.', 75)

    # Variability Detection
    variable_sections, timbre_variability, loudness_variability, pitch_diff_variability = detect_variable_sections(
        mfccs, rms, pitches, sr, HOP_LENGTH, hop_size=HOP_SIZE, window_size=WINDOW_SIZE, segment_length_factor=segment_length_factor
    )
    variable_sections = np.vstack([variable_sections, min_articulation_section, max_articulation_section])
    emit_progress(socketio, '(7/9) Variability calculation complete.', 85)

    # Normalization
    time_axis = np.arange(len(timbre_variability)) * (HOP_LENGTH * HOP_SIZE) / sr
    timbre_variability_normalized = normalize_array(timbre_variability)
    loudness_variability_normalized = normalize_array(loudness_variability)
    pitch_diff_variability_normalized = normalize_array(pitch_diff_variability)
    articulation_levels_normalized = normalize_array(articulation_levels)
    emit_progress(socketio, '(8/9) Variability calculation complete.', 95)

    # Convert all numpy arrays to lists
    response = {
        'audio': audio.tolist(),
        'mfccs': mfccs.tolist(),
        'rms': rms.tolist(),
        'loudness_smoothed': loudness_smoothed.tolist(),
        'pitches': pitches.tolist(),
        'pitches_smoothed': pitches_smoothed.tolist(),
        'zcr': zcr.tolist(),
        'hop_length': HOP_LENGTH,
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
        'dynamic_tempo': dynamic_tempo.tolist(),
        'global_tempo': global_tempo_array.tolist(),
    }

    # Ensure all elements in the response are JSON serializable
    for key, value in response.items():
        if isinstance(value, np.ndarray):
            response[key] = value.tolist()

    emit_progress(socketio, '(9/9) Processing complete!', 100)
    return jsonify(response)

@app.route('/python-service/')
def home():
    return "Welcome to the Audio Processing API!"

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    socketio.run(app, debug=True, port=port)
