from flask import Blueprint, request, jsonify
from services.pitch_service import process_pitch
from utils.json_utils import convert_to_builtin_types

pitch_blueprint = Blueprint('pitch', __name__)

@pitch_blueprint.route('/python-service/process-pitch', methods=['POST'])
def handle_pitch():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_pitch(audio_file.read(), method="librosa")
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)