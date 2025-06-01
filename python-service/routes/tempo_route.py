from flask import Blueprint, request, jsonify
from services.tempo_service import process_tempo
from utils.json_utils import convert_to_builtin_types

tempo_blueprint = Blueprint('tempo', __name__)

@tempo_blueprint.route('/python-service/process-tempo', methods=['POST'])
def handle_dynamics():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_tempo(audio_file.read())
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)