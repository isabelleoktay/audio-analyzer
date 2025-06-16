from flask import Blueprint, request, jsonify
from services.vibrato_service import process_vibrato
from utils.json_utils import convert_to_builtin_types

vibrato_blueprint = Blueprint('vibrato', __name__)

@vibrato_blueprint.route('/python-service/process-vibrato', methods=['POST'])
def handle_vibrato():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_vibrato(audio_file.read())
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)