from flask import Blueprint, request, jsonify
from services.phonation_service import process_phonation
from utils.json_utils import convert_to_builtin_types

phonation_blueprint = Blueprint('phonation', __name__)

@phonation_blueprint.route('/python-service/process-phonation', methods=['POST'])
def handle_phonation():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_phonation(audio_file.read())
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)