from flask import Blueprint, request, jsonify
from services.dynamics_service import process_dynamics
from utils.json_utils import convert_to_builtin_types

dynamics_blueprint = Blueprint('dynamics', __name__)

@dynamics_blueprint.route('/python-service/process-dynamics', methods=['POST'])
def handle_dynamics():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_dynamics(audio_file.read())
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)