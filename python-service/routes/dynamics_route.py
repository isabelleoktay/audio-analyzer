from flask import Blueprint, request, jsonify
from services.dynamics_service import process_dynamics
from utils.json_utils import convert_to_builtin_types

dynamics_blueprint = Blueprint('dynamics', __name__)

@dynamics_blueprint.route('/python-service/process-dynamics', methods=['POST'])
def handle_dynamics():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    # read optional sessionId and fileKey from the form so backend knows which cache to use
    session_id = request.form.get("sessionId")
    file_key = request.form.get("fileKey", "input")

    print("changed file")

    result, error = process_dynamics(audio_file.read(), session_id=session_id, file_key=file_key, ignore_cache=False)
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)