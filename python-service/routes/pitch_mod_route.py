from flask import Blueprint, request, jsonify
from services.pitch_mod_service import process_pitch_mod
from utils.json_utils import convert_to_builtin_types

phonation_blueprint = Blueprint('pitch-mod', __name__)

@phonation_blueprint.route('/python-service/process-pitch-mod', methods=['POST'])
def handle_pitch_mod():
    audio_file = request.files.get('audioFile')

    voice_type = request.files.get('voiceType')
    if voice_type == "tenor" or voice_type == "bass":
        gender = "male"
    else:
        gender = "female" 

    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_pitch_mod(audio_file.read(), gender)
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)