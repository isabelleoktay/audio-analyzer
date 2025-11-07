from flask import Blueprint, request, jsonify
from services.vocal_tone_service import process_vocal_tone
from utils.json_utils import convert_to_builtin_types

phonation_blueprint = Blueprint('vocal-tone', __name__)

@phonation_blueprint.route('/python-service/process-vocal-tone', methods=['POST'])
def handle_vocal_tone():
    audio_file = request.files.get('audioFile')

    voice_type = request.files.get('voiceType')
    if voice_type == "tenor" or voice_type == "bass":
        gender = "male"
    else:
        gender = "female" 

    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    result, error = process_vocal_tone(audio_file.read(), gender)
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)