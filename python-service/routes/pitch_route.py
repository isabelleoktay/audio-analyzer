# from flask import Blueprint, request, jsonify, current_app
# from services.pitch_service import process_pitch
# from utils.json_utils import convert_to_builtin_types

# pitch_blueprint = Blueprint('pitch', __name__)

# @pitch_blueprint.route('/python-service/process-pitch', methods=['POST'])
# def handle_pitch():
#     audio_file = request.files.get('audioFile')
#     if not audio_file:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     current_app.logger.info(f"Request files keys: {list(request.files.keys())}")
#     current_app.logger.info(f"Request form keys: {list(request.form.keys())}")

#     if 'audioFile' not in request.files:
#         current_app.logger.error("No audioFile found in request.files")
#         return {"error": "No audioFile provided"}, 400

#     result, error = process_pitch(audio_file.read(), sample_rate=16000, method="crepe")
#     result = convert_to_builtin_types(result)
#     if error:
#         return jsonify({'error': error}), 400
#     return jsonify(result)

from flask import Blueprint, request, jsonify, current_app
from services.pitch_service import process_pitch
from utils.json_utils import convert_to_builtin_types

pitch_blueprint = Blueprint('pitch', __name__)

@pitch_blueprint.route('/python-service/process-pitch', methods=['POST'])
def handle_pitch():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    # read optional sessionId and fileKey so backend knows which cache to use
    session_id = request.form.get("sessionId")
    file_key = request.form.get("fileKey", "input")

    current_app.logger.info(f"Request files keys: {list(request.files.keys())}")
    current_app.logger.info(f"Request form keys: {list(request.form.keys())}")

    result, error = process_pitch(
        audio_file.read(),
        sample_rate=16000,
        method="crepe",
        session_id=session_id,
        file_key=file_key
    )
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)