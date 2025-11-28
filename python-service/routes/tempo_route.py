# from flask import Blueprint, request, jsonify
# from services.tempo_service import process_tempo
# from utils.json_utils import convert_to_builtin_types

# tempo_blueprint = Blueprint('tempo', __name__)

# @tempo_blueprint.route('/python-service/process-tempo', methods=['POST'])
# def handle_tempo():
#     audio_file = request.files.get('audioFile')
#     if not audio_file:
#         return jsonify({'error': 'No file uploaded'}), 400

#     result, error = process_tempo(audio_file.read())
#     result = convert_to_builtin_types(result)
#     if error:
#         return jsonify({'error': error}), 400
#     return jsonify(result)

from flask import Blueprint, request, jsonify, current_app
from services.tempo_service import process_tempo
from utils.json_utils import convert_to_builtin_types

tempo_blueprint = Blueprint('tempo', __name__)

@tempo_blueprint.route('/python-service/process-tempo', methods=['POST'])
def handle_tempo():
    audio_file = request.files.get('audioFile')
    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400

    # read optional sessionId and fileKey from the form so backend knows which cache to use
    session_id = request.form.get("sessionId")
    file_key = request.form.get("fileKey", "input")

    current_app.logger.info(f"handle_tempo called session_id={session_id} file_key={file_key}")

    result, error = process_tempo(audio_file.read(), session_id=session_id, file_key=file_key)
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)