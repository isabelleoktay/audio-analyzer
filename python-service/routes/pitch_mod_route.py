from flask import Blueprint, request, jsonify
from services.pitch_mod_service import process_pitch_mod
from utils.json_utils import convert_to_builtin_types

pitch_mod_blueprint = Blueprint('pitch-mod', __name__)

@pitch_mod_blueprint.route('/python-service/process-pitch-mod', methods=['POST'])
def handle_pitch_mod():
    audio_file = request.files.get('audioFile')

    voice_type = request.files.get('voiceType')
    if voice_type == "tenor" or voice_type == "bass":
        gender = "male"
    else:
        gender = "female" 

    if not audio_file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    use_clap = request.form.get('useCLAP', 'true').lower() == 'true'
    use_whisper = request.form.get('useWhisper', 'true').lower() == 'true'
    if not use_clap and not use_whisper:
        use_clap = True 
        use_whisper = True

    monitor_resources = request.form.get('monitorResources', 'true').lower() == 'true'

    result, error = process_pitch_mod(audio_file.read(), gender, use_clap=use_clap, use_whisper=use_whisper, monitor_resources=monitor_resources)
    result = convert_to_builtin_types(result)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)