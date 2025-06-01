from flask import Blueprint, current_app, send_from_directory, abort
import os

audio_blueprint = Blueprint('audio', __name__, url_prefix='/python-service/audio')

@audio_blueprint.route('/<filename>')
def serve_audio(filename):
    folder = current_app.config['AUDIO_FOLDER']
    full = os.path.join(folder, filename)
    if not os.path.isfile(full):
        # abort with a message so you can see it in your browser
        abort(404, f"{filename} not found in {folder}")  
    return send_from_directory(folder, filename, mimetype='audio/wav')