from flask import Blueprint, current_app, send_from_directory, abort, request, jsonify
import os
from utils.audio_loader import convert_to_wav_if_needed, get_user_id_from_token, get_jwt_manager

audio_blueprint = Blueprint('audio', __name__, url_prefix='/python-service/audio')

@audio_blueprint.route('/cleanup-temp-files', methods=['POST'])
def cleanup_temp_files():
    base_folder = current_app.config['AUDIO_FOLDER']
    protected_folders = {"testing", "collected"}  # Add any other protected folders here

    # read clear_cache flag (default = false)
    clear_cache = request.args.get("clear_cache", "false").lower() == "true"

    try:
        # ---------------------------------------------------------
        # 1. Delete non-protected files and folders from AUDIO_FOLDER
        # ---------------------------------------------------------
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)

            # skip protected folders
            if os.path.isdir(item_path) and item in protected_folders:
                continue

            # delete files
            if os.path.isfile(item_path):
                os.remove(item_path)
            # delete folders (only if empty)
            elif os.path.isdir(item_path):
                try:
                    os.rmdir(item_path)
                except OSError:
                    pass  # ignore: folder not empty

        # ---------------------------------------------------------
        # 2. Also clear the user's entire cache if clear_cache=true
        # ---------------------------------------------------------
        if clear_cache:
            user_id = get_user_id_from_token()
            if user_id:
                jwt_manager = get_jwt_manager()
                jwt_manager.update_user_cache(user_id, {})  # wipe everything
            else:
                return jsonify({"error": "Could not clear cache: no valid user token"}), 400

        # ---------------------------------------------------------
        return jsonify({
            "message": "Temporary files cleaned successfully",
            "cache_cleared": clear_cache
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to clean up temporary files: {str(e)}"}), 500

    
@audio_blueprint.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    group = request.form.get('group')  # Group can be None
    stage = request.form.get('stage')  # Stage can be None
    audio_feature = request.form.get('feature')  # Feature can be None

    base_folder = current_app.config['AUDIO_FOLDER']

    # Determine target folder based on group, stage, and feature
    if group in ['feedback', 'none'] and stage in ['before', 'during', 'after']:
        testing_folder = os.path.join(base_folder, 'testing')
        target_folder = os.path.join(testing_folder, group, stage)
    else:
        target_folder = os.path.join(base_folder, 'collected')  # Default to normal /static/audio folder

    # Add feature to the path if it exists
    if audio_feature:
        target_folder = os.path.join(target_folder, audio_feature)

    # Create directories if they don't exist
    os.makedirs(target_folder, exist_ok=True)

    # Check if the file already exists
    file_path = os.path.join(target_folder, file.filename)
    if os.path.isfile(file_path):
        return jsonify({'message': 'File already exists', 'path': file_path}), 200

    # Convert the uploaded file to WAV format if needed
    try:
        input_bytes = file.read()
        wav_bytes = convert_to_wav_if_needed(input_bytes)
    except RuntimeError as e:
        return jsonify({'error': f'Failed to convert audio to WAV: {str(e)}'}), 500

    # Save the converted WAV file
    with open(file_path, 'wb') as f:
        f.write(wav_bytes.getvalue())

    return jsonify({'message': f'File uploaded successfully to {target_folder}', 'path': file_path}), 200

@audio_blueprint.route('/<filename>')
def serve_audio(filename):
    if filename == "cleanup-temp-files" or filename == "upload":
        abort(404)  # Prevent this route from handling cleanup-temp-files

    folder = current_app.config['AUDIO_FOLDER']
    full = os.path.join(folder, filename)
    if not os.path.isfile(full):
        abort(404, f"{filename} not found in {folder}")  
    return send_from_directory(folder, filename, mimetype='audio/wav')