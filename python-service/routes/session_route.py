# In your audio_route.py or a separate route file
from flask import Blueprint, jsonify
from utils.audio_loader import clear_user_cache

session_blueprint = Blueprint('session', __name__)

@session_blueprint.route('/python-service/cleanup-session', methods=['POST'])
def cleanup_session():
    """
    Manually clear the user's session cache.
    This can be called when the user navigates away or closes the app.
    """
    try:
        clear_user_cache()
        return jsonify({"message": "Session cache cleaned up successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to clean up session cache: {str(e)}"}), 500