from flask import Blueprint, jsonify, request
from utils.jwt_manager import get_jwt_manager
import uuid

auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/python-service/auth/token', methods=['POST'])
def generate_token():
    jwt_manager = get_jwt_manager()

    if not jwt_manager:
        return jsonify({'error': 'JWT manager not initialized'}), 500

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    token = jwt_manager.generate_token(user_id)
    return jsonify({'token': token, 'message': 'Token generated successfully'}), 200

@auth_blueprint.route('/python-service/auth/verify', methods=['POST'])
def verify_token():
    """Verify a JWT token"""
    data = request.get_json()
    
    if not data or 'token' not in data:
        return jsonify({'error': 'Token is required'}), 400
    
    token = data.get('token')
    print(f"Generated token: {token}")
    jwt_manager = get_jwt_manager()
    
    if not jwt_manager:
        return jsonify({'error': 'JWT manager not initialized'}), 500
    
    user_id = jwt_manager.verify_token(token)
    
    if user_id:
        return jsonify({
            'valid': True,
            'user_id': user_id
        }), 200
    else:
        return jsonify({
            'valid': False,
            'error': 'Invalid or expired token'
        }), 401