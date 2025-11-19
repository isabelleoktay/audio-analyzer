import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_cors import CORS

from utils.jwt_manager import init_jwt_manager 
from config import get_redis_client

from routes.pitch_route import pitch_blueprint
from routes.dynamics_route import dynamics_blueprint
from routes.tempo_route import tempo_blueprint
from routes.audio_route import audio_blueprint
from routes.vibrato_route import vibrato_blueprint
from routes.phonation_route import phonation_blueprint
from routes.auth_route import auth_blueprint
from routes.vocal_tone_route import vocal_tone_blueprint
from routes.pitch_mod_route import pitch_mod_blueprint

app = Flask(__name__)

if os.getenv("FLASK_ENV") == "production":
    app.config["ENV"] = "production"
else:
    app.config["ENV"] = "development"

redis_client = get_redis_client()

# Initialize JWT Manager with Redis
jwt_secret = os.environ.get('FLASK_SECRET_KEY', 'your_jwt_secret_key_here')
init_jwt_manager(secret_key=jwt_secret, redis_client=redis_client)

CORS(app, supports_credentials=True)

# Create directory to store audio files
AUDIO_FOLDER = os.path.join(os.getcwd(), 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Register blueprints
app.register_blueprint(pitch_blueprint)
app.register_blueprint(dynamics_blueprint)
app.register_blueprint(tempo_blueprint)
app.register_blueprint(audio_blueprint)
app.register_blueprint(vibrato_blueprint)
app.register_blueprint(phonation_blueprint)
app.register_blueprint(auth_blueprint)
app.register_blueprint(vocal_tone_blueprint)
app.register_blueprint(pitch_mod_blueprint)

@app.route('/python-service/')
def home():
    cache_type = "Redis" if redis_client else "In-Memory"
    return f"Welcome to the Audio Processing API! Cache: {cache_type}"

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
