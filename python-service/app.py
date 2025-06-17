import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import tensorflow_hub as hub
from datetime import timedelta

from flask import Flask, session
from flask_cors import CORS
from flask_session import Session
import redis

from routes.pitch_route import pitch_blueprint
from routes.dynamics_route import dynamics_blueprint
from routes.tempo_route import tempo_blueprint
from routes.audio_route import audio_blueprint
from routes.vibrato_route import vibrato_blueprint
from routes.phonation_route import phonation_blueprint
from routes.session_route import session_blueprint

logging.info("Loading VGGish model...")
VGGISH = hub.load("https://tfhub.dev/google/vggish/1")
logging.info("Successfully loaded VGGish model.")

app = Flask(__name__)

# Environment-specific Redis configuration
if os.getenv("FLASK_ENV") == "production":
    # Production Redis configuration
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_password = os.environ.get('REDIS_PASSWORD')
    
    if redis_password:
        redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    else:
        redis_url = f"redis://{redis_host}:{redis_port}/0"
    
    app.config['SESSION_REDIS'] = redis.from_url(redis_url)
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config["ENV"] = "production"
else:
    # Development Redis configuration
    app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
    app.config['SESSION_COOKIE_SECURE'] = False
    app.config["ENV"] = "development"

# Common session configuration
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'audio-analyzer:'
app.config['SESSION_COOKIE_HTTPONLY'] = True

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')

Session(app)
CORS(app)

if os.getenv("FLASK_ENV") == "production":
    app.config["ENV"] = "production"
else:
    app.config["ENV"] = "development"

# Create directory to store audio files
AUDIO_FOLDER = os.path.join(os.getcwd(), 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Register blueprints
app.register_blueprint(pitch_blueprint)
app.register_blueprint(dynamics_blueprint)
app.register_blueprint(tempo_blueprint)
app.register_blueprint(audio_blueprint)
app.register_blueprint(vibrato_blueprint)
app.register_blueprint(phonation_blueprint)
app.register_blueprint(session_blueprint)

@app.before_request
def make_session_permanent():
    session.permanent = True

@app.route('/python-service/')
def home():
    return "Welcome to the Audio Processing API!"

@app.route("/python-service/test-vggish", methods=["GET"])
def test_vggish():
    import numpy as np
    dummy_audio = np.random.rand(16000).astype(np.float32)
    embeddings = VGGISH(dummy_audio)
    return {"embeddings_shape": embeddings.shape.as_list()}

if __name__ == '__main__':
    app.run(app, debug=False, use_reloader=False)
