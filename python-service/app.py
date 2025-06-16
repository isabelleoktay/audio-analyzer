import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import tensorflow_hub as hub

from flask import Flask
from flask_cors import CORS

from routes.pitch_route import pitch_blueprint
from routes.dynamics_route import dynamics_blueprint
from routes.tempo_route import tempo_blueprint
from routes.audio_route import audio_blueprint
from routes.vibrato_route import vibrato_blueprint
from routes.phonation_route import phonation_blueprint

logging.info("Loading VGGish model...")
VGGISH = hub.load("https://tfhub.dev/google/vggish/1")
logging.info("Successfully loaded VGGish model.")

app = Flask(__name__)
CORS(app)

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
