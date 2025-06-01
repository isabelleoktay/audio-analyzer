import os
from flask import Flask
from flask_cors import CORS

from routes.pitch_route import pitch_blueprint
from routes.dynamics_route import dynamics_blueprint
from routes.tempo_route import tempo_blueprint
from routes.audio_route import audio_blueprint

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

@app.route('/python-service/')
def home():
    return "Welcome to the Audio Processing API!"

if __name__ == '__main__':
    app.run(app, debug=True)
