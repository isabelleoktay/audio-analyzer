from flask import current_app, request
import os, soundfile as sf, tempfile

from io import BytesIO
import hashlib
import wave
import subprocess
from essentia.standard import MonoLoader 
import librosa

audio_cache = {
    'hash': None,
    'audio': None,
    'sr': None,
    'audio_url': None,
    'pitch': {
        'audio': None,  
        'audio_url': None,
        'sr': None,  
        'pitch': None,  
        'smoothed_pitch': None,  
        'highlighted_section': None,
        'x_axis': None,
        'hop_sec_duration': None
    }
}

def load_audio(file_path, sample_rate=44100):
    """Load audio file using librosa."""
    mono_loader = MonoLoader(filename=file_path, sampleRate=sample_rate)
    audio = mono_loader()
    return audio, sample_rate

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def get_audio_url(audio, hash, sr=44100):
    filename = f"{hash}.wav"
    outpath = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
    sf.write(outpath, audio, sr)
    print(">>> Written trimmed audio to", outpath)  # Debugging

    # Determine the protocol based on the environment
    if current_app.config.get("ENV") == "production":
        protocol = "https"
    else:
        protocol = "http"

    return f"{protocol}://{request.host}{request.script_root}/python-service/audio/{filename}"

def get_cached_or_loaded_audio(file_bytes, sample_rate=44100, return_path=True):
    global audio_cache
    file_hash = get_file_hash(file_bytes)

    if audio_cache["hash"] != file_hash:
        print("Cache miss or hash mismatch, reloading audio...")  # <--- debug
        audio_cache['hash'] = file_hash
        audio_cache['audio'] = None
        audio_cache['sr'] = None
        audio_cache['audio_url'] = None
        audio_cache['pitch'] = {
            'audio': None,
            'audio_url': None,
            'sr': None,
            'pitch': None,
            'smoothed_pitch': None,
            'highlighted_section': None,
            'x_axis': None,
            'hop_sec_duration': None
        }

        audio_cache['hash'] = file_hash
    
    if audio_cache["sr"] == None or audio_cache["sr"] != sample_rate:
        try:
            file_stream = convert_to_wav_if_needed(file_bytes, return_path=return_path)
            audio, sr = load_audio(file_stream, sample_rate=sample_rate)
            audio, _ = librosa.effects.trim(audio, top_db=20)

            audio_url = get_audio_url(audio, file_hash, sr=sr)

        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None, None, str(e)

        audio_cache['audio'] = audio
        audio_cache['sr'] = sr
        audio_cache['audio_url'] = audio_url

    return audio_cache["audio"], audio_cache["sr"], audio_cache["audio_url"], None

def convert_to_wav_if_needed(input_bytes, return_path=False):
    # Try reading the file as a WAV
    try:
        with wave.open(BytesIO(input_bytes), 'rb') as wav_file:
            # If this succeeds, the input is already a valid WAV file
            wav_file.getparams()  # Trigger reading header
            if return_path:
               # write bytes out to a temp .wav and return its path
                tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tf.write(input_bytes)
                tf.flush()
                return tf.name
            else:
                return BytesIO(input_bytes)
    except wave.Error:
        pass  # Not a valid WAV file, proceed to convert with ffmpeg

    # Not a valid WAV file — convert with ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.input', delete=False) as temp_in, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
        
        temp_in.write(input_bytes)
        temp_in.flush()

        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_in.name,
                '-f', 'wav', '-acodec', 'pcm_s16le',
                temp_out.name
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise RuntimeError("ffmpeg conversion failed")

    if return_path:
       # caller wants the filename
        return temp_out.name
    else:
        with open(temp_out.name, 'rb') as f:
            return BytesIO(f.read())