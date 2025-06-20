from utils.jwt_manager import get_jwt_manager
from flask import current_app, request, session
import os, soundfile as sf, tempfile
import numpy as np

from io import BytesIO
import hashlib
import wave
import subprocess
from essentia.standard import MonoLoader 
import librosa

def get_user_cache():
    """Get user cache from JWT token"""
    jwt_manager = get_jwt_manager()
    
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    user_id = jwt_manager.verify_token(token)
    
    if not user_id:
        return None
    
    return jwt_manager.get_user_cache(user_id)

def update_user_cache(user_id: str, cache_data: dict):
    """Update user cache data"""
    jwt_manager = get_jwt_manager()
    jwt_manager.update_user_cache(user_id, cache_data)

def get_user_id_from_token():
    """Extract user_id from JWT token"""
    jwt_manager = get_jwt_manager()
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    return jwt_manager.verify_token(token)
    
def clear_cache_if_new_file(file_bytes):
    """
    Check if this is a new file and clear all feature caches if so.
    """
    audio_cache = get_user_cache()
    user_id = get_user_id_from_token()
    
    file_hash = get_file_hash(file_bytes)
    current_hash = audio_cache.get('current_file_hash') if audio_cache else None
    
    if current_hash is None or current_hash != file_hash:
        print("New file detected, clearing all feature caches...")

        if audio_cache:
            cleanup_old_audio_files(audio_cache)
        
        # Clear all feature caches
        updated_cache = {
            'current_file_hash': file_hash,
            'pitch': {
                'audio': None,
                'sr': None,
                'pitch': None,
                'smoothed_pitch': None,
                'highlighted_section': None,
                'x_axis': None,
                'audio_url': None,
                'hop_sec_duration': None
            },
            'dynamics': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'dynamics': None,
                'smoothed_dynamics': None,
                'highlighted_section': None,
                'x_axis': None,
            },
            'tempo': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'tempo': None,
                'beats': None
            },
            'vibrato': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'vibrato_rate': None,
                'vibrato_extent': None,
                'highlighted_section': None,
            },
            'phonation': {
                'audio': None,
                'sr': None,
                'audio_url': None,
                'jitter': None,
                'shimmer': None,
                'hnr': None
            }
        }

        # Store the new file hash
        jwt_manager = get_jwt_manager()
        jwt_manager.update_user_cache(user_id, updated_cache)

def load_audio(file_path, sample_rate=44100):
    """Load audio file using librosa."""
    mono_loader = MonoLoader(filename=file_path, sampleRate=sample_rate)
    audio = mono_loader()
    return audio, sample_rate

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def get_audio_url(audio, hash, sr=44100):
    filename = f"{hash}_{sr}Hz.wav"
    outpath = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
    sf.write(outpath, audio, sr)
    print(">>> Written trimmed audio to", outpath)  

    # Determine the protocol based on the environment
    if current_app.config.get("ENV") == "production":
        protocol = "https"
    else:
        protocol = "http"

    return f"{protocol}://{request.host}{request.script_root}/python-service/audio/{filename}"

def load_and_process_audio(file_bytes, sample_rate=44100, return_path=True):
    """
    Simple function to load audio from file bytes and return processed audio data.
    No caching logic - just loads, trims, and returns audio with URL.
    """
    try:
        file_stream = convert_to_wav_if_needed(file_bytes, return_path=return_path)
        audio, sr = load_audio(file_stream, sample_rate=sample_rate)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        file_hash = get_file_hash(file_bytes)
        audio_url = get_audio_url(audio, file_hash, sr=sr)
        
        return audio, sr, audio_url, None
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None, None, str(e)

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
        
def cleanup_old_audio_files(audio_cache):
    """
    Delete old audio files referenced in the cache to prevent folder crowding.
    """
    feature_types = ['pitch', 'dynamics', 'tempo', 'vibrato', 'phonation']
    
    for feature_type in feature_types:
        if (feature_type in audio_cache and 
            isinstance(audio_cache[feature_type], dict) and 
            audio_cache[feature_type].get('audio_url')):
            
            audio_url = audio_cache[feature_type]['audio_url']
            # Extract filename from URL (assumes URL format: .../audio/filename.wav)
            if '/audio/' in audio_url:
                filename = audio_url.split('/audio/')[-1]
                file_path = os.path.join(current_app.config['AUDIO_FOLDER'], filename)
                
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted old audio file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")