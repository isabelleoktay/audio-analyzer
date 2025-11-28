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

def _get_user_cache_dict():
    jwt_manager = get_jwt_manager()
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None, None
    token = auth_header.split(' ')[1]
    user_id = jwt_manager.verify_token(token)
    if not user_id:
        return None, None
    user_cache = jwt_manager.get_user_cache(user_id) or {}
    return user_cache, user_id

def get_session_cache(session_id, create_if_missing=True):
    """
    Return (user_cache, session_cache, user_id)
    session_cache has 'files' mapping where each file_key (e.g. 'input'/'reference' or file id)
    holds feature caches.
    """
    user_cache, user_id = _get_user_cache_dict()
    if user_cache is None:
        return None, None, None

    sessions = user_cache.get("sessions", {})
    session_cache = sessions.get(session_id) if session_id else sessions.get("default")
    if session_cache is None and create_if_missing:
        session_cache = {"files": {}}
        sessions[session_id or "default"] = session_cache
        user_cache["sessions"] = sessions
        jwt_manager = get_jwt_manager()
        jwt_manager.update_user_cache(user_id, user_cache)
    return user_cache, session_cache, user_id

def get_file_cache(session_id, file_key="input", create_if_missing=True):
    user_cache, session_cache, user_id = get_session_cache(session_id, create_if_missing=create_if_missing)
    if session_cache is None:
        return None, None, None
    files = session_cache.get("files", {})
    file_cache = files.get(file_key)
    if file_cache is None and create_if_missing:
        # Template single-file cache (same structure as old top-level but per file)
        file_cache = {
            "current_file_hash": None,
            "pitch": {"audio": None, "sr": None, "audio_url": None, "pitch": None, "smoothed_pitch": None, "highlighted_section": None, "x_axis": None, "hop_sec_duration": None},
            "dynamics": {"audio": None, "sr": None, "audio_url": None, "dynamics": None, "smoothed_dynamics": None, "highlighted_section": None, "x_axis": None},
            "tempo": {"audio": None, "sr": None, "audio_url": None, "tempo": None, "beats": None},
            "vibrato": {"audio": None, "sr": None, "audio_url": None, "vibrato_rate": None, "vibrato_extent": None, "highlighted_section": None},
            "phonation": {"audio": None, "sr": None, "audio_url": None, "breathy": None, "flow": None, "neutral": None, "pressed": None},
            "pitch mod.": {"audio": None, "sr": None, "audio_url": None, "CLAP": {}, "Whisper": {}},
            "vocal tone": {"audio": None, "sr": None, "audio_url": None, "CLAP": {}, "Whisper": {}},
        }
        files[file_key] = file_cache
        session_cache["files"] = files
        jwt_manager = get_jwt_manager()
        jwt_manager.update_user_cache(user_id, user_cache)
    return file_cache, user_cache, user_id

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

def clear_cache_if_new_file(file_bytes, session_id=None, file_key="input"):
    """
    Check if this is a new file for a particular session/file_key and clear only that file's feature caches.
    If session_id is None, uses 'default' session.
    """
    file_cache, user_cache, user_id = get_file_cache(session_id, file_key, create_if_missing=True)
    if file_cache is None:
        return

    file_hash = get_file_hash(file_bytes)
    current_hash = file_cache.get("current_file_hash")

    if current_hash is None or current_hash != file_hash:
        print(f"New file detected for session={session_id} file_key={file_key}, clearing that file's feature caches...")
        cleanup_old_audio_files(file_cache)

        # Reset only this file's cache (preserve other session/file caches)
        new_file_cache = {
            "current_file_hash": file_hash,
            # same structure as template above — set features to None / empty
            "pitch": {"audio": None, "sr": None, "audio_url": None, "pitch": None, "smoothed_pitch": None, "highlighted_section": None, "x_axis": None, "hop_sec_duration": None},
            "dynamics": {"audio": None, "sr": None, "audio_url": None, "dynamics": None, "smoothed_dynamics": None, "highlighted_section": None, "x_axis": None},
            "tempo": {"audio": None, "sr": None, "audio_url": None, "tempo": None, "beats": None},
            "vibrato": {"audio": None, "sr": None, "audio_url": None, "vibrato_rate": None, "vibrato_extent": None, "highlighted_section": None},
            "phonation": {"audio": None, "sr": None, "audio_url": None, "breathy": None, "flow": None, "neutral": None, "pressed": None},
            "pitch mod.": {"audio": None, "sr": None, "audio_url": None, "CLAP": {}, "Whisper": {}},
            "vocal tone": {"audio": None, "sr": None, "audio_url": None, "CLAP": {}, "Whisper": {}},
        }

        # save back into user cache
        _, session_cache, _ = get_session_cache(session_id, create_if_missing=True)
        session_cache_files = session_cache.get("files", {})
        session_cache_files[file_key] = new_file_cache
        session_cache["files"] = session_cache_files

        user_cache, _ = _get_user_cache_dict()
        user_cache_sessions = user_cache.get("sessions", {})
        user_cache_sessions[session_id or "default"] = session_cache
        user_cache["sessions"] = user_cache_sessions
        jwt_manager = get_jwt_manager()
        jwt_manager.update_user_cache(user_id, user_cache)
    
# def clear_cache_if_new_file(file_bytes):
#     """
#     Check if this is a new file and clear all feature caches if so.
#     """
#     audio_cache = get_user_cache()
#     user_id = get_user_id_from_token()
    
#     file_hash = get_file_hash(file_bytes)
#     current_hash = audio_cache.get('current_file_hash') if audio_cache else None
    
#     if current_hash is None or current_hash != file_hash:
#         print("New file detected, clearing all feature caches...")

#         if audio_cache:
#             cleanup_old_audio_files(audio_cache)
        
#         # Clear all feature caches
#         updated_cache = {
#             'current_file_hash': file_hash,
#             'pitch': {
#                 'audio': None,
#                 'sr': None,
#                 'pitch': None,
#                 'smoothed_pitch': None,
#                 'highlighted_section': None,
#                 'x_axis': None,
#                 'audio_url': None,
#                 'hop_sec_duration': None
#             },
#             'dynamics': {
#                 'audio': None,
#                 'sr': None,
#                 'audio_url': None,
#                 'dynamics': None,
#                 'smoothed_dynamics': None,
#                 'highlighted_section': None,
#                 'x_axis': None,
#             },
#             'tempo': {
#                 'audio': None,
#                 'sr': None,
#                 'audio_url': None,
#                 'tempo': None,
#                 'beats': None
#             },
#             'vibrato': {
#                 'audio': None,
#                 'sr': None,
#                 'audio_url': None,
#                 'vibrato_rate': None,
#                 'vibrato_extent': None,
#                 # 'jitter': None,
#                 # 'shimmer': None,
#                 # 'hnr': None
#                 'highlighted_section': None,
#             },
#             'phonation': {
#                 'audio': None,
#                 'sr': None,
#                 'audio_url': None,
#                 'breathy': None,
#                 'flow': None,
#                 'neutral': None,
#                 'pressed': None,
#             },
#             'pitch mod.': {
#                 'audio': None,
#                 'sr': None,
#                 'audio_url': None,
#                 'CLAP': {
#                     'straight' : None,
#                     'vibrato' : None,
#                     'trill' : None,
#                     'trillo' : None,
#                 },
#                 'Whisper': {
#                     'straight' : None,
#                     'vibrato' : None,
#                     'trill' : None,
#                     'trillo' : None,
#                 },
#             },
#             'vocal tone': {
#                 'audio': None,
#                 'sr': None,
#                 'audio_url': None,
#                 'CLAP': {
#                     'breathy': None,
#                     'belt': None,
#                     'vocal_fry': None,
#                     'inhaled': None,
#                     'spoken': None,
#                 },
#                 'Whisper': {
#                     'breathy': None,
#                     'belt': None,
#                     'vocal_fry': None,
#                     'inhaled': None,
#                     'spoken': None,
#                 },
#             },
#         }

#         # Store the new file hash
#         jwt_manager = get_jwt_manager()
#         jwt_manager.update_user_cache(user_id, updated_cache)

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

def universal_trim(audio, sr, top_db=20):
    """
    Downsample to a fixed rate for consistent silence detection
    """
    trim_sr = 16000
    audio_ds = librosa.resample(audio, orig_sr=sr, target_sr=trim_sr)

    # Trim at the fixed rate
    trimmed_ds, idx = librosa.effects.trim(audio_ds, top_db=top_db)

    # Map index back to original sample rate
    start = int(idx[0] * (sr / trim_sr))
    end   = int(idx[1] * (sr / trim_sr))

    # Trim original audio with mapped indices
    return audio[start:end], (start, end)

def load_and_process_audio(file_bytes, sample_rate=44100, return_path=True):
    """
    Simple function to load audio from file bytes and return processed audio data.
    No caching logic - just loads, trims, and returns audio with URL.
    """
    try:
        file_stream = convert_to_wav_if_needed(file_bytes, return_path=return_path)
        audio, sr = load_audio(file_stream, sample_rate=sample_rate)
        audio, _ = universal_trim(audio, sr, top_db=20)
        
        file_hash = get_file_hash(file_bytes)
        audio_url = get_audio_url(audio, file_hash, sr=sr)

        # SAVE AUDIO TO TEMP FILE for CLAP and Whisper)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio, sr)
        audio_path = temp_wav.name

        return audio, sr, audio_url, audio_path, None
        
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
    feature_types = ['pitch', 'dynamics', 'tempo', 'vibrato', 'phonation', 'pitch_mod', 'vocal tone']
    
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