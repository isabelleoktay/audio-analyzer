import numpy as np 
import torch
from models.Whisper.whisper_model_prediction import whisper_extract_features_and_predict
from models.CLAP.CLAP_model_prediction import clap_extract_features_and_predict
from config import CLAP_MALE_PITCH_MODEL_PATH, CLAP_FEMALE_PITCH_MODEL_PATH, WHISPER_MALE_PITCH_MODEL_PATH, WHISPER_FEMALE_PITCH_MODEL_PATH
from utils.resource_monitoring import ResourceMonitor, get_resource_logger

def extract_pitch_mod(audio_path, gender, use_clap=True, use_whisper=True, monitor_resources=True):
    file_logger = get_resource_logger()

    if gender == "male":
        best_whisper_model_weights_path = WHISPER_MALE_PITCH_MODEL_PATH
        best_clap_model_weights_path = CLAP_MALE_PITCH_MODEL_PATH
    elif gender == "female":
        best_whisper_model_weights_path = WHISPER_FEMALE_PITCH_MODEL_PATH
        best_clap_model_weights_path = CLAP_FEMALE_PITCH_MODEL_PATH
    else:
        raise ValueError(f"Gender {gender} not recognised.")
    
    if use_clap:
        # get CLAP predictions 
        if monitor_resources:
            clap_monitor = ResourceMonitor(interval=0.1)
            clap_monitor.start()

        clap_class_names, clap_probs_array, __ = clap_extract_features_and_predict(
            audio_path,
            best_model_weights_path=best_clap_model_weights_path,
            classify = "pitch",
        )
        if monitor_resources:
            clap_monitor.stop()
            clap_stats = clap_monitor.summary(feature_type="clap_pitch_modulation")
            print(f"CLAP pitch modulation inference metrics: {clap_stats}")
            file_logger.info(f"CLAP pitch modulation inference metrics: {clap_stats}")
    else:
        clap_class_names = []
        clap_probs_array = np.zeros((0,0))

    if use_whisper:
        # Get Whisper predictions 
        if monitor_resources:
            whisper_monitor = ResourceMonitor(interval=0.1)
            whisper_monitor.start()
        whisper_class_names, whisper_probs_array, __ = whisper_extract_features_and_predict(
            audio_path,
            best_model_weights_path=best_whisper_model_weights_path,
            classify = "pitch",
            gender = gender,
        )
        if monitor_resources:
            whisper_monitor.stop()
            whisper_stats = whisper_monitor.summary(feature_type="whisper_pitch_modulation")
            print(f"Whisper pitch modulation inference metrics: {whisper_stats}")
            file_logger.info(f"Whisper pitch modulation inference metrics: {whisper_stats}")
    else:
        whisper_class_names = []
        whisper_probs_array = np.zeros((0,0))

     # return both 
    return whisper_class_names, whisper_probs_array, clap_class_names, clap_probs_array 