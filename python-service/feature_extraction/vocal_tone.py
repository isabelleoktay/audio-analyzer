import torch
import logging
from models.Whisper.whisper_model_prediction import whisper_extract_features_and_predict
from models.CLAP.CLAP_model_prediction import clap_extract_features_and_predict
from config import CLAP_MALE_TIMBRE_MODEL_PATH, CLAP_FEMALE_TIMBRE_MODEL_PATH, WHISPER_MALE_TIMBRE_MODEL_PATH, WHISPER_FEMALE_TIMBRE_MODEL_PATH

logging.basicConfig(level=logging.INFO)


def extract_vocal_tone(audio_path, gender):

    if gender == "male":
        best_whisper_model_weights_path = WHISPER_MALE_TIMBRE_MODEL_PATH
        best_clap_model_weights_path = CLAP_MALE_TIMBRE_MODEL_PATH
    elif gender == "female":
        best_whisper_model_weights_path = WHISPER_FEMALE_TIMBRE_MODEL_PATH
        best_clap_model_weights_path = CLAP_FEMALE_TIMBRE_MODEL_PATH
    else:
        raise ValueError(f"Gender {gender} not recognised.")
    
    # Get Whisper predictions 
    whisper_class_names, whisper_probs_array, __ = whisper_extract_features_and_predict(
        audio_path,
        best_model_weights_path=best_whisper_model_weights_path,
        classify = "timbre",
        gender = gender,
    )

    # get CLAP predictions 
    clap_class_names, clap_probs_array, __ = clap_extract_features_and_predict(
        audio_path,
        best_model_weights_path=best_clap_model_weights_path,
        classify = "timbre",
    )

    # return both 
    return whisper_class_names, whisper_probs_array, clap_class_names, clap_probs_array