import gc
import os
import torch
import torchaudio
import joblib
import logging
import numpy as np
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel
from config import VTC_FRAME_DURATION_SEC, VTC_OVERLAP

logging.basicConfig(level=logging.INFO)

GLOBAL_WHISPER_MODEL = None
GLOBAL_WHISPER_EXTRACTOR = None
GLOBAL_NUM_CLASSES = None  # Track the number of classes currently loaded

class WhisperClassifier(nn.Module):
    """
    Base Whisper encoder + linear classifier
    """
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(base_model_name)
        self.whisper.encoder.gradient_checkpointing = False
        self.classifier = nn.Linear(self.whisper.config.d_model, num_classes)

    def forward(self, input_features):
        encoder_outputs = self.whisper.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

def get_whisper_model_and_extractor(base_model_name, num_classes, device="cpu"):
    """
    Get a global Whisper model and feature extractor.
    Resets the model if num_classes changes.
    """
    global GLOBAL_WHISPER_MODEL, GLOBAL_WHISPER_EXTRACTOR, GLOBAL_NUM_CLASSES

    if GLOBAL_WHISPER_MODEL is None or GLOBAL_NUM_CLASSES != num_classes:
        logging.info(f"Initializing Whisper model for {num_classes} classes.")
        GLOBAL_WHISPER_MODEL = WhisperClassifier(base_model_name, num_classes).to(device)
        GLOBAL_WHISPER_MODEL.eval()
        GLOBAL_NUM_CLASSES = num_classes

    if GLOBAL_WHISPER_EXTRACTOR is None:
        GLOBAL_WHISPER_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(base_model_name)

    return GLOBAL_WHISPER_MODEL, GLOBAL_WHISPER_EXTRACTOR

def whisper_extract_features_and_predict(
    audio_path: str,
    best_model_weights_path: str,
    classify: str = "pitch",
    gender: str = "female",
    sample_rate: int = 16000,
    window_len_secs: float = VTC_FRAME_DURATION_SEC,
    overlap: float = VTC_OVERLAP,
    whisper_base_model_name: str = "openai/whisper-base",
    device: str = "cpu",
):
    logging.info("Extracting Whisper features and predicting...")

    # Load label encoder
    label_encoder_path = os.path.join(
        "./models/Whisper",
        f"{classify}_{gender}_{str(window_len_secs).replace('.', '_')}_label_encoder.joblib",
    )
    label_encoder = joblib.load(label_encoder_path)
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    # Load model & feature extractor (reset if num_classes changed)
    model, feature_extractor = get_whisper_model_and_extractor(
        whisper_base_model_name, num_classes, device
    )

    # Load classifier weights
    if not os.path.exists(best_model_weights_path):
        raise FileNotFoundError(f"No weights file found at {best_model_weights_path}")
    best_model_state = torch.load(best_model_weights_path, map_location=device)
    model.load_state_dict(best_model_state)
    model.eval()

    # Load audio
    audio, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sample_rate:
        audio = torchaudio.functional.resample(audio, orig_sr, sample_rate)
    audio = audio[0].numpy().astype(np.float32)

    # Windowing
    window_size = int(window_len_secs * sample_rate)
    step_size = int(window_size * (1 - overlap))
    starts = np.arange(0, len(audio) - window_size + 1, step_size)

    all_probs = []
    all_times = []

    # Process windows
    with torch.no_grad():
        for start in starts:
            segment = audio[start:start + window_size]
            inputs = feature_extractor(segment, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs["input_features"].to(device)
            logits = model(input_features)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
            all_times.append(start / sample_rate)

    global GLOBAL_WHISPER_MODEL, GLOBAL_WHISPER_EXTRACTOR, GLOBAL_NUM_CLASSES
    del model, feature_extractor
    GLOBAL_WHISPER_MODEL = None
    GLOBAL_WHISPER_EXTRACTOR = None
    GLOBAL_NUM_CLASSES = None

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return class_names, np.stack(all_probs), np.array(all_times)

