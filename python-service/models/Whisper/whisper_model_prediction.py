import os 
import torch
import librosa
import joblib
import logging
import numpy as np
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel
from config import VTC_FRAME_DURATION_SEC, VTC_OVERLAP

logging.basicConfig(level=logging.INFO)

class WhisperClassifier(nn.Module):
    """
    A classifier model that utilizes the encoder of a pretrained Whisper model for feature extraction,
    followed by a linear layer for classification.
    Args:
        model_name (str): Name or path of the pretrained Whisper model to load.
        num_classes (int): Number of output classes for classification.
    Attributes:
        whisper (WhisperModel): The pretrained Whisper model (encoder used for feature extraction).
        classifier (nn.Linear): Linear layer mapping encoder outputs to class logits.
    Forward Args:
        input_features (torch.Tensor): Input features of shape (batch, 80, time), as expected by Whisper encoder.
    Returns:
        torch.Tensor: Logits of shape (batch, num_classes) representing class predictions.
    """

    def __init__(self, model_name, num_classes):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.whisper.encoder.gradient_checkpointing = False
        self.classifier = nn.Linear(self.whisper.config.d_model, num_classes)

    def forward(self, input_features):
        # input_features shape: (batch, 80, time) -> Whisper expects (batch, 80, time)
        # Use only the encoder part of the Whisper model
        encoder_outputs = self.whisper.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (batch, time, hidden)
        pooled = hidden_states.mean(dim=1)  # mean pooling over time
        logits = self.classifier(pooled)
        return logits


def whisper_extract_features_and_predict(
    audio_path: str,
    best_model_weights_path: str,
    classify: str = "pitch",
    gender: str = "female",
    sample_rate: int = 16000,
    window_len_secs: float = VTC_FRAME_DURATION_SEC,
    overlap: float = VTC_OVERLAP,
    whisper_base_model_name: str = "openai/whisper-base",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Extracts Whisper features from a single audio file and passes them through a model
    loaded from a .pth checkpoint (trained in PyTorch).
    """

    logging.info("Extracting Whisper features and predicting...")

    # Load label encoder
    label_encoder_path = os.path.join(
        "./models/Whisper",
        f"{classify}_{gender}_{str(window_len_secs).replace('.', '_')}_label_encoder.joblib",
    )

    logging.info("Loading label encoder from file %s", label_encoder_path)
    label_encoder = joblib.load(label_encoder_path)
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    # Load the feature extractor from the *base* Whisper model
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_base_model_name)

    # Initialize model architecture and load your custom weights
    if os.path.exists(best_model_weights_path) and os.path.isfile(best_model_weights_path):
        model = WhisperClassifier(whisper_base_model_name, num_classes).to(device)
        best_model_state = torch.load(
            best_model_weights_path, map_location=torch.device(device)
        )
        model.load_state_dict(best_model_state)
        logging.info("Loaded saved best model weights.")
    else:
        raise ValueError("No best weights file found for evaluation.")

    model.to(device)
    model.eval()

    # Load and segment the audio file
    audio, __ = librosa.load(audio_path, sr=sample_rate)
    window_size = int(window_len_secs * sample_rate)
    step_size = int(window_size * (1 - overlap))
    starts = np.arange(0, len(audio) - window_size + 1, step_size)

    all_probs, window_times = [], []

    for start in starts:
        segment = audio[start:start + window_size]
        inputs = feature_extractor(segment, sampling_rate=sample_rate, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=-1)
            all_probs.append(probs.cpu().numpy()[0])

        window_times.append(start / sample_rate)

    probs_array = np.stack(all_probs)
    return class_names, probs_array, np.array(window_times)