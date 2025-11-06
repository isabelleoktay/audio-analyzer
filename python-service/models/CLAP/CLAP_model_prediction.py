import os
import torch
import numpy as np
import torch.nn.functional as F
import joblib
import librosa
import torchaudio
import os
import tempfile
import numpy as np
import soundfile as sf

from pathlib import Path
from msclap import CLAP
from sklearn.preprocessing import LabelEncoder
from utils.utils import get_class_names

from config import VTC_FRAME_DURATION_SEC, VTC_OVERLAP

WINDOW_LEN_SECS = 0.5

class ClapClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def clap_extract_features_and_predict(
    audio_path: str,
    best_model_weights_path: str,
    classify: str = "vocalset_10",
    model_selected_data: str = "mixed",
    sample_rate: int = 16000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Extracts CLAP embeddings from a single audio file and passes them through
    a trained CLAP classifier loaded from a .pth checkpoint.
    """

    # Load label encoder
    label_encoder_path = os.path.join(
        "./models/CLAP",
        f"clap_{classify}_{model_selected_data}_{str(VTC_FRAME_DURATION_SEC).replace(".", "_")}_label_encoder.npy",
    )

    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

    label_encoder = joblib.load(label_encoder_path)
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    # Segment the audio
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    window_size = int(VTC_FRAME_DURATION_SEC * sample_rate)
    step_size = int(window_size * (1 - VTC_OVERLAP))
    starts = np.arange(0, len(audio) - window_size + 1, step_size)

    # Load the CLAP model (e.g., from LAION-CLAP or your local version)
    try:
        from laion_clap import CLAP_Module
        clap_model = CLAP_Module(enable_fusion=False)
        clap_model.load_ckpt()  # assumes env var or default ckpt path
        clap_model.to(device)
        clap_model.eval()
    except Exception as e:
        raise RuntimeError("Error loading CLAP model. Ensure LAION-CLAP is installed and configured.") from e

    # Load classifier and weights
    if not os.path.exists(best_model_weights_path):
        raise FileNotFoundError(f"Classifier weights not found: {best_model_weights_path}")

    # Probe first segment to get embedding dimension
    first_seg = audio[0:window_size]
    first_emb = clap_model.get_audio_embedding_from_data(x=torch.tensor(first_seg).unsqueeze(0).to(device))
    embedding_dim = first_emb.shape[-1]

    classifier = ClapClassifier(embedding_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(best_model_weights_path, map_location=device))
    classifier.eval()

    # Extract embeddings and classify each segment
    all_probs, window_times = [], []

    with torch.no_grad():
        for start in starts:
            segment = audio[start:start + window_size]
            audio_tensor = torch.tensor(segment).unsqueeze(0).to(device)

            # Get CLAP embedding
            embedding = clap_model.get_audio_embedding_from_data(x=audio_tensor)
            logits = classifier(embedding)
            probs = F.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy()[0])
            window_times.append(start / sample_rate)

    probs_array = np.stack(all_probs)
    return class_names, probs_array, np.array(window_times)

import os
import torch
import librosa
import soundfile as sf
import numpy as np
import tempfile
from clap import CLAP  # adjust import if your CLAP class is elsewhere


def prepare_clap_single_file(
    audio_path,
    clap_model_version="2023",
    sample_rate=16000,
    window_len_secs=3.0,
    overlap=0.25,
):
    """
    Extract CLAP embeddings from a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        clap_model_version (str): Version of CLAP model ("2023" or "2024").
        sample_rate (int): Sampling rate for processing.
        window_len_secs (float): Duration of each segment (in seconds).
        overlap (float): Overlap fraction between segments (0.0–1.0).

    Returns:
        np.ndarray: Embeddings for each segment (shape [num_segments, embedding_dim])
        np.ndarray: Segment start times (in seconds)
    """

    # ---- Load audio ----
    print(f"Loading audio: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=sample_rate)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    # ---- Slice into overlapping segments ----
    window_size = int(window_len_secs * sample_rate)
    hop_size = int(window_size * (1 - overlap))
    starts = np.arange(0, len(waveform) - window_size + 1, hop_size)
    print(f"Slicing into {len(starts)} segments (window={window_len_secs}s, overlap={overlap})")

    # ---- Initialize CLAP model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = CLAP(version=clap_model_version, use_cuda=(device == "cuda"))

    embeddings = []
    times = []

    # ---- Temporary file directory ----
    tmp_dir = tempfile.mkdtemp(prefix="clap_single_")

    for i, start in enumerate(starts):
        seg = waveform[start:start + window_size]
        tmp_path = os.path.join(tmp_dir, f"segment_{i}.wav")
        sf.write(tmp_path, seg, samplerate=sample_rate)

        emb = clap_model.get_audio_embeddings([tmp_path])
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        if emb.ndim > 1:
            emb = emb[0]

        embeddings.append(emb)
        times.append(start / sample_rate)

        os.remove(tmp_path)

    embeddings = np.stack(embeddings)
    times = np.array(times)

    print(f"Extracted embeddings: {embeddings.shape}")
    return embeddings, times