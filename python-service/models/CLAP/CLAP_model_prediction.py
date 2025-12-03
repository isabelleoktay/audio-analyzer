import gc
import os
import torch
import logging
import tempfile
import torchaudio
import numpy as np
import torch.nn.functional as F
import soundfile as sf
from msclap import CLAP
from sklearn.preprocessing import LabelEncoder
from config import VTC_FRAME_DURATION_SEC, VTC_OVERLAP

logging.basicConfig(level=logging.INFO)


GLOBAL_CLAP = None

def get_clap_model(device, version="2023"):
    global GLOBAL_CLAP
    if GLOBAL_CLAP is None:
        GLOBAL_CLAP = CLAP(version=version, use_cuda=(device=="cuda"))
    return GLOBAL_CLAP


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


def get_clap_embeddings_windowed(audio_array, sample_rate, clap_model_version, window_len_secs, overlap, device="cpu"):
    audio_array = audio_array.mean(axis=0) if audio_array.ndim == 2 else audio_array
    window_size = int(window_len_secs * sample_rate)
    hop_size = int(window_size * (1 - overlap))
    starts = np.arange(0, len(audio_array) - window_size + 1, hop_size)

    clap_model = get_clap_model(device, version=clap_model_version)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
    tmp_path = tmpfile.name

    for start in starts:
        segment = audio_array[start:start + window_size]
        sf.write(tmp_path, segment, samplerate=sample_rate)
        emb = clap_model.get_audio_embeddings([tmp_path], resample=False)
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()[0]
        yield emb.astype(np.float32), start / sample_rate  # yield one embedding at a time

    tmpfile.close()


def clap_extract_features_and_predict(
    audio_path: str,
    best_model_weights_path: str,
    classify: str = "pitch",
    sample_rate: int = 16000,
    window_len_secs: float = VTC_FRAME_DURATION_SEC,
    overlap: float = VTC_OVERLAP,
    clap_model_version="2023",
    device: str = "cpu",
):
    logging.info("Extracting CLAP features and predicting...")

    # Load label encoder
    label_encoder_path = os.path.join(
        "./models/CLAP/",
        f"{classify}_{str(window_len_secs).replace('.', '_')}_label_encoder_gendered.npy",
    )
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
    class_names = np.load(label_encoder_path, allow_pickle=True)
    num_classes = len(class_names)

    # Load audio
    audio, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sample_rate:
        audio = torchaudio.functional.resample(audio, orig_sr, sample_rate)
    audio = audio[0].numpy().astype(np.float32)

    # Load classifier weights
    if not os.path.exists(best_model_weights_path):
        raise FileNotFoundError(f"Classifier weights not found: {best_model_weights_path}")

    # Probe first window to get embedding dimension
    first_emb, _ = next(get_clap_embeddings_windowed(audio, sample_rate, clap_model_version, window_len_secs, overlap, device))
    embedding_dim = first_emb.shape[-1]

    # Initialize classifier
    classifier = ClapClassifier(embedding_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(best_model_weights_path, map_location=device))
    classifier.eval()

    all_probs = []
    all_times = []

    with torch.no_grad():
        for emb, t in get_clap_embeddings_windowed(audio, sample_rate, clap_model_version, window_len_secs, overlap, device):
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            logits = classifier(emb_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
            all_times.append(t)

    global GLOBAL_CLAP
    del classifier
    if GLOBAL_CLAP is not None:
        del GLOBAL_CLAP
        GLOBAL_CLAP = None

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return class_names, np.stack(all_probs), np.array(all_times)
