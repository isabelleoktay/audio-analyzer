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
from config import VTC_FRAME_DURATION_SEC, VTC_OVERLAP

logging.basicConfig(level=logging.INFO)


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


def get_clap_model(device, version="2023"):
    """Create a new CLAP instance each time."""
    return CLAP(version=version, use_cuda=(device=="cuda"))


def get_clap_embeddings_windowed(audio_array, sample_rate, clap_model_version,
                                 window_len_secs, overlap, device="cpu"):

    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=0)
    audio_array = audio_array.astype(np.float32)

    window_size = int(window_len_secs * sample_rate)
    hop_size = max(1, int(window_size * (1 - overlap)))
    starts = np.arange(0, len(audio_array) - window_size + 1, hop_size)

    clap_model = get_clap_model(device, version=clap_model_version)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmpfile.name
    tmpfile.close()  # Close so sf.write can access it

    try:
        for start in starts:
            segment = audio_array[start:start + window_size]

            # Overwrite the same temp file
            sf.write(tmp_path, segment, samplerate=sample_rate, format="WAV")
            emb = clap_model.get_audio_embeddings([tmp_path], resample=False)

            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()[0]

            yield emb.astype(np.float32), start / sample_rate

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        del clap_model
        gc.collect()

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

    # Use no_grad + inference_mode for memory efficiency
    with torch.no_grad(), torch.inference_mode():
        for emb, t in get_clap_embeddings_windowed(audio, sample_rate, clap_model_version, window_len_secs, overlap, device):
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            logits = classifier(emb_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
            all_times.append(t)

    # Clean up
    del classifier, emb_tensor, logits
    gc.collect()

    return class_names, np.stack(all_probs), np.array(all_times)
