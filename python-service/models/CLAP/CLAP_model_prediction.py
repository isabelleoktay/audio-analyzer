import os
import torch
import numpy as np
import torch.nn.functional as F
import librosa
import os
import tempfile
import numpy as np
import soundfile as sf
from msclap import CLAP
from sklearn.preprocessing import LabelEncoder
from config import VTC_FRAME_DURATION_SEC, VTC_OVERLAP


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


def get_clap_embeddings_from_audio_array(
    audio_array,
    sample_rate=16000,
    clap_model_version="2023",
    window_len_secs=VTC_FRAME_DURATION_SEC,
    overlap=VTC_OVERLAP,
    device = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Extract CLAP embeddings from an in-memory audio array
    by saving temporary segment WAVs (consistent with training).
    """

    # Ensure mono
    if audio_array.ndim == 2:
        audio_array = np.mean(audio_array, axis=0)

    # Slice into overlapping segments
    window_size = int(window_len_secs * sample_rate)
    hop_size = int(window_size * (1 - overlap))
    starts = np.arange(0, len(audio_array) - window_size + 1, hop_size)

    # Load the CLAP model (e.g., from LAION-CLAP or your local version)
    try:
        clap_model = CLAP(version=clap_model_version, use_cuda=(device == "cuda"))
    except Exception as e:
        raise RuntimeError("Error loading CLAP model. Ensure LAION-CLAP is installed and configured.") from e


    embeddings, times = [], []

    # Temporary directory for segment files
    tmp_dir = tempfile.mkdtemp(prefix="clap_segments_")

    try:
        for i, start in enumerate(starts):
            seg = audio_array[start:start + window_size]
            # Save temporary segment as WAV
            tmp_path = os.path.join(tmp_dir, f"segment_{i}.wav")
            sf.write(tmp_path, seg, samplerate=sample_rate)
            # Get embedding exactly like training
            emb = clap_model.get_audio_embeddings([tmp_path])
            # Convert to numpy
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            if emb.ndim > 1:
                emb = emb[0]
            embeddings.append(emb)
            times.append(start / sample_rate)
            # Clean up temporary file
            os.remove(tmp_path)

    finally:
        # Cleanup temporary directory
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    embeddings = np.stack(embeddings)
    times = np.array(times)

    print(f"Extracted embeddings (via temp files): {embeddings.shape}")
    return embeddings, times


def clap_extract_features_and_predict(
    audio_path: str,
    best_model_weights_path: str,
    classify: str = "pitch",
    sample_rate: int = 16000,
    window_len_secs: float = VTC_FRAME_DURATION_SEC,
    overlap: float = VTC_OVERLAP,
    clap_model_version="2023",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Extracts CLAP embeddings from a single audio file and passes them through
    a trained CLAP classifier loaded from a .pth checkpoint.
    """

    # Load label encoder
    label_encoder_path = os.path.join(
        "/content/drive/MyDrive/DATASETS/VOCALSET/ORG_VOCALSET_SPLIT/clap_org_split",
        f"{classify}_{str(VTC_FRAME_DURATION_SEC).replace(".", "_")}_label_encoder_classes_gendered.npy",
    )

    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

    class_names = np.load(label_encoder_path, allow_pickle=True)
    num_classes = len(class_names)
    le = LabelEncoder()
    le.fit(class_names)  

    # Segment the audio
    audio, _ = librosa.load(audio_path, sr=sample_rate)

    # Load classifier and weights
    if not os.path.exists(best_model_weights_path):
        raise FileNotFoundError(f"Classifier weights not found: {best_model_weights_path}")
    
    # Get embeddings for file
    embeddings, times = get_clap_embeddings_from_audio_array(
        audio_array=audio,
        sample_rate=sample_rate,
        clap_model_version=clap_model_version,
        window_len_secs=window_len_secs,
        overlap=overlap,
    )

    # Probe first segment to get embedding dimension
    first_emb = embeddings[0]
    embedding_dim = first_emb.shape[-1]
    print(f"Embedding dimension: {embedding_dim}")

    classifier = ClapClassifier(embedding_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(best_model_weights_path, map_location=device))
    classifier.eval()

    # Extract embeddings and classify each segment
    all_probs, window_times = [], []

    with torch.no_grad():
      for i, emb in enumerate(embeddings):
          emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
          logits = classifier(emb_tensor)
          probs = F.softmax(logits, dim=-1)
          all_probs.append(probs.cpu().numpy()[0])
          window_times.append(times[i])   

    probs_array = np.stack(all_probs)
    return class_names, probs_array, np.array(window_times)