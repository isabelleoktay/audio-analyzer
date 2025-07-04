from config import PHONATION_DIMENSION, VGGISH, PHONATION_MODEL

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(level=logging.INFO)

def extract_phonation(audio):

    # 1) Prepare audio for model input
    logging.info("Getting audio embeddings...")
    audio_embeddings = VGGISH(audio)
    audio_embeddings = audio_embeddings.numpy()  # Convert to numpy array
    logging.info("Retrieved audio embeddings.")

    if audio_embeddings is None or audio_embeddings.shape[0] == 0:
        raise ValueError("Failed to extract audio embeddings: Embeddings are empty or invalid.")
    
    window_size = PHONATION_DIMENSION
    hop_size = window_size // 2 
    
    if window_size > audio_embeddings.shape[0]:
        raise ValueError("Window size exceeds the dimensions of audio embeddings.")
    
    logging.info("Successfully loaded phonation model.")
    all_preds = []
    
    # predict class for window based on dataset audio length dimension and hop through longer audio
    for start in range(0, audio_embeddings.shape[0] - window_size + 1, hop_size):
        window_feats = audio_embeddings[start:start + window_size]

        # Pad or truncate just in case (shouldn't be needed if slicing exactly)
        window_feats_padded = pad_sequences([window_feats], maxlen=window_size,
                                            dtype='float32', padding='post', truncating='post')
        window_feats_padded = np.expand_dims(window_feats_padded, -1)  # add channel dim

        logging.info("Predicting window...")
        preds = PHONATION_MODEL.predict(window_feats_padded)  # shape: (1, num_classes)
        all_preds.append(preds[0])  # collect the prediction vector
        logging.info("Successfully predicted window...")

    all_preds = np.array(all_preds)  # shape: (num_windows, num_classes)
    print(f"All predictions shape: {all_preds.shape}")
    print(f"All predictions: {all_preds}")

    return all_preds