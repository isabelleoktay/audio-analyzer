from config import PHONATION_DIMENSION, VGGISH, PHONATION_MODEL

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(level=logging.INFO)

def extract_phonation(audio):

    # 1) Prepare audio for model input
    logging.info("Getting audio embeddings...")
    try:
        audio_embeddings = VGGISH(audio)
    except tf.errors.UnknownError as e:
        if "libdevice" in str(e) or "JIT compilation failed" in str(e):
            logging.warning("VGGish GPU JIT failed (libdevice issue), falling back to CPU...")
            with tf.device('/CPU:0'):
                audio_embeddings = VGGISH(audio)
        else:
            raise
    audio_embeddings = audio_embeddings.numpy()
    logging.info("Retrieved audio embeddings.")

    print(f"Audio embeddings shape: {audio_embeddings.shape}")

    if audio_embeddings is None or audio_embeddings.shape[0] == 0:
        raise ValueError("Failed to extract audio embeddings: Embeddings are empty or invalid.")
    
    window_size = PHONATION_DIMENSION
    hop_size = window_size // 2 
    
    if window_size > audio_embeddings.shape[0]:
        raise ValueError("Window size exceeds the dimensions of audio embeddings.")
    
    logging.info("Successfully loaded phonation model.")
    all_preds = []
    
    for start in range(0, audio_embeddings.shape[0] - window_size + 1, hop_size):
        window_feats = audio_embeddings[start:start + window_size]

        window_feats_padded = pad_sequences([window_feats], maxlen=window_size,
                                            dtype='float32', padding='post', truncating='post')
        window_feats_padded = np.expand_dims(window_feats_padded, -1)

        logging.info("Predicting window...")
        try:
            preds = PHONATION_MODEL.predict(window_feats_padded)
        except tf.errors.UnknownError as e:
            if "libdevice" in str(e) or "JIT compilation failed" in str(e):
                logging.warning("PHONATION_MODEL GPU JIT failed, falling back to CPU...")
                with tf.device('/CPU:0'):
                    preds = PHONATION_MODEL.predict(window_feats_padded)
            else:
                raise
        all_preds.append(preds[0])
        logging.info("Successfully predicted window...")

    all_preds = np.array(all_preds)
    print(f"All predictions shape: {all_preds.shape}")
    print(f"All predictions: {all_preds}")

    return all_preds