import logging
import tensorflow_hub as hub
from keras.models import load_model
from essentia.standard import PitchCREPE
from .config import PHONATION_MODEL_PATH, PHONATION_VGGISH_URL, CREPE_MODEL_PATH, CREPE_HOP_SIZE_MS

def load_vggish_model():
    """
    Load and return the VGGish model from TensorFlow Hub.
    """
    try:
        logging.info("Loading VGGish model...")
        model = hub.load(PHONATION_VGGISH_URL)
        logging.info("Successfully loaded VGGish model.")
        return model
    except Exception as e:
        logging.error(f"Failed to load VGGish model: {e}")
        return None
    
def load_phonation_model():
    """
    Load and return the phonation classification model.
    """
    try:
        logging.info("Loading phonation model...")
        model = load_model(PHONATION_MODEL_PATH)
        logging.info("Successfully loaded phonation model.")
        return model
    except Exception as e:
        logging.error(f"Failed to load phonation model from path '{PHONATION_MODEL_PATH}': {e}")
        return None
    
def load_crepe_extractor():
    """
    Load and return the CREPE pitch extractor with default parameters.
    """
    try:
        logging.info("Loading CREPE extractor...")
        # Use default parameters from config
        extractor = PitchCREPE(
            graphFilename=CREPE_MODEL_PATH,
            hopSize=CREPE_HOP_SIZE_MS,
            batchSize=32  # default batch size
        )
        logging.info("Successfully loaded CREPE extractor.")
        return extractor
    except Exception as e:
        logging.error(f"Failed to load CREPE extractor: {e}")
        return None

# Initialize the model when this module is imported
VGGISH = load_vggish_model()
PHONATION_MODEL = load_phonation_model()
CREPE_EXTRACTOR = load_crepe_extractor()