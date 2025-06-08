import tensorflow_hub as hub
import numpy as np

print("Loading VGGish model...")
vggish = hub.load('https://tfhub.dev/google/vggish/1')
print("Successfully loaded VGGish model.")

# Test with dummy input
dummy_audio = np.random.rand(16000).astype(np.float32)  # 1 second of random audio
embeddings = vggish(dummy_audio)
print("Embeddings shape:", embeddings.shape)