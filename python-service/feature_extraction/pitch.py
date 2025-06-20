import numpy as np
import librosa
from essentia.standard import PitchCREPE
from config import CREPE_MODEL_PATH, CREPE_EXTRACTOR

class PitchExtractorBase:
    def extract_pitch(self, audio, **kwargs):
        raise NotImplementedError("Subclasses must implement extract_pitch")

class CrepePitchExtractor(PitchExtractorBase):
    def extract_pitch(self, audio, hop_ms, batch):
        # Check if we can use the pre-loaded extractor with matching parameters
        if (CREPE_EXTRACTOR is not None and 
            hasattr(CREPE_EXTRACTOR, 'hopSize') and 
            CREPE_EXTRACTOR.hopSize == hop_ms and CREPE_EXTRACTOR.batchSize == batch):
            extractor = CREPE_EXTRACTOR
        else:
            # Create new extractor with custom parameters
            extractor = PitchCREPE(
                graphFilename=CREPE_MODEL_PATH,
                hopSize=hop_ms,
                batchSize=batch
            )

        times, pitch, conf, _ = extractor(audio)
        return np.array(times), np.array(pitch), np.array(conf)

class LibrosaPitchExtractor(PitchExtractorBase):
    def extract_pitch(self, audio, sr, hop_length):
        # use librosa.pyin for pitch estimation
        pitch, _, conf = librosa.pyin(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            fmin=librosa.note_to_hz('C2'),  # minimum frequency (e.g., C2 ~ 65 Hz)
            fmax=librosa.note_to_hz('C7')   # maximum frequency (e.g., C7 ~ 2093 Hz)
        )
        times = librosa.frames_to_time(np.arange(len(pitch)), sr=sr, hop_length=hop_length)
        
        # interpolate NaN values with the nearest valid pitch value
        if np.any(np.isnan(pitch)):
            valid_indices = np.where(~np.isnan(pitch))[0]
            if valid_indices.size > 0:
                pitch = np.interp(
                    np.arange(len(pitch)),  # indices of all frames
                    valid_indices,          # indices of valid frames
                    pitch[valid_indices]    # valid pitch values
                )
            else:
                pitch = np.zeros_like(pitch)  # if no valid pitch values exist, set all to zero
        
        return np.array(times), np.array(pitch), np.array(conf)