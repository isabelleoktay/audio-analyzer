import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def process_chunk(audio, sr, start, end, min_note, max_note, n_fft, hop_length):
    """Process a chunk of audio and extract features."""
    chunk = audio[start:end]
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length).astype(np.float32)
    rms = librosa.feature.rms(y=chunk, frame_length=n_fft, hop_length=hop_length).astype(np.float32)
    pitches = librosa.yin(chunk, fmin=librosa.note_to_hz(min_note), fmax=librosa.note_to_hz(max_note), sr=sr, hop_length=hop_length).astype(np.float32)
    zcr = librosa.feature.zero_crossing_rate(y=chunk, frame_length=n_fft, hop_length=hop_length).astype(np.float32)
    
    # Return start index and extracted features
    return start, mfccs.shape[1], rms.shape[1], pitches.shape[0], zcr.shape[1], mfccs, rms, pitches, zcr  

def extract_features_parallel(audio, sr, min_note, max_note, n_fft=2048, hop_length=512):
    """Parallelize feature extraction with correct reordering and concatenation."""
    n_workers = min(cpu_count(), len(audio) // (hop_length * 10))

    frames_per_worker = (len(audio) // hop_length) // n_workers
    chunk_size = frames_per_worker * hop_length
    overlap = n_fft  

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i in range(n_workers):
            start = max(0, i * chunk_size - overlap)
            end = min(len(audio), (i + 1) * chunk_size + overlap)

            futures.append(
                executor.submit(process_chunk, audio, sr, start, end, min_note, max_note, n_fft, hop_length)
            )

    results = sorted([future.result() for future in futures], key=lambda x: x[0])

    # Unpack results
    _, mfcc_sizes, rms_sizes, pitch_sizes, zcr_sizes, mfccs_list, rms_list, pitches_list, zcr_list = zip(*results)

    # Ensure proper alignment for concatenation
    total_mfcc_frames = sum(mfcc_sizes)
    total_rms_frames = sum(rms_sizes)
    total_pitch_frames = sum(pitch_sizes)
    total_zcr_frames = sum(zcr_sizes)

    mfccs = np.zeros((13, total_mfcc_frames), dtype=np.float32)
    rms = np.zeros((1, total_rms_frames), dtype=np.float32)
    pitches = np.zeros(total_pitch_frames, dtype=np.float32)
    zcr = np.zeros((1, total_zcr_frames), dtype=np.float32)

    # Stitch features in correct order
    mfcc_index, rms_index, pitch_index, zcr_index = 0, 0, 0, 0

    for i in range(n_workers):
        mfccs[:, mfcc_index:mfcc_index + mfcc_sizes[i]] = mfccs_list[i]
        rms[:, rms_index:rms_index + rms_sizes[i]] = rms_list[i]
        pitches[pitch_index:pitch_index + pitch_sizes[i]] = pitches_list[i]
        zcr[:, zcr_index:zcr_index + zcr_sizes[i]] = zcr_list[i]

        mfcc_index += mfcc_sizes[i]
        rms_index += rms_sizes[i]
        pitch_index += pitch_sizes[i]
        zcr_index += zcr_sizes[i]

    return mfccs, rms, pitches, zcr
