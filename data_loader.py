"""Data loader and feature extraction for RAVDESS dataset."""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import joblib
from config import (
    DATA_PATH, FEATURES_PATH, EMOTIONS, SAMPLE_RATE, 
    DURATION, N_MFCC, N_MELS, HOP_LENGTH, N_FFT
)


def extract_features(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Extract audio features from a single file.
    
    Features extracted:
    - MFCCs (Mel-frequency cepstral coefficients)
    - Mel spectrogram
    - Chroma features
    - Spectral contrast
    - Zero crossing rate
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad or truncate to fixed length
        max_len = sr * duration
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)), mode='constant')
        else:
            y = y[:max_len]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)
        
        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Extract Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std,           # 80 features
            mel_mean, mel_std,                # 256 features
            chroma_mean, chroma_std,          # 24 features
            contrast_mean, contrast_std,      # 14 features
            [zcr_mean, zcr_std],              # 2 features
            [rms_mean, rms_std]               # 2 features
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_mel_spectrogram(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Extract mel spectrogram for CNN input."""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad or truncate
        max_len = sr * duration
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)), mode='constant')
        else:
            y = y[:max_len]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract metadata.
    
    Filename format: XX-XX-XX-XX-XX-XX-XX.wav
    - Modality (01=full-AV, 02=video-only, 03=audio-only)
    - Vocal channel (01=speech, 02=song)
    - Emotion (01-08)
    - Emotional intensity (01=normal, 02=strong)
    - Statement (01="Kids...", 02="Dogs...")
    - Repetition (01 or 02)
    - Actor (01-24, odd=male, even=female)
    """
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) != 7:
        return None
    
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    }


def load_ravdess_data(data_path=DATA_PATH, use_mel_spec=True):
    """
    Load RAVDESS dataset and extract features.
    
    Args:
        data_path: Path to RAVDESS dataset
        use_mel_spec: If True, extract mel spectrograms; else extract combined features
    
    Returns:
        X: Features array
        y: Labels array
        metadata: DataFrame with file metadata
    """
    features_list = []
    labels = []
    metadata_list = []
    
    print(f"Loading data from: {data_path}")
    
    # Get all actor directories
    actor_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('Actor_')])
    
    for actor_dir in tqdm(actor_dirs, desc="Processing actors"):
        actor_path = os.path.join(data_path, actor_dir)
        
        if not os.path.isdir(actor_path):
            continue
        
        # Get all wav files
        wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            file_path = os.path.join(actor_path, wav_file)
            
            # Parse filename
            file_info = parse_ravdess_filename(wav_file)
            if file_info is None:
                continue
            
            # Only use speech (vocal_channel = 01)
            if file_info['vocal_channel'] != '01':
                continue
            
            # Extract features
            if use_mel_spec:
                feat = extract_mel_spectrogram(file_path)
            else:
                feat = extract_features(file_path)
            
            if feat is None:
                continue
            
            # Get emotion label
            emotion_code = file_info['emotion']
            emotion_label = EMOTIONS.get(emotion_code, 'unknown')
            
            features_list.append(feat)
            labels.append(emotion_label)
            
            # Store metadata
            file_info['file_path'] = file_path
            file_info['emotion_label'] = emotion_label
            file_info['gender'] = 'male' if int(file_info['actor']) % 2 == 1 else 'female'
            metadata_list.append(file_info)
    
    X = np.array(features_list)
    y = np.array(labels)
    metadata = pd.DataFrame(metadata_list)
    
    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Emotion distribution:\n{pd.Series(y).value_counts()}")
    
    return X, y, metadata


def save_features(X, y, metadata, prefix='ravdess'):
    """Save extracted features to disk."""
    np.save(os.path.join(FEATURES_PATH, f'{prefix}_X.npy'), X)
    np.save(os.path.join(FEATURES_PATH, f'{prefix}_y.npy'), y)
    metadata.to_csv(os.path.join(FEATURES_PATH, f'{prefix}_metadata.csv'), index=False)
    print(f"Features saved to {FEATURES_PATH}")


def load_saved_features(prefix='ravdess'):
    """Load previously saved features."""
    X = np.load(os.path.join(FEATURES_PATH, f'{prefix}_X.npy'))
    y = np.load(os.path.join(FEATURES_PATH, f'{prefix}_y.npy'), allow_pickle=True)
    metadata = pd.read_csv(os.path.join(FEATURES_PATH, f'{prefix}_metadata.csv'))
    return X, y, metadata


if __name__ == "__main__":
    # Extract and save features
    print("Extracting mel spectrogram features...")
    X, y, metadata = load_ravdess_data(use_mel_spec=True)
    save_features(X, y, metadata, prefix='ravdess_mel')
    
    print("\nExtracting combined features...")
    X, y, metadata = load_ravdess_data(use_mel_spec=False)
    save_features(X, y, metadata, prefix='ravdess_combined')
