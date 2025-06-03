import os
import torch
import pandas as pd
import torchaudio
import librosa
import numpy as np

from tqdm import tqdm
from kalman_utils import apply_kalman_filter
from chaos_utils import extract_chaos_features

# === 参数设置 ===
META_PATH = "../data/huggingface/metadata.csv"
SAVE_DIR = "../data/preprocessed_vibravox"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 3
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_DURATION

os.makedirs(SAVE_DIR, exist_ok=True)

meta_df = pd.read_csv(META_PATH)

for i, row in tqdm(meta_df.iterrows(), total = len(meta_df), desc="Preprocessing audio+piezo"):
    audio_path = row["audio_path"].replace("\\","/")
    piezo_path = row["piezo_path"].replace("\\","/")
    label = row["speaker_label"]    #speaker_id 编码后的值

    audio_id = os.path.splitext(os.path.basename(audio_path))[0]

    try:
        audio_waveform, sr_a = torchaudio.load(audio_path)
        piezo_waveform, sr_p = torchaudio.load(piezo_path)

        if len(audio_waveform.shape)>1: # 多通道 -> 单通道
            audio_waveform = audio_waveform.mean(dim = 0)
        else:
            audio_waveform = audio_waveform.squeeze(0)

        if len(piezo_waveform.shape) > 1:
            piezo_waveform = piezo_waveform.mean(dim = 0)
        else:
            piezo_waveform = piezo_waveform.squeeze(0)

        audio_waveform = torchaudio.functional.resample(audio_waveform, sr_a, SAMPLE_RATE)
        piezo_waveform = torchaudio.functional.resample(piezo_waveform, sr_p, SAMPLE_RATE)

        # padding
        if audio_waveform.shape[-1] < SEGMENT_LENGTH:
            pad = SEGMENT_LENGTH - audio_waveform.shape[-1]
            audio_waveform = torch.nn.functional.pad(audio_waveform, (0, pad))
        if piezo_waveform.shape[-1] < SEGMENT_LENGTH:
            pad = SEGMENT_LENGTH - piezo_waveform.shape[-1]
            piezo_waveform = torch.nn.functional.pad(piezo_waveform, (0, pad))

        audio_waveform = audio_waveform[:SEGMENT_LENGTH]
        piezo_waveform = piezo_waveform[:SEGMENT_LENGTH]

        # Mel 特征
        mel = librosa.feature.melspectrogram(
            y=audio_waveform.numpy(), sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).contiguous()  # (1, 64, T)

        # Kalman 压电
        piezo_filtered = apply_kalman_filter(piezo_waveform.numpy())
        piezo_filtered = torch.tensor(piezo_filtered, dtype=torch.float32).unsqueeze(-1).contiguous()  # (48000, 1)

        # === 混沌特征 ===
        audio_chaos = extract_chaos_features(audio_waveform.numpy())
        piezo_chaos = extract_chaos_features(piezo_waveform.numpy())
        chaos_feat = np.concatenate([
            np.array([audio_chaos["lyapunov"]]),
            np.array([audio_chaos["spectral_entropy"]]),
            np.array([piezo_chaos["lyapunov"]]),
            np.array([piezo_chaos["spectral_entropy"]])
        ])
        chaos_feat = torch.tensor(chaos_feat, dtype = torch.float32)

        sample = {
            "mel": mel,                  # [1, 64, T]
            "piezo": piezo_filtered,     # [48000, 1]
            "chaos": chaos_feat,         #(4,)
            "label": label
        }

        torch.save(sample, os.path.join(SAVE_DIR, f"{audio_id}.pt"))

    except Exception as e:
        print(f"Failed processing {audio_id}: {e}")









