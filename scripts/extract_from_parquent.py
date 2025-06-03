import os
import io
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# === 参数设置 ===
data_dir = "../data/huggingface/speech_clean"
save_audio_dir = "../data/huggingface/audio"
save_piezo_dir = "../data/huggingface/piezo"
meta_path = "../data/huggingface/metadata.csv"

os.makedirs(save_piezo_dir, exist_ok=True)
os.makedirs(save_audio_dir, exist_ok=True)

meta_list = []

# === 遍历所有parquet文件 ===
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
print(f" 共检测到{len(files)}个parquet 文件， 开始处理...")

for filename in tqdm(files, desc="Extracting audio + piezo"):
    path = os.path.join(data_dir, filename)
    table = pq.read_table(path)
    df = table.to_pandas()

    for idx, row in df.iterrows():
        try:
            # === 音频 ===
            audio_bytes = row['audio.headset_microphone']['bytes']
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype = 'float32')

            # === 压电 ===
            piezo_bytes = row['audio.temple_vibration_pickup']['bytes']
            piezo_data, sr_p = sf.read(io.BytesIO(piezo_bytes), dtype = 'float32')

            # === 标签 ===
            gender = row['gender']
            speaker_id = row['speaker_id']

            # === 保存路径 ===
            base_name = f"{speaker_id}_{idx}"
            audio_path = os.path.join(save_audio_dir, f"{base_name}.wav")
            piezo_path = os.path.join(save_piezo_dir, f"{base_name}.wav")

            sf.write(audio_path, audio_data, sr)
            sf.write(piezo_path, piezo_data, sr_p)

            # === 保存元信息 ===
            meta_list.append({
                "audio_path": audio_path,
                "piezo_path": piezo_path,
                "gender": gender,
                "speaker_id": speaker_id
            })
        except Exception as e:
            print(f" Error at row{idx} in {filename}: {e}")

# === 保存metadata ===
meta_df = pd.DataFrame(meta_list)

# 添加 speaker_id 编号列
le = LabelEncoder()
meta_df['speaker_label'] = le.fit_transform(meta_df['speaker_id'])

meta_df.to_csv(meta_path, index = False)
print(f" 完成提取并保存至{meta_path}")









