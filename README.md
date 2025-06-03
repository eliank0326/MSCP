# Multimodal OSA Early Warning System

## Project Overview

This project aims to develop an intelligent early warning system for Obstructive Sleep Apnea (OSA) based on multimodal sensing. It integrates audio signals, piezoelectric data, Kalman filtering, and chaotic feature modeling techniques to achieve non-contact, early-stage OSA risk detection and alerting.

## Model Architecture

Audio Branch: Extracts features from log-Mel spectrograms using CNN + BiGRU.

Piezo Branch: Processes piezoelectric signals with Kalman filtering, then extracts features using 1D-CNN + GRU.

Chaotic Branches: Both audio and piezo signals are modeled with Takens' embedding and Lyapunov exponent to capture chaotic dynamics.

Fusion Layer: Features from all four paths are concatenated and passed into an MLP classifier to predict OSA risk level or speaker ID.

## Project Structure

```text
project_root/
├── models/
│   └── fusion_model.py          # Main model architecture
├── preprocess/
│   └── dataset_vibravox.py      # Custom dataset loader
├── scripts/
│   ├── preprocess_vibravox.py   # Main data preprocessing pipeline
│   ├── kalman_utils.py          # Kalman filter utilities
│   └── extract_from_parquent.py # Extracts audio, piezo, and labels
├── train/
│   ├── train_vibravox.py        # Model training script
│   └── focal_loss.py            # Focal Loss implementation
```


## Environment Requirements

We recommend managing the environment via Conda:

conda create -n mscp python=3.8
conda activate mscp
pip install -r requirements.txt

Key dependencies include:

torch >= 1.10
librosa
numpy
scikit-learn
matplotlib
tqdm

## Usage

1. Download the Public Dataset
This project uses the open-access VibraVox dataset:
https://huggingface.co/datasets/Cnam-LMSSC/vibravox
(Specifically the speech_clean subset.)
Place the dataset under the data/huggingface directory.

2. Data Preprocessing

python scripts/extract_from_parquent.py
python scripts/preprocess_vibravox.py

3. Train the Model

python train/train_vibravox.py

## Highlights

Multimodal fusion (audio + piezo + chaos-based features)
Kalman filtering enhances signal quality
Chaos dynamics improve model robustness
Focal Loss addresses class imbalance effectively

Feel free to star ⭐, fork 🍴, or open issues 🛠️ to contribute and improve this project together!
