import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess.dataset_vibravox import VibravoxDataset
from models.fusion_model import FusionModel
from focal_loss import FocalLoss


# === 配置 ===
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../data/preprocessed_vibravox"
NUM_CLASSES = 188
SAVE_PATH = "../results/checkpoints/fusion_model_best.pth"
os.makedirs("checkpoints", exist_ok=True)

# === Dataset & Loader ===
full_dataset = VibravoxDataset(DATA_PATH)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# === Model & Loss & Optimizer ===
model = FusionModel(num_classes=NUM_CLASSES).to(DEVICE)
criterion = FocalLoss(gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# === Training ===
best_val_acc = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc = f"Epoch {epoch}/{EPOCHS}")

    for mel, piezo, chaos, label in pbar:
        mel, piezo, chaos, label = mel.to(DEVICE), piezo.to(DEVICE), chaos.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        output = model(mel, piezo, chaos)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        preds = output.argmax(dim = 1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        pbar.set_postfix(acc = correct / total, loss = total_loss / total)

    train_acc = correct / total
    train_loss = total_loss / total

    # === Validation ===
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for mel, piezo, chaos, label in val_loader:
            mel, piezo, chaos, label = mel.to(DEVICE), piezo.to(DEVICE), chaos.to(DEVICE), label.to(DEVICE)
            output = model(mel, piezo, chaos)
            loss = criterion(output, label)
            val_loss += loss.item() * label.size(0)
            val_correct += (output.argmax(1) == label).sum().item()
            val_total += label.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    print(f"Epoch {epoch}: Trian Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    # === Early Stopping ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print("\t New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch}.")
            break

print(f"Training complete. Best Val Acc = {best_val_acc:.4f}")


# import os
# import torch10
# import pandas as pd
#
# from torch import nn
# from torch.utils.data import DataLoader, random_split
# from models.fusion_model import FusionModel
# from preprocess.dataset_vibravox import VibravoxDataset
# from focal_loss import FocalLoss
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
#
#
# # === 参数设置 ===
# DATA_DIR = "../data/preprocessed_vibravox"
# BATCH_SIZE = 8
# EPOCHS = 30
# LR = 1e-3
# PATIENCE = 5
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LOG_PATH = "../results/logs/train_vibravox_log.csv"
# MODEL_SAVE_PATH = "../results/checkpoints/best_model_vibravox.pt"
#
# # === 数据准备 ===
# all_paths = [
#     os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pt")
# ]
#
# # data = torch.load(all_paths[0])
# # print("Available keys in this file:", data.keys())
#
# # === 读取 speaker_label 用于 stratify 分层抽样 ===
# labels = [torch.load(p)['label'] for p in all_paths]
# unique_labels = set(labels)
# print(sorted(unique_labels))
#
# train_paths, val_paths = train_test_split(
#     all_paths, test_size=0.2, random_state=42, stratify=labels
# )
#
# train_dataset = VibravoxDataset(train_paths)
# val_dataset = VibravoxDataset(val_paths)
# train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
# val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
#
#
# # === 模型定义 ===
# model = FusionModel(num_classes=188).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-4)
# criterion = FocalLoss(gamma = 2.0)
#
# # === 训练准备 ===
# beat_val_acc = 0
# patience_counter = 0
# logs = []
#
# # === 训练循环 ===
# for epoch in range(EPOCHS):
#     model.train()
#     train_loss, train_correct, total = 0, 0, 0
#     loop = tqdm(train_loader, desc = f"Epoch {epoch + 1}/{EPOCHS}")
#
#     for mel, piezo, label in loop:
#         mel, piezo, label = mel.to(DEVICE), piezo.to(DEVICE), label.to(DEVICE)
#         optimizer.zero_grad()
#         # print("mel shape:", mel.shape)
#         # print("piezo shape:", piezo.shape)
#         output = model(mel,piezo)
#         loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item() * label.size(0)
#         train_correct += (output.argmax(dim=1) == label).sum().item()
#         total += label.size(0)
#         loop.set_postfix(loss = loss.item(), acc = train_correct/total)
#
#     train_acc = train_correct / total
#     train_loss /= total
#
#     # === 验证 ===
#     model.eval()
#     val_correct, val_total = 0,0
#     with torch.no_grad():
#         for mel, piezo, label in val_loader:
#             mel, piezo, label = mel.to(DEVICE), piezo.to(DEVICE), label.to(DEVICE)
#             output = model(mel, piezo)
#             val_correct += (output.argmax(dim = 1) == label).sum().item()
#             val_total += label.size(0)
#
#     val_acc = val_correct / val_total
#     logs.append([epoch + 1, train_loss, train_acc, val_acc])
#
#     print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc: {val_acc:.4f}")
#
#     if val_acc > beat_val_acc:
#         best_val_acc = val_acc
#         patience_counter = 0
#         torch.save(model.state_dict(), MODEL_SAVE_PATH)
#     else:
#         patience_counter += 1
#         if patience_counter >= PATIENCE:
#             print(f"Early stopping at epoch {epoch + 1}. Best val acc: {best_val_acc:.4f}")
#             break
#
# # === 保存日志 ===
# log_df = pd.DataFrame(logs, columns = ["Epoch", "TrainLoss", "TrainAcc", "ValAcc"])
# log_df.to_csv(LOG_PATH, index = False)






