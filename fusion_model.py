import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    def __init__(self, input_channels = 1, input_freq_bins = 64, dropout_p = 0.3):
        super(CNNBranch, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_p)
        )
        self.output_dim = (input_freq_bins//4)*64

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2) #[B, T, C, F]
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)
        return x

class GRUBranch(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout_p=0.3):
        super(GRUBranch, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])  # 最后一个时间步输出
        return out

class FusionModel(nn.Module):
    def __init__(self, audio_input_channels = 1, input_freq_bins = 64,piezo_input_channels = 1, num_classes = 188, dropout_p = 0.3):
        super(FusionModel, self).__init__()
        self.audio_cnn = CNNBranch(input_channels = audio_input_channels, input_freq_bins=input_freq_bins, dropout_p=dropout_p)
        self.audio_gru = GRUBranch(input_dim = self.audio_cnn.output_dim, dropout_p=dropout_p)
        self.piezo_branch = GRUBranch(input_dim=piezo_input_channels, dropout_p=dropout_p)

        self.chaos_fc = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Dropout(p = dropout_p)
        )

        fusion_dim = 128 * 2 + 128 * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(p = dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel, piezo, chaos):
        audio_feat = self.audio_cnn(mel)
        audio_feat = self.audio_gru(audio_feat)
        piezo_feat = self.piezo_branch(piezo)
        chaos_feat = self.chaos_fc(chaos.float())

        fused = torch.cat([audio_feat, piezo_feat, chaos_feat], dim = -1)
        output = self.classifier(fused)
        return output
