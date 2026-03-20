import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, 1, n_mfcc, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x


class AVDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(AVDeepfakeDetector, self).__init__()
        # Video branch: pretrained ResNet18
        self.video_model = models.resnet18(pretrained=True)
        self.video_model.fc = nn.Identity()  # Remove final FC layer to get features

        # Audio branch: 3-layer CNN
        self.audio_model = AudioCNN()

        # Calculate audio cnn output size dynamically:
        dummy_audio = torch.zeros(1, 1, 40, 44)
        audio_out_dim = self.audio_model(dummy_audio).shape[1]

        video_feat_dim = 512  # ResNet18 feature vector size
        fusion_dim = video_feat_dim + audio_out_dim

        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, video_frames, audio_mfcc):
        # video_frames shape: (batch, num_frames, C, H, W)
        # audio_mfcc shape: (batch, 1, n_mfcc, time)

        batch_size, num_frames, C, H, W = video_frames.shape

        # Flatten frames batch to process with 2D CNN
        video_frames = video_frames.view(batch_size * num_frames, C, H, W)
        video_feats = self.video_model(video_frames)  # (batch_size * num_frames, 512)
        video_feats = video_feats.view(batch_size, num_frames, -1)
        video_feats = video_feats.mean(dim=1)  # Temporal average pooling

        # Process audio MFCC - **No unsqueeze here**
        audio_feats = self.audio_model(audio_mfcc)  # (batch, audio_feat_dim)

        # Concatenate features
        combined = torch.cat((video_feats, audio_feats), dim=1)

        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        out = self.fc2(x)

        return out
