#!/usr/bin/env python3
"""
Video Radioactive Poison Verification Script

This script verifies that video poisoning works by:
1. Training a simple 3D CNN on clean + poisoned videos
2. Detecting the signature in the trained model

Usage:
    python verification/verify_video_poison.py \
        --data verification_video_data \
        --signature verification_video_data/signature.json \
        --epochs 10
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

# Add poison-core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from video_poison import VideoRadioactiveMarker, VideoRadioactiveDetector


class Simple3DCNN(nn.Module):
    """
    Simple 3D CNN for video classification.

    Architecture:
    - 3D Conv layers to process spatiotemporal features
    - Pooling to reduce dimensions
    - FC layers for classification
    """

    def __init__(self, num_classes=2, num_frames=16):
        super(Simple3DCNN, self).__init__()

        self.num_frames = num_frames

        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Calculate flattened size: 256 * (T/2) * (H/4) * (W/4)
        # For 16 frames, 112x112 input: 256 * 8 * 28 * 28 = 1,605,632
        # But we need to compute this dynamically
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 7, 7))  # Output: 256 * 4 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x):
        """Extract features before final classification layer."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))

        return x


class VideoDataset(Dataset):
    """Dataset for video files."""

    def __init__(self, video_paths, labels, num_frames=16):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video
        frames = self._load_video(video_path)

        if frames is None:
            # Return dummy data if loading fails
            frames = torch.zeros(3, self.num_frames, 112, 112)

        return frames, label

    def _load_video(self, video_path):
        """Load video and extract frames."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return None

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (112, 112))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if len(frames) < self.num_frames:
                return None

            # Convert to tensor (T, H, W, C) -> (C, T, H, W)
            video_tensor = np.stack(frames, axis=0)  # (T, H, W, C)
            video_tensor = torch.from_numpy(video_tensor).float()
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            video_tensor = video_tensor / 255.0  # Normalize

            return video_tensor

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None


def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    """Train the model."""
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


def verify_poison(data_dir, signature_path, epochs=10, device='cpu'):
    """
    Verify video poisoning by training and detecting.

    Args:
        data_dir: Directory containing clean/ and poisoned/ subdirectories
        signature_path: Path to signature JSON file
        epochs: Number of training epochs
        device: 'cpu' or 'cuda'
    """
    print("=" * 60)
    print("Video Radioactive Poison Verification")
    print("=" * 60)
    print()

    data_path = Path(data_dir)
    clean_dir = data_path / 'clean'
    poisoned_dir = data_path / 'poisoned'

    # Load video paths
    clean_videos = list(clean_dir.glob('*.mp4'))
    poisoned_videos = list(poisoned_dir.glob('*.mp4'))

    print(f"Found {len(clean_videos)} clean videos")
    print(f"Found {len(poisoned_videos)} poisoned videos")
    print()

    if len(clean_videos) == 0 or len(poisoned_videos) == 0:
        print("Error: Need both clean and poisoned videos!")
        return

    # Create dataset
    video_paths = clean_videos + poisoned_videos
    labels = [0] * len(clean_videos) + [1] * len(poisoned_videos)  # 0=clean, 1=poisoned

    dataset = VideoDataset(video_paths, labels, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Create model
    print("Creating Simple3DCNN model...")
    model = Simple3DCNN(num_classes=2, num_frames=16)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    train_model(model, dataloader, criterion, optimizer, device, epochs=epochs)

    print("\n" + "=" * 60)
    print("Testing Detection on Trained Model")
    print("=" * 60)
    print()

    # Load signature and detect
    detector = VideoRadioactiveDetector(str(signature_path), device=device)

    # Use clean videos as test set
    test_videos = [str(v) for v in clean_videos[:min(5, len(clean_videos))]]

    print(f"Testing with {len(test_videos)} clean videos...")
    is_poisoned, correlation = detector.detect(model, test_videos, threshold=0.05)

    print()
    print("🎯 Detection Result:")
    print(f"   Poisoned: {is_poisoned}")
    print(f"   Correlation: {correlation:.6f}")
    print(f"   Threshold: 0.05")

    if is_poisoned:
        ratio = correlation / 0.05
        print(f"   Ratio: {ratio:.1f}x above threshold")
        print()
        print("✅ SUCCESS! The poison signature was detected in the trained model!")
    else:
        print()
        print("❌ FAILED! No signature detected. Try:")
        print("   1. More poisoned videos (increase ratio)")
        print("   2. Higher epsilon (stronger poisoning)")
        print("   3. More training epochs")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify video radioactive poisoning')
    parser.add_argument('--data', required=True, help='Data directory with clean/ and poisoned/')
    parser.add_argument('--signature', required=True, help='Path to signature JSON file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    verify_poison(args.data, args.signature, args.epochs, args.device)
