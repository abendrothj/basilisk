#!/usr/bin/env python3
"""
Proper validation with train/test split and dropout regularization.

This is the REAL test:
- 50 total videos (enough to force learning patterns vs memorization)
- 70/30 train/test split
- Dropout + weight decay to prevent overfitting
- Test detection on HELD-OUT videos only
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'verification'))

from video_poison import VideoRadioactiveMarker, VideoRadioactiveDetector
from verify_video_poison import Simple3DCNN, VideoDataset, train_model


def main():
    print("=" * 60)
    print("PROPER VALIDATION: Train/Test Split + Regularization")
    print("=" * 60)
    print()

    data_dir = 'verification_video_data_large'
    signature_path = f'{data_dir}/signature.json'

    if not Path(data_dir).exists():
        print("ERROR: Run create_video_dataset.py with --clean 25 --poisoned 25")
        sys.exit(1)

    # Load all videos
    clean_videos = list((Path(data_dir) / 'clean').glob('*.mp4'))
    poisoned_videos = list((Path(data_dir) / 'poisoned').glob('*.mp4'))

    print(f"Total dataset: {len(clean_videos)} clean + {len(poisoned_videos)} poisoned")

    if len(clean_videos) < 20 or len(poisoned_videos) < 20:
        print("ERROR: Need at least 20 of each for proper split")
        sys.exit(1)

    # Create labels
    all_videos = clean_videos + poisoned_videos
    all_labels = [0] * len(clean_videos) + [1] * len(poisoned_videos)

    # 70/30 train/test split
    train_videos, test_videos, train_labels, test_labels = train_test_split(
        all_videos, all_labels,
        test_size=0.3,
        random_state=42,
        stratify=all_labels  # Ensure balanced split
    )

    print(f"Training set: {len(train_videos)} videos")
    print(f"Test set: {len(test_videos)} videos")
    print(f"  - Train clean: {sum(1 for l in train_labels if l == 0)}")
    print(f"  - Train poisoned: {sum(1 for l in train_labels if l == 1)}")
    print(f"  - Test clean: {sum(1 for l in test_labels if l == 0)}")
    print(f"  - Test poisoned: {sum(1 for l in test_labels if l == 1)}")
    print()

    # Create datasets
    train_dataset = VideoDataset(train_videos, train_labels, num_frames=16)
    test_dataset = VideoDataset(test_videos, test_labels, num_frames=16)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Create model with regularization
    device = 'cpu'
    model = Simple3DCNN(num_classes=2, num_frames=16)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # L2 regularization

    # Train
    print("Training with dropout + L2 regularization...")
    print()

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(50):  # More epochs with early stopping
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in test_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1:2d}: Train Acc={train_acc:5.1f}% | Val Acc={val_acc:5.1f}% | Val Loss={val_loss/len(test_loader):.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print()
    print("=" * 60)
    print("DETECTION TEST ON HELD-OUT VIDEOS")
    print("=" * 60)
    print()

    # Get only CLEAN test videos for detection
    test_clean_videos = [v for v, l in zip(test_videos, test_labels) if l == 0]

    print(f"Testing detection on {len(test_clean_videos)} held-out clean videos...")
    print("(Videos the model has NEVER seen during training)")
    print()

    detector = VideoRadioactiveDetector(signature_path, device=device)
    is_poisoned, score = detector.detect(
        model,
        [str(v) for v in test_clean_videos[:min(5, len(test_clean_videos))]],
        threshold=0.05,
        method='temporal'
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Detection score on held-out videos: {score:.6f}")
    print(f"Detected as poisoned: {is_poisoned}")
    print()

    if score > 0.15:
        print("✅✅ SUCCESS ✅✅")
        print("   Model learned the signature and generalizes!")
        print("   This is REAL detection, not memorization.")
    elif score > 0.05:
        print("⚠️  WEAK SIGNAL")
        print(f"   Detection works but score is low ({score:.3f})")
        print("   May need more data or stronger poisoning")
    else:
        print("❌ FAILED TO GENERALIZE")
        print("   Model didn't learn the signature pattern")
        print()
        print("   Possible causes:")
        print("   1. Not enough training data (need 100+ videos)")
        print("   2. Optical flow perturbation too weak")
        print("   3. Signature doesn't survive model learning")

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
