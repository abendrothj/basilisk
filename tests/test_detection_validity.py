#!/usr/bin/env python3
"""
Rigorous test to verify detection isn't spurious.

This tests:
1. CONTROL: Untrained model should score ~0.0 (no signature)
2. EXPERIMENTAL: Trained model should score > threshold
3. NEGATIVE CONTROL: Model trained on CLEAN data should score ~0.0
4. Dataset size matters: More data = more stable signal

Goal: Prove the 0.292 score isn't just random luck.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'verification'))

from video_poison import VideoRadioactiveMarker, VideoRadioactiveDetector
from verify_video_poison import Simple3DCNN, VideoDataset, train_model


def test_untrained_model(signature_path, test_videos):
    """
    CONTROL: Untrained model should have ~0 correlation.

    If this scores high, our detection is broken.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Untrained Model (Control)")
    print("=" * 60)
    print("Hypothesis: Random initialization → correlation ≈ 0")
    print()

    model = Simple3DCNN(num_classes=2, num_frames=16)
    model.eval()

    detector = VideoRadioactiveDetector(signature_path, device='cpu')

    # Test 5 random initializations
    scores = []
    for i in range(5):
        # Re-initialize model
        model = Simple3DCNN(num_classes=2, num_frames=16)
        model.eval()

        _, score = detector.detect(model, test_videos, threshold=0.05, method='temporal')
        scores.append(score)
        print(f"  Init {i+1}: score = {score:.6f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print()
    print(f"Mean score: {mean_score:.6f} ± {std_score:.6f}")

    if abs(mean_score) < 0.05:
        print("✅ PASS: Untrained models show no signature")
    else:
        print(f"⚠️  WARNING: Untrained models showing bias (mean={mean_score:.3f})")
        print("   This suggests detection may be picking up random correlations")

    return mean_score, std_score


def test_clean_trained_model(signature_path, clean_videos, test_videos):
    """
    NEGATIVE CONTROL: Model trained on CLEAN data should score ~0.

    If this scores high, the signature isn't from poisoning.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Clean-Trained Model (Negative Control)")
    print("=" * 60)
    print("Hypothesis: Training on clean data → correlation ≈ 0")
    print()

    # Create dataset with ONLY clean videos
    labels = [0] * len(clean_videos)  # All class 0
    dataset = VideoDataset(clean_videos, labels, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Train model
    model = Simple3DCNN(num_classes=2, num_frames=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training on CLEAN data only...")
    train_model(model, dataloader, criterion, optimizer, device='cpu', epochs=20)

    # Detect
    detector = VideoRadioactiveDetector(signature_path, device='cpu')
    is_poisoned, score = detector.detect(model, test_videos, threshold=0.05, method='temporal')

    print()
    print(f"Detection score: {score:.6f}")

    if score < 0.1:
        print("✅ PASS: Clean-trained model shows no signature")
    else:
        print(f"❌ FAIL: Clean-trained model scored {score:.3f} > 0.1")
        print("   This means detection is picking up spurious correlations")

    return score


def test_poisoned_trained_model(signature_path, poisoned_videos, clean_videos, test_videos, epochs=20):
    """
    EXPERIMENTAL: Model trained on poisoned data should score HIGH.

    This is the real test. We need convergence before measuring.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Poisoned-Trained Model (Experimental)")
    print("=" * 60)
    print(f"Hypothesis: Training on poisoned data for {epochs} epochs → correlation > 0.15")
    print()

    # Create mixed dataset
    video_paths = clean_videos + poisoned_videos
    labels = [0] * len(clean_videos) + [1] * len(poisoned_videos)
    dataset = VideoDataset(video_paths, labels, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Train model
    model = Simple3DCNN(num_classes=2, num_frames=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training on mixed data (clean + poisoned)...")
    train_model(model, dataloader, criterion, optimizer, device='cpu', epochs=epochs)

    # Detect
    detector = VideoRadioactiveDetector(signature_path, device='cpu')
    is_poisoned, score = detector.detect(model, test_videos, threshold=0.05, method='temporal')

    print()
    print(f"Detection score: {score:.6f}")
    print(f"Detected as poisoned: {is_poisoned}")

    if score > 0.15:
        print("✅ PASS: Poisoned-trained model shows strong signature")
    elif score > 0.05:
        print("⚠️  WEAK: Signature detected but below expected threshold")
    else:
        print("❌ FAIL: No signature detected in poisoned-trained model")

    return score


def test_effect_of_training_duration(signature_path, poisoned_videos, clean_videos, test_videos):
    """
    TEST 4: More training → stronger signal?

    If detection score doesn't increase with training, it's not learning the signature.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Effect of Training Duration")
    print("=" * 60)
    print("Hypothesis: More epochs → higher detection score")
    print()

    epoch_counts = [1, 5, 10, 20]
    scores = []

    for epochs in epoch_counts:
        print(f"\n--- Training for {epochs} epochs ---")

        # Create dataset
        video_paths = clean_videos + poisoned_videos
        labels = [0] * len(clean_videos) + [1] * len(poisoned_videos)
        dataset = VideoDataset(video_paths, labels, num_frames=16)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        # Train
        model = Simple3DCNN(num_classes=2, num_frames=16)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, dataloader, criterion, optimizer, device='cpu', epochs=epochs)

        # Detect
        detector = VideoRadioactiveDetector(signature_path, device='cpu')
        _, score = detector.detect(model, test_videos, threshold=0.05, method='temporal')
        scores.append(score)

        print(f"  Detection score: {score:.6f}")

    print("\n" + "-" * 60)
    print("Epoch vs Score:")
    for epochs, score in zip(epoch_counts, scores):
        print(f"  {epochs:2d} epochs → {score:.6f}")

    # Check if score increases with training
    if scores[-1] > scores[0]:
        print()
        print("✅ PASS: Detection score increases with training")
        print(f"   Score went from {scores[0]:.3f} to {scores[-1]:.3f}")
    else:
        print()
        print("❌ FAIL: Detection score doesn't increase with training")
        print("   This suggests the signal is spurious")

    return scores


if __name__ == '__main__':
    print("=" * 60)
    print("RIGOROUS DETECTION VALIDATION")
    print("=" * 60)
    print()
    print("This test suite verifies that detection isn't spurious.")
    print("We need to prove:")
    print("  1. Untrained models score ~0")
    print("  2. Clean-trained models score ~0")
    print("  3. Poisoned-trained models score >0.15")
    print("  4. More training → higher scores")
    print()

    # Setup
    data_dir = Path('verification_video_data')
    signature_path = str(data_dir / 'signature.json')

    clean_videos = list((data_dir / 'clean').glob('*.mp4'))
    poisoned_videos = list((data_dir / 'poisoned').glob('*.mp4'))

    if len(clean_videos) < 3 or len(poisoned_videos) < 3:
        print("ERROR: Need at least 3 clean and 3 poisoned videos")
        print("Run: python verification/create_video_dataset.py")
        sys.exit(1)

    test_videos = clean_videos[:min(3, len(clean_videos))]

    print(f"Dataset:")
    print(f"  Clean videos: {len(clean_videos)}")
    print(f"  Poisoned videos: {len(poisoned_videos)}")
    print(f"  Test videos: {len(test_videos)}")

    # Run tests
    test_untrained_model(signature_path, test_videos)

    # Only run expensive tests if we have enough data
    if len(clean_videos) >= 3 and len(poisoned_videos) >= 3:
        clean_score = test_clean_trained_model(signature_path, clean_videos, test_videos)
        poisoned_score = test_poisoned_trained_model(signature_path, poisoned_videos, clean_videos, test_videos, epochs=20)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Clean-trained model:    {clean_score:.6f}")
        print(f"Poisoned-trained model: {poisoned_score:.6f}")
        print(f"Difference:             {poisoned_score - clean_score:.6f}")
        print()

        if poisoned_score > clean_score + 0.1:
            print("✅ VALIDATION PASSED")
            print("   Poisoned models show significantly higher correlation")
            print("   This is a real signal, not spurious.")
        else:
            print("❌ VALIDATION FAILED")
            print("   No significant difference between clean and poisoned models")
            print("   Detection is likely picking up artifacts.")

    print()
    print("=" * 60)
    print("To run full validation with training duration test:")
    print("  python tests/test_detection_validity.py --full")
    print("=" * 60)
