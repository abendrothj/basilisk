#!/usr/bin/env python3
"""
Test if detection generalizes or just memorizes.

CRITICAL TEST: Does the model learn the SIGNATURE or just MEMORIZE videos?

If it's just memorization:
- Training on video A (poisoned) → learns "video A = class 1"
- Testing on video B (clean) → no signature should be detected
- But we're testing on the SAME videos used for training!

Real test:
1. Poison 10 videos with same signature
2. Train on 8 of them
3. Test detection on the OTHER 2 (held-out)
4. If detection still works → learning signature
5. If detection fails → just memorizing
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


def test_train_test_split_detection(data_dir, signature_path):
    """
    The REAL test: Train on some videos, detect on DIFFERENT videos.

    If detection works on held-out videos → learned signature
    If detection fails → just memorized training videos
    """
    print("=" * 60)
    print("GENERALIZATION TEST: Train/Test Split")
    print("=" * 60)
    print()

    clean_videos = list((Path(data_dir) / 'clean').glob('*.mp4'))
    poisoned_videos = list((Path(data_dir) / 'poisoned').glob('*.mp4'))

    if len(clean_videos) < 3 or len(poisoned_videos) < 3:
        print("ERROR: Need at least 3 clean and 3 poisoned videos")
        return

    # Split: Use first 2 for training, last 1 for testing
    clean_train = clean_videos[:2]
    clean_test = clean_videos[2:3]

    poisoned_train = poisoned_videos[:2]
    poisoned_test = poisoned_videos[2:3]

    print(f"Training set: {len(clean_train)} clean + {len(poisoned_train)} poisoned")
    print(f"Test set: {len(clean_test)} clean + {len(poisoned_test)} poisoned")
    print()

    # Train on TRAINING SET
    train_videos = clean_train + poisoned_train
    train_labels = [0] * len(clean_train) + [1] * len(poisoned_train)

    dataset = VideoDataset(train_videos, train_labels, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    model = Simple3DCNN(num_classes=2, num_frames=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training on subset...")
    train_model(model, dataloader, criterion, optimizer, device='cpu', epochs=30)

    # Detect on TEST SET (videos the model has NEVER seen)
    test_videos = clean_test
    detector = VideoRadioactiveDetector(signature_path, device='cpu')

    print()
    print("Testing detection on HELD-OUT videos...")
    is_poisoned, score = detector.detect(model, test_videos, threshold=0.05, method='temporal')

    print()
    print("=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Detection score on held-out videos: {score:.6f}")
    print(f"Detected as poisoned: {is_poisoned}")
    print()

    if score > 0.15:
        print("✅ GENERALIZATION CONFIRMED")
        print("   Model learned the SIGNATURE, not just memorized videos")
    elif score > 0.05:
        print("⚠️  WEAK GENERALIZATION")
        print("   Some signal but weaker than training set")
    else:
        print("❌ NO GENERALIZATION")
        print("   Model just memorized training videos")
        print("   Detection is NOT valid for unseen data")

    return score


def test_cross_signature_detection(data_dir):
    """
    CONTROL: Train with signature A, detect with signature B.

    If this scores high → detector is broken (false positives)
    If this scores low → detector is specific to signatures
    """
    print()
    print("=" * 60)
    print("CONTROL TEST: Wrong Signature Detection")
    print("=" * 60)
    print()

    # Create dataset with signature A
    marker_a = VideoRadioactiveMarker(epsilon=0.03, device='cpu')
    marker_a.generate_signature(seed=11111)
    sig_a_path = "/tmp/signature_a.json"
    marker_a.save_signature(sig_a_path)

    # Use existing dataset (poisoned with different signature)
    clean_videos = list((Path(data_dir) / 'clean').glob('*.mp4'))
    poisoned_videos = list((Path(data_dir) / 'poisoned').glob('*.mp4'))

    # Train model on original poisoned data
    train_videos = clean_videos + poisoned_videos
    train_labels = [0] * len(clean_videos) + [1] * len(poisoned_videos)

    dataset = VideoDataset(train_videos, train_labels, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    model = Simple3DCNN(num_classes=2, num_frames=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training on videos poisoned with original signature...")
    train_model(model, dataloader, criterion, optimizer, device='cpu', epochs=20)

    # Try to detect with WRONG signature
    detector_wrong = VideoRadioactiveDetector(sig_a_path, device='cpu')
    test_videos = clean_videos[:min(3, len(clean_videos))]

    print()
    print("Testing detection with WRONG signature...")
    is_poisoned, score = detector_wrong.detect(model, test_videos, threshold=0.05, method='temporal')

    print()
    print("=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Detection score with wrong signature: {score:.6f}")
    print()

    if abs(score) < 0.1:
        print("✅ SPECIFICITY CONFIRMED")
        print("   Detector doesn't trigger on wrong signatures")
    else:
        print("❌ FALSE POSITIVE")
        print(f"   Detector triggered on wrong signature (score={score:.3f})")
        print("   This means detection is not signature-specific")

    os.remove(sig_a_path)
    return score


if __name__ == '__main__':
    data_dir = 'verification_video_data'
    signature_path = f'{data_dir}/signature.json'

    if not Path(data_dir).exists():
        print("ERROR: Run create_video_dataset.py first")
        sys.exit(1)

    print("=" * 60)
    print("CRITICAL VALIDATION TESTS")
    print("=" * 60)
    print()
    print("These tests determine if detection is REAL or MEMORIZATION")
    print()

    # Test 1: Generalization
    gen_score = test_train_test_split_detection(data_dir, signature_path)

    # Test 2: Specificity
    spec_score = test_cross_signature_detection(data_dir)

    print()
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"Generalization score: {gen_score:.6f}")
    print(f"Specificity score: {abs(spec_score):.6f}")
    print()

    if gen_score > 0.15 and abs(spec_score) < 0.1:
        print("✅✅ DETECTION IS VALID ✅✅")
        print("   - Generalizes to unseen videos")
        print("   - Specific to correct signature")
        print()
        print("This is REAL. The approach works.")
    elif gen_score > 0.05:
        print("⚠️  WEAK VALIDATION")
        print("   Detection works but signal is weak")
        print("   Need more data for confidence")
    else:
        print("❌ DETECTION IS NOT VALID")
        print("   Model is just memorizing, not learning signature")
        print("   Back to the drawing board")
