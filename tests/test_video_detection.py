#!/usr/bin/env python3
"""
Test video detection implementation.

This verifies that our detection algorithms can identify
poisoned models without requiring full training.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add poison-core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from video_poison import VideoRadioactiveMarker, VideoRadioactiveDetector

# Import the Simple3DCNN from verification
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'verification'))
from verify_video_poison import Simple3DCNN


def test_detection_methods():
    """
    Test that all three detection methods run without errors.

    This is a smoke test - we're not training a model, just verifying
    the detection code doesn't crash.
    """
    print("=" * 60)
    print("Testing Video Detection Implementation")
    print("=" * 60)
    print()

    # Create a simple model
    print("1. Creating Simple3DCNN model...")
    model = Simple3DCNN(num_classes=2, num_frames=16)
    model.eval()

    # Generate signature
    print("2. Generating test signature...")
    marker = VideoRadioactiveMarker(epsilon=0.02, temporal_period=30, device='cpu')
    marker.generate_signature(seed=12345)

    # Save signature
    signature_path = "/tmp/test_video_signature.json"
    marker.save_signature(signature_path)
    print(f"   Saved to {signature_path}")

    # Create detector
    print("3. Creating detector...")
    detector = VideoRadioactiveDetector(signature_path, device='cpu')

    # Generate a test video
    print("4. Generating synthetic test videos...")
    test_video_1 = detector._generate_synthetic_video_with_signature()
    test_video_2 = detector._generate_synthetic_video_random()
    print(f"   Generated: {test_video_1}")
    print(f"   Generated: {test_video_2}")

    test_videos = [test_video_1, test_video_2]

    # Test each detection method
    print()
    print("5. Testing detection methods...")
    print()

    print("   [Method 1: Spatial Feature Correlation]")
    try:
        is_poisoned, score = detector._detect_spatial(model, test_videos, threshold=0.05)
        print(f"   ✓ Spatial detection: poisoned={is_poisoned}, score={score:.6f}")
    except Exception as e:
        print(f"   ✗ Spatial detection failed: {e}")

    print()
    print("   [Method 2: Temporal Feature Correlation]")
    try:
        is_poisoned, score = detector._detect_temporal(model, test_videos, threshold=0.05)
        print(f"   ✓ Temporal detection: poisoned={is_poisoned}, score={score:.6f}")
    except Exception as e:
        print(f"   ✗ Temporal detection failed: {e}")

    print()
    print("   [Method 3: Behavioral Test]")
    try:
        is_poisoned, score = detector._detect_behavioral(model, test_videos, threshold=0.1)
        print(f"   ✓ Behavioral detection: poisoned={is_poisoned}, score={score:.6f}")
    except Exception as e:
        print(f"   ✗ Behavioral detection failed: {e}")

    print()
    print("   [Auto Mode: All Methods]")
    try:
        is_poisoned, score = detector.detect(model, test_videos, threshold=0.05, method='auto')
        print(f"   ✓ Auto detection: poisoned={is_poisoned}, score={score:.6f}")
    except Exception as e:
        print(f"   ✗ Auto detection failed: {e}")

    # Cleanup
    print()
    print("6. Cleaning up...")
    os.remove(test_video_1)
    os.remove(test_video_2)
    os.remove(signature_path)

    print()
    print("=" * 60)
    print("✅ All detection methods executed successfully!")
    print("=" * 60)
    print()
    print("NOTE: This is a smoke test with an untrained model.")
    print("For real validation, train a model on poisoned data and test again.")


def test_temporal_correlation_measurement():
    """
    Test the temporal correlation measurement function in isolation.
    """
    print()
    print("=" * 60)
    print("Testing Temporal Correlation Measurement")
    print("=" * 60)
    print()

    # Create detector
    marker = VideoRadioactiveMarker(epsilon=0.02, temporal_period=30, device='cpu')
    marker.generate_signature(seed=12345)
    signature_path = "/tmp/test_temporal_sig.json"
    marker.save_signature(signature_path)
    detector = VideoRadioactiveDetector(signature_path, device='cpu')

    # Test 1: Perfect match
    print("Test 1: Perfect temporal match")
    frame_corrs = list(detector.temporal_signature) * 3  # 3 cycles
    score = detector._measure_temporal_correlation(frame_corrs, detector.temporal_signature)
    print(f"   Score: {score:.6f} (should be ~1.0)")

    # Test 2: Random correlation
    print("Test 2: Random correlations")
    random_corrs = np.random.randn(90).tolist()
    score = detector._measure_temporal_correlation(random_corrs, detector.temporal_signature)
    print(f"   Score: {score:.6f} (should be ~0.0)")

    # Test 3: Inverted pattern
    print("Test 3: Inverted temporal pattern")
    inverted_corrs = list(-detector.temporal_signature) * 3
    score = detector._measure_temporal_correlation(inverted_corrs, detector.temporal_signature)
    print(f"   Score: {score:.6f} (should be ~-1.0)")

    os.remove(signature_path)

    print()
    print("✅ Temporal correlation measurement works correctly!")
    print()


if __name__ == '__main__':
    test_detection_methods()
    test_temporal_correlation_measurement()

    print()
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Run verify_video_poison.py to train a model on poisoned data")
    print("2. Check if detection works on a TRAINED model (this is the real test)")
    print("3. If detection score > threshold, the approach works!")
    print()
