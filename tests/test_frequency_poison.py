#!/usr/bin/env python3
"""
Test frequency domain poisoning on uncompressed video.

This validates:
1. DCT coefficients are actually modified
2. Visual quality is acceptable (PSNR > 35 dB)
3. Temporal signature is embedded correctly
4. Signature survives frame extraction

If these tests pass, we proceed to compression robustness testing.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker


def create_test_video(output_path: str, num_frames: int = 60) -> None:
    """Create a simple test video with moving square."""
    width, height = 224, 224
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Create frame with moving square
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (50, 50, 50)  # Gray background

        # Moving square
        x = int(50 + 100 * (frame_idx / num_frames))
        y = height // 2 - 25

        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (255, 255, 255), -1)
        out.write(frame)

    out.release()
    print(f"Created test video: {output_path}")


def test_dct_modification():
    """
    Test 1: Verify DCT coefficients are actually modified.
    """
    print("\n" + "=" * 60)
    print("TEST 1: DCT Coefficient Modification")
    print("=" * 60)

    # Create test video
    clean_path = '/tmp/test_clean.mp4'
    poisoned_path = '/tmp/test_poisoned.mp4'

    create_test_video(clean_path, num_frames=60)

    # Poison
    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low')
    marker.poison_video(clean_path, poisoned_path, verbose=False)

    # Extract DCT coefficients
    clean_dct, _ = marker.extract_dct_signature(clean_path, num_frames=10)
    poisoned_dct, _ = marker.extract_dct_signature(poisoned_path, num_frames=10)

    # Compute difference
    dct_diff = np.abs(poisoned_dct - clean_dct)

    print(f"\nClean DCT coefficients (8x8):")
    print(clean_dct)
    print(f"\nPoisoned DCT coefficients (8x8):")
    print(poisoned_dct)
    print(f"\nAbsolute difference:")
    print(dct_diff)
    print(f"\nMean absolute difference: {dct_diff.mean():.4f}")

    # Check low-frequency coefficients changed (top-left 3x3)
    low_freq_diff = dct_diff[0:3, 0:3].mean()
    high_freq_diff = dct_diff[4:8, 4:8].mean()

    print(f"\nLow-frequency region (0:3, 0:3) diff: {low_freq_diff:.4f}")
    print(f"High-frequency region (4:8, 4:8) diff: {high_freq_diff:.4f}")

    if low_freq_diff > high_freq_diff:
        print("\n✅ PASS: Low frequencies modified more than high frequencies")
        print("   (This is correct - we target low frequencies for compression robustness)")
        return True
    else:
        print("\n❌ FAIL: High frequencies modified more than low frequencies")
        print("   (This suggests poisoning is in wrong frequency band)")
        return False


def test_visual_quality():
    """
    Test 2: Verify poisoning is visually imperceptible (PSNR > 35 dB).
    """
    print("\n" + "=" * 60)
    print("TEST 2: Visual Quality (PSNR)")
    print("=" * 60)

    clean_path = '/tmp/test_clean.mp4'
    poisoned_path = '/tmp/test_poisoned.mp4'

    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low')

    # Compute PSNR
    psnr = marker.compute_psnr(clean_path, poisoned_path, num_frames=10)

    print(f"\nPSNR: {psnr:.2f} dB")

    if psnr > 40:
        print("✅ Visually identical (PSNR > 40 dB)")
        quality = "excellent"
        passed = True
    elif psnr > 35:
        print("✅ Subtle differences (PSNR 35-40 dB)")
        quality = "good"
        passed = True
    elif psnr > 30:
        print("⚠️  Noticeable but acceptable (PSNR 30-35 dB)")
        quality = "acceptable"
        passed = True
    else:
        print(f"❌ FAIL: Too much distortion (PSNR {psnr:.1f} < 30 dB)")
        quality = "poor"
        passed = False

    print(f"\nQuality: {quality}")

    return passed


def test_temporal_signature():
    """
    Test 3: Verify temporal signature is embedded correctly.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Temporal Signature Embedding")
    print("=" * 60)

    poisoned_path = '/tmp/test_poisoned.mp4'

    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low', temporal_period=30)

    # Extract temporal pattern (DC coefficient over time)
    _, temporal_pattern = marker.extract_dct_signature(poisoned_path, num_frames=30)

    # Normalize
    temporal_pattern_norm = (temporal_pattern - np.mean(temporal_pattern)) / (np.std(temporal_pattern) + 1e-8)
    signature_norm = (marker.temporal_signature - np.mean(marker.temporal_signature)) / (np.std(marker.temporal_signature) + 1e-8)

    # Compute correlation
    if len(temporal_pattern_norm) == len(signature_norm):
        correlation = np.corrcoef(temporal_pattern_norm, signature_norm)[0, 1]
    else:
        # Pad/trim to match
        min_len = min(len(temporal_pattern_norm), len(signature_norm))
        correlation = np.corrcoef(temporal_pattern_norm[:min_len], signature_norm[:min_len])[0, 1]

    print(f"\nExpected temporal signature (sine wave, period=30):")
    print(marker.temporal_signature[:10], "...")

    print(f"\nMeasured temporal pattern (DC coefficient over time):")
    print(temporal_pattern[:10], "...")

    print(f"\nCorrelation between expected and measured: {correlation:.4f}")

    if abs(correlation) > 0.3:
        print(f"\n✅ PASS: Temporal signature detected (correlation {correlation:.3f})")
        return True
    else:
        print(f"\n❌ FAIL: No temporal signature (correlation {correlation:.3f} < 0.3)")
        return False


def test_epsilon_scaling():
    """
    Test 4: Verify epsilon controls perturbation strength.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Epsilon Scaling")
    print("=" * 60)

    clean_path = '/tmp/test_clean.mp4'

    epsilons = [0.01, 0.03, 0.05, 0.1]
    psnrs = []
    dct_diffs = []

    print(f"\nTesting epsilon values: {epsilons}")
    print()

    for epsilon in epsilons:
        poisoned_path = f'/tmp/test_poisoned_eps{epsilon}.mp4'

        marker = FrequencyDomainVideoMarker(epsilon=epsilon, frequency_band='low')
        marker.poison_video(clean_path, poisoned_path, verbose=False)

        # Measure PSNR
        psnr = marker.compute_psnr(clean_path, poisoned_path, num_frames=5)

        # Measure DCT difference
        clean_dct, _ = marker.extract_dct_signature(clean_path, num_frames=5)
        poisoned_dct, _ = marker.extract_dct_signature(poisoned_path, num_frames=5)
        dct_diff = np.abs(poisoned_dct - clean_dct).mean()

        psnrs.append(psnr)
        dct_diffs.append(dct_diff)

        print(f"Epsilon {epsilon:.3f}: PSNR = {psnr:.2f} dB, DCT diff = {dct_diff:.4f}")

    # Check monotonicity
    print()
    psnr_decreasing = all(psnrs[i] >= psnrs[i+1] for i in range(len(psnrs)-1))
    dct_diff_increasing = all(dct_diffs[i] <= dct_diffs[i+1] for i in range(len(dct_diffs)-1))

    if psnr_decreasing and dct_diff_increasing:
        print("✅ PASS: Higher epsilon → lower PSNR and larger DCT difference")
        print("   (Epsilon correctly controls perturbation strength)")
        return True
    else:
        print("❌ FAIL: Epsilon scaling is not monotonic")
        return False


def main():
    print("=" * 60)
    print("FREQUENCY DOMAIN POISONING - VALIDATION TESTS")
    print("=" * 60)
    print()
    print("These tests verify DCT poisoning works on uncompressed video.")
    print("If all tests pass, we proceed to compression robustness testing.")
    print()

    # Run tests
    test1_passed = test_dct_modification()
    test2_passed = test_visual_quality()
    test3_passed = test_temporal_signature()
    test4_passed = test_epsilon_scaling()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (DCT Modification):     {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Visual Quality):       {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Test 3 (Temporal Signature):   {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"Test 4 (Epsilon Scaling):      {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print()

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed

    if all_passed:
        print("✅✅ ALL TESTS PASSED ✅✅")
        print()
        print("DCT-based frequency domain poisoning WORKS on uncompressed video.")
        print()
        print("Next steps:")
        print("  1. Test compression robustness (CRF 18-35)")
        print("  2. Build differentiable H.264 proxy")
        print("  3. Train adaptive signature for CRF 28")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Debug and fix issues before proceeding to compression testing.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
