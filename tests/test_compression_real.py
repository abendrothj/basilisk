#!/usr/bin/env python3
"""
Real Compression Test: Measure signature DETECTION, not video similarity.

The previous test had a flaw: it measured correlation between poisoned and compressed videos,
which will always be ~1.0 because they're the same video.

This test measures: Can we DETECT the signature in the compressed video?
- Extract DCT coefficients from compressed video
- Correlate with ORIGINAL signature pattern (not the video)
- This tells us if the signature survives in a detectable form
"""

import sys
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker


def create_test_video(output_path: str, num_frames: int = 120) -> None:
    """Create test video with realistic motion."""
    width, height = 224, 224
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient background
        for i in range(height):
            intensity = int(50 + 100 * i / height)
            frame[i, :] = (intensity, intensity // 2, intensity // 3)

        # Moving objects
        t = frame_idx / num_frames
        x = int(width / 2 + width / 4 * np.sin(2 * np.pi * t * 2))
        y = int(height / 2 + height / 4 * np.cos(2 * np.pi * t * 3))
        cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

        out.write(frame)

    out.release()


def compress_video(input_path: str, output_path: str, crf: int) -> bool:
    """Compress with H.264."""
    cmd = [
        'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', str(crf),
        '-preset', 'medium', '-y', output_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return result.returncode == 0
    except:
        return False


def measure_signature_detection(
    video_path: str,
    marker: FrequencyDomainVideoMarker,
    num_frames: int = 30
) -> dict:
    """
    Measure if we can DETECT the signature in the video.

    This correlates extracted DCT coefficients with the ORIGINAL signature pattern,
    not with another video.
    """
    # Extract DCT coefficients from video
    extracted_dct, extracted_temporal = marker.extract_dct_signature(video_path, num_frames=num_frames)

    # Get the ORIGINAL signature (what we embedded)
    original_signature_dct = marker.signature_dct
    original_signature_temporal = marker.temporal_signature

    # Focus on low-frequency region (where we poisoned)
    extracted_low_freq = extracted_dct[0:3, 0:3]
    original_low_freq = original_signature_dct[0:3, 0:3]

    # Flatten and normalize
    extracted_flat = extracted_low_freq.flatten()
    original_flat = original_low_freq.flatten()

    extracted_norm = extracted_flat / (np.linalg.norm(extracted_flat) + 1e-8)
    original_norm = original_flat / (np.linalg.norm(original_flat) + 1e-8)

    # Compute correlation with ORIGINAL signature
    dct_detection_score = float(np.dot(extracted_norm, original_norm))

    # Temporal detection
    min_len = min(len(extracted_temporal), len(original_signature_temporal))
    if min_len > 0:
        # Normalize temporal patterns
        ext_t_norm = (extracted_temporal[:min_len] - np.mean(extracted_temporal[:min_len])) / (np.std(extracted_temporal[:min_len]) + 1e-8)
        orig_t_norm = (original_signature_temporal[:min_len] - np.mean(original_signature_temporal[:min_len])) / (np.std(original_signature_temporal[:min_len]) + 1e-8)

        temporal_detection_score = float(np.corrcoef(ext_t_norm, orig_t_norm)[0, 1])
    else:
        temporal_detection_score = 0.0

    return {
        'dct_detection': dct_detection_score,
        'temporal_detection': temporal_detection_score
    }


def test_signature_detection_clean_vs_poisoned():
    """
    Control test: Clean video should show NO signature, poisoned should show HIGH signature.
    """
    print("=" * 80)
    print("CONTROL TEST: Clean vs Poisoned Detection")
    print("=" * 80)
    print()

    clean_path = '/tmp/detect_test_clean.mp4'
    poisoned_path = '/tmp/detect_test_poisoned.mp4'

    create_test_video(clean_path, num_frames=120)

    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low', temporal_period=30)
    marker.poison_video(clean_path, poisoned_path, verbose=False)

    # Measure detection on clean video (should be ~0)
    clean_scores = measure_signature_detection(clean_path, marker, num_frames=30)

    # Measure detection on poisoned video (should be high)
    poisoned_scores = measure_signature_detection(poisoned_path, marker, num_frames=30)

    print(f"Clean video detection:")
    print(f"  DCT signature:      {clean_scores['dct_detection']:.4f}")
    print(f"  Temporal signature: {clean_scores['temporal_detection']:.4f}")
    print()

    print(f"Poisoned video detection:")
    print(f"  DCT signature:      {poisoned_scores['dct_detection']:.4f}")
    print(f"  Temporal signature: {poisoned_scores['temporal_detection']:.4f}")
    print()

    # Check if we can distinguish
    dct_diff = poisoned_scores['dct_detection'] - clean_scores['dct_detection']

    if dct_diff > 0.3:
        print(f"✅ PASS: Clear separation ({dct_diff:.3f} difference)")
        return True
    else:
        print(f"❌ FAIL: Cannot distinguish poisoned from clean ({dct_diff:.3f} difference)")
        return False


def test_compression_robustness_real():
    """
    Real compression test: Measure signature DETECTION at various CRF levels.
    """
    print("\n" + "=" * 80)
    print("COMPRESSION ROBUSTNESS: Signature Detection Test")
    print("=" * 80)
    print()

    clean_path = '/tmp/compress_test_clean.mp4'
    poisoned_path = '/tmp/compress_test_poisoned.mp4'

    create_test_video(clean_path, num_frames=120)

    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low', temporal_period=30)
    marker.poison_video(clean_path, poisoned_path, verbose=False)

    # Baseline: uncompressed poisoned video
    baseline_scores = measure_signature_detection(poisoned_path, marker, num_frames=30)
    print(f"Baseline (uncompressed poisoned video):")
    print(f"  DCT detection:      {baseline_scores['dct_detection']:.4f}")
    print(f"  Temporal detection: {baseline_scores['temporal_detection']:.4f}")
    print()

    # Test compression levels
    crf_levels = [18, 23, 28, 31, 35]
    results = []

    print("Testing compression...")
    print()

    for crf in crf_levels:
        compressed_path = f'/tmp/compress_test_crf{crf}.mp4'

        # Compress
        success = compress_video(poisoned_path, compressed_path, crf)
        if not success:
            print(f"CRF {crf}: ❌ Compression failed")
            continue

        # Measure signature detection
        scores = measure_signature_detection(compressed_path, marker, num_frames=30)

        # Compute survival percentage (relative to baseline)
        if baseline_scores['dct_detection'] > 0:
            dct_survival_pct = (scores['dct_detection'] / baseline_scores['dct_detection']) * 100
        else:
            dct_survival_pct = 0

        if abs(baseline_scores['temporal_detection']) > 0:
            temp_survival_pct = (abs(scores['temporal_detection']) / abs(baseline_scores['temporal_detection'])) * 100
        else:
            temp_survival_pct = 0

        results.append({
            'crf': crf,
            'dct_detection': scores['dct_detection'],
            'temporal_detection': scores['temporal_detection'],
            'dct_survival_pct': dct_survival_pct,
            'temp_survival_pct': temp_survival_pct
        })

        print(f"CRF {crf:2d}: DCT={scores['dct_detection']:6.4f} ({dct_survival_pct:5.1f}%), "
              f"Temporal={scores['temporal_detection']:6.4f} ({temp_survival_pct:5.1f}%)")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'CRF':<6} {'DCT Detection':<16} {'Survival %':<12} {'Status'}")
    print("-" * 80)

    for r in results:
        if r['crf'] == 28:
            status = "✅ TARGET" if r['dct_survival_pct'] > 50 else "❌ MISS"
        elif r['dct_survival_pct'] > 70:
            status = "✅ Excellent"
        elif r['dct_survival_pct'] > 50:
            status = "✅ Good"
        elif r['dct_survival_pct'] > 30:
            status = "⚠️  Weak"
        else:
            status = "❌ Failed"

        print(f"{r['crf']:<6} {r['dct_detection']:<16.4f} {r['dct_survival_pct']:<12.1f} {status}")

    print()

    # Verdict
    crf28_result = next((r for r in results if r['crf'] == 28), None)

    if crf28_result and crf28_result['dct_survival_pct'] > 50:
        print(f"✅ SUCCESS: {crf28_result['dct_survival_pct']:.1f}% signature survival at CRF 28")
        print("Signature is DETECTABLE after YouTube-level compression.")
        return True
    elif crf28_result:
        print(f"❌ FAILURE: {crf28_result['dct_survival_pct']:.1f}% signature survival at CRF 28")
        print("Signature degrades below detection threshold.")
        return False
    else:
        print("❌ ERROR: CRF 28 test failed")
        return False


def main():
    print("=" * 80)
    print("REAL COMPRESSION ROBUSTNESS TEST")
    print("=" * 80)
    print()
    print("This measures signature DETECTION, not video similarity.")
    print()

    # Control test
    control_passed = test_signature_detection_clean_vs_poisoned()

    if not control_passed:
        print("\n❌ Control test failed - signature not detectable even without compression")
        return 1

    # Compression test
    compression_passed = test_compression_robustness_real()

    if compression_passed:
        print("\n✅ Frequency domain poisoning survives compression")
        return 0
    else:
        print("\n❌ Signature does not survive compression")
        return 1


if __name__ == '__main__':
    sys.exit(main())
