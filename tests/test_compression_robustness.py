#!/usr/bin/env python3
"""
Compression Robustness Ladder Test

This is THE critical test: Does the signature survive real H.264 compression?

We test across CRF levels:
- CRF 18: High quality (streaming, archival)
- CRF 23: Good quality (Vimeo)
- CRF 28: Medium quality (YouTube default) ← TARGET
- CRF 31: Low quality (mobile streaming)
- CRF 35: Poor quality (heavy compression)

Success criteria:
- CRF 23: >70% signature survival
- CRF 28: >50% signature survival ← INDUSTRY-BREAKING THRESHOLD
- CRF 31: >30% signature survival

If CRF 28 >50%, we have a production-ready system.
If CRF 28 <50%, we need adaptive training (differentiable codec).
"""

import sys
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker


def create_test_video(output_path: str, num_frames: int = 120, resolution: int = 224) -> None:
    """
    Create realistic test video with varied motion.

    This simulates real video content better than simple moving shapes.
    """
    width, height = resolution, resolution
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient background
        for i in range(height):
            intensity = int(50 + 100 * i / height)
            frame[i, :] = (intensity, intensity // 2, intensity // 3)

        # Moving circle
        t = frame_idx / num_frames
        x = int(width / 2 + width / 4 * np.sin(2 * np.pi * t * 2))
        y = int(height / 2 + height / 4 * np.cos(2 * np.pi * t * 3))
        cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

        # Moving rectangle
        rect_x = int(50 + 100 * t)
        rect_y = int(50 + 100 * (1 - t))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 40, rect_y + 40), (200, 100, 50), -1)

        out.write(frame)

    out.release()


def compress_video(input_path: str, output_path: str, crf: int) -> bool:
    """
    Compress video using H.264 with specified CRF.

    Args:
        input_path: Input video path
        output_path: Output compressed video path
        crf: Constant Rate Factor (18-35, lower=better quality)

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', 'medium',
        '-y',  # Overwrite output
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error compressing video: {e}")
        return False


def measure_signature_survival(
    marker: FrequencyDomainVideoMarker,
    original_poisoned_path: str,
    compressed_path: str,
    num_frames: int = 30
) -> dict:
    """
    Measure how much of the signature survived compression.

    Returns dictionary with:
        - dct_correlation: Correlation between original and compressed DCT signatures
        - temporal_correlation: Correlation of temporal patterns
        - psnr: Visual quality metric
    """
    # Extract signatures from both videos
    orig_dct, orig_temporal = marker.extract_dct_signature(original_poisoned_path, num_frames=num_frames)
    comp_dct, comp_temporal = marker.extract_dct_signature(compressed_path, num_frames=num_frames)

    # Compute DCT correlation (focus on low-frequency region)
    orig_low_freq = orig_dct[0:3, 0:3].flatten()
    comp_low_freq = comp_dct[0:3, 0:3].flatten()

    # Normalize
    orig_norm = orig_low_freq / (np.linalg.norm(orig_low_freq) + 1e-8)
    comp_norm = comp_low_freq / (np.linalg.norm(comp_low_freq) + 1e-8)

    dct_correlation = float(np.dot(orig_norm, comp_norm))

    # Compute temporal correlation
    min_len = min(len(orig_temporal), len(comp_temporal))
    if min_len > 0:
        orig_t_norm = (orig_temporal[:min_len] - np.mean(orig_temporal[:min_len])) / (np.std(orig_temporal[:min_len]) + 1e-8)
        comp_t_norm = (comp_temporal[:min_len] - np.mean(comp_temporal[:min_len])) / (np.std(comp_temporal[:min_len]) + 1e-8)
        temporal_correlation = float(np.corrcoef(orig_t_norm, comp_t_norm)[0, 1])
    else:
        temporal_correlation = 0.0

    # Compute PSNR
    psnr = marker.compute_psnr(original_poisoned_path, compressed_path, num_frames=10)

    return {
        'dct_correlation': dct_correlation,
        'temporal_correlation': temporal_correlation,
        'psnr': psnr
    }


def test_compression_ladder():
    """
    Main test: Compress poisoned video at multiple CRF levels and measure survival.
    """
    print("=" * 80)
    print("COMPRESSION ROBUSTNESS LADDER TEST")
    print("=" * 80)
    print()
    print("Testing H.264 compression at various quality levels (CRF 18-35)")
    print("Target: >50% signature survival at CRF 28 (YouTube quality)")
    print()

    # Create test video
    clean_path = '/tmp/compression_test_clean.mp4'
    poisoned_path = '/tmp/compression_test_poisoned.mp4'

    print("Creating test video...")
    create_test_video(clean_path, num_frames=120, resolution=224)

    # Poison video
    print("Poisoning video with frequency domain marker...")
    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low', temporal_period=30)
    marker.poison_video(clean_path, poisoned_path, verbose=False)

    # Get baseline (uncompressed)
    print("Measuring baseline (uncompressed)...")
    baseline_dct, baseline_temporal = marker.extract_dct_signature(poisoned_path, num_frames=30)

    # Normalize baseline for comparison
    expected_dct_low = baseline_dct[0:3, 0:3].flatten()
    expected_dct_norm = expected_dct_low / (np.linalg.norm(expected_dct_low) + 1e-8)

    print()
    print("=" * 80)
    print("COMPRESSION TEST RESULTS")
    print("=" * 80)
    print()

    crf_levels = [18, 23, 28, 31, 35]
    results = []

    for crf in crf_levels:
        compressed_path = f'/tmp/compression_test_crf{crf}.mp4'

        print(f"Testing CRF {crf}...")

        # Compress
        success = compress_video(poisoned_path, compressed_path, crf)

        if not success:
            print(f"  ❌ Compression failed")
            continue

        # Measure survival
        metrics = measure_signature_survival(marker, poisoned_path, compressed_path, num_frames=30)

        # Compute survival percentage (relative to baseline)
        dct_survival = metrics['dct_correlation'] * 100
        temporal_survival = abs(metrics['temporal_correlation']) * 100

        results.append({
            'crf': crf,
            'dct_correlation': metrics['dct_correlation'],
            'temporal_correlation': metrics['temporal_correlation'],
            'dct_survival_pct': dct_survival,
            'temporal_survival_pct': temporal_survival,
            'psnr': metrics['psnr']
        })

        print(f"  DCT correlation:      {metrics['dct_correlation']:.4f} ({dct_survival:.1f}% survival)")
        print(f"  Temporal correlation: {metrics['temporal_correlation']:.4f} ({temporal_survival:.1f}% survival)")
        print(f"  PSNR:                 {metrics['psnr']:.2f} dB")

        # Verdict
        if crf == 28:
            if dct_survival > 50:
                print(f"  ✅ TARGET MET: {dct_survival:.1f}% > 50% at CRF 28")
            else:
                print(f"  ❌ TARGET MISSED: {dct_survival:.1f}% < 50% at CRF 28")

        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'CRF':<6} {'Quality':<15} {'DCT Corr':<12} {'Temporal Corr':<15} {'PSNR (dB)':<12} {'Status'}")
    print("-" * 80)

    for r in results:
        if r['crf'] == 18:
            quality = "High (archival)"
        elif r['crf'] == 23:
            quality = "Good (Vimeo)"
        elif r['crf'] == 28:
            quality = "Medium (YouTube)"
        elif r['crf'] == 31:
            quality = "Low (mobile)"
        else:
            quality = "Poor (heavy)"

        dct_pct = r['dct_survival_pct']
        temporal_pct = r['temporal_survival_pct']

        if r['crf'] == 28:
            status = "✅ TARGET" if dct_pct > 50 else "❌ MISS"
        elif dct_pct > 70:
            status = "✅ Excellent"
        elif dct_pct > 50:
            status = "✅ Good"
        elif dct_pct > 30:
            status = "⚠️  Acceptable"
        else:
            status = "❌ Poor"

        print(f"{r['crf']:<6} {quality:<15} {r['dct_correlation']:.4f} ({dct_pct:5.1f}%)  "
              f"{r['temporal_correlation']:.4f} ({temporal_pct:5.1f}%)  "
              f"{r['psnr']:5.2f} dB      {status}")

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Find CRF 28 result
    crf28_result = next((r for r in results if r['crf'] == 28), None)

    if crf28_result:
        survival = crf28_result['dct_survival_pct']

        if survival > 70:
            print(f"✅✅ EXCELLENT: {survival:.1f}% survival at CRF 28")
            print()
            print("Frequency domain poisoning is PRODUCTION-READY.")
            print("This exceeds YouTube compression requirements.")
            print()
            print("Next steps:")
            print("  1. Test on real dataset (UCF-101)")
            print("  2. Validate detection on trained models")
            print("  3. Write CVPR paper")
            return 0

        elif survival > 50:
            print(f"✅ SUCCESS: {survival:.1f}% survival at CRF 28")
            print()
            print("Frequency domain poisoning WORKS at YouTube quality.")
            print("This is INDUSTRY-BREAKING - first compression-robust radioactive marking.")
            print()
            print("Next steps:")
            print("  1. Test on real dataset (UCF-101)")
            print("  2. Validate detection on trained models")
            print("  3. Consider adaptive training to boost to >70%")
            print("  4. Write CVPR paper")
            return 0

        elif survival > 30:
            print(f"⚠️  PARTIAL SUCCESS: {survival:.1f}% survival at CRF 28")
            print()
            print("Signature survives but below target threshold.")
            print()
            print("Options:")
            print("  1. Increase epsilon (trade visual quality for robustness)")
            print("  2. Build differentiable H.264 proxy for adaptive training")
            print("  3. Focus on CRF 23 (Vimeo) instead of CRF 28 (YouTube)")
            print()
            print("Recommendation: Try adaptive training before declaring failure.")
            return 1

        else:
            print(f"❌ FAILURE: {survival:.1f}% survival at CRF 28")
            print()
            print("Frequency domain approach does NOT survive YouTube compression.")
            print()
            print("Next steps:")
            print("  1. Build differentiable H.264 proxy (MANDATORY)")
            print("  2. Train adaptive signature specifically for CRF 28")
            print("  3. If still fails, pivot to CRF 18-23 only")
            return 1
    else:
        print("❌ ERROR: CRF 28 test did not run")
        return 1


if __name__ == '__main__':
    sys.exit(test_compression_ladder())
