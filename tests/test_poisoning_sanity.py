#!/usr/bin/env python3
"""
Sanity check: Does poisoning actually create a detectable difference?

This tests if the optical flow perturbation is:
1. Actually applied to the videos
2. Measurably different from clean videos
3. Preserved after encoding/decoding

If this fails, the entire approach is broken at the poisoning step.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from video_poison import VideoRadioactiveMarker


def extract_optical_flow_stats(video_path):
    """
    Extract optical flow statistics from a video.

    Returns:
        - mean_magnitude: Average magnitude of optical flow vectors
        - std_magnitude: Std dev of optical flow magnitudes
        - temporal_pattern: Per-frame mean flow magnitudes
    """
    cap = cv2.VideoCapture(str(video_path))

    ret, prev_frame = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow_magnitudes = []
    temporal_pattern = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute magnitude
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        flow_magnitudes.extend(magnitude.flatten())
        temporal_pattern.append(np.mean(magnitude))

        prev_gray = gray

    cap.release()

    return {
        'mean_magnitude': np.mean(flow_magnitudes),
        'std_magnitude': np.std(flow_magnitudes),
        'temporal_pattern': np.array(temporal_pattern)
    }


def test_poisoning_creates_difference():
    """
    Test 1: Does poisoning create measurably different optical flow?
    """
    print("=" * 60)
    print("TEST 1: Optical Flow Difference")
    print("=" * 60)
    print()

    data_dir = Path('verification_video_data_large')

    if not data_dir.exists():
        print("ERROR: Dataset not found")
        return False

    clean_videos = list((data_dir / 'clean').glob('*.mp4'))[:5]
    poisoned_videos = list((data_dir / 'poisoned').glob('*.mp4'))[:5]

    print(f"Analyzing {len(clean_videos)} clean and {len(poisoned_videos)} poisoned videos...")
    print()

    # Extract flow stats
    clean_stats = []
    for video in clean_videos:
        stats = extract_optical_flow_stats(video)
        if stats:
            clean_stats.append(stats)

    poisoned_stats = []
    for video in poisoned_videos:
        stats = extract_optical_flow_stats(video)
        if stats:
            poisoned_stats.append(stats)

    # Compare mean magnitudes
    clean_means = [s['mean_magnitude'] for s in clean_stats]
    poisoned_means = [s['mean_magnitude'] for s in poisoned_stats]

    clean_avg = np.mean(clean_means)
    poisoned_avg = np.mean(poisoned_means)

    print(f"Average optical flow magnitude:")
    print(f"  Clean:    {clean_avg:.6f}")
    print(f"  Poisoned: {poisoned_avg:.6f}")
    print(f"  Difference: {abs(poisoned_avg - clean_avg):.6f}")
    print()

    # Statistical test
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_ind(clean_means, poisoned_means)

    print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")

    if p_value < 0.05:
        print("✅ SIGNIFICANT DIFFERENCE in optical flow")
        print("   Poisoning IS creating a detectable change")
        return True
    else:
        print("❌ NO SIGNIFICANT DIFFERENCE")
        print("   Poisoning is NOT creating a detectable change")
        print("   The perturbation is too weak or not preserved")
        return False


def test_temporal_signature_preserved():
    """
    Test 2: Is the temporal signature (cyclic pattern) preserved?
    """
    print()
    print("=" * 60)
    print("TEST 2: Temporal Signature Preservation")
    print("=" * 60)
    print()

    data_dir = Path('verification_video_data_large')
    signature_path = data_dir / 'signature.json'

    # Load signature
    marker = VideoRadioactiveMarker(device='cpu')
    marker.load_signature(str(signature_path))

    print(f"Expected temporal pattern (period={marker.temporal_period}):")
    print(f"  Sine wave with {marker.temporal_period} frames per cycle")
    print()

    # Analyze one poisoned video
    poisoned_videos = list((data_dir / 'poisoned').glob('*.mp4'))
    if not poisoned_videos:
        print("ERROR: No poisoned videos found")
        return False

    video_path = poisoned_videos[0]
    stats = extract_optical_flow_stats(video_path)

    if not stats:
        print("ERROR: Failed to extract flow")
        return False

    temporal_pattern = stats['temporal_pattern']

    # Check for cyclic correlation
    period = marker.temporal_period
    expected_signature = marker.temporal_signature

    # Compute cross-correlation
    if len(temporal_pattern) < period:
        print(f"ERROR: Video too short ({len(temporal_pattern)} < {period})")
        return False

    # Tile signature to match video length
    num_tiles = int(np.ceil(len(temporal_pattern) / period))
    signature_tiled = np.tile(expected_signature, num_tiles)[:len(temporal_pattern)]

    # Normalize
    pattern_norm = (temporal_pattern - np.mean(temporal_pattern)) / (np.std(temporal_pattern) + 1e-8)
    sig_norm = (signature_tiled - np.mean(signature_tiled)) / (np.std(signature_tiled) + 1e-8)

    # Correlation
    correlation = np.corrcoef(pattern_norm, sig_norm)[0, 1]

    print(f"Temporal correlation with expected signature: {correlation:.6f}")
    print()

    if abs(correlation) > 0.3:
        print("✅ TEMPORAL SIGNATURE DETECTED")
        print("   The cyclic pattern is preserved in optical flow")
        return True
    else:
        print("❌ NO TEMPORAL SIGNATURE")
        print("   The cyclic pattern is NOT preserved")
        print("   Possible causes:")
        print("   - Epsilon too small (perturbation washed out)")
        print("   - Video encoding destroys the pattern")
        print("   - Flow extraction doesn't capture it")
        return False


def test_visual_inspection():
    """
    Test 3: Visual sanity check - can a human see the difference?
    """
    print()
    print("=" * 60)
    print("TEST 3: Visual Inspection")
    print("=" * 60)
    print()

    data_dir = Path('verification_video_data_large')

    clean_videos = list((data_dir / 'clean').glob('*.mp4'))
    poisoned_videos = list((data_dir / 'poisoned').glob('*.mp4'))

    if not clean_videos or not poisoned_videos:
        print("ERROR: Missing videos")
        return

    clean_path = clean_videos[0]
    poisoned_path = poisoned_videos[0]

    print("Extracting sample frames for visual comparison...")
    print()

    # Extract middle frames
    cap_clean = cv2.VideoCapture(str(clean_path))
    cap_poisoned = cv2.VideoCapture(str(poisoned_path))

    total_frames = int(cap_clean.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2

    cap_clean.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret1, frame_clean = cap_clean.read()

    cap_poisoned.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret2, frame_poisoned = cap_poisoned.read()

    cap_clean.release()
    cap_poisoned.release()

    if ret1 and ret2:
        # Compute PSNR
        mse = np.mean((frame_clean.astype(float) - frame_poisoned.astype(float))**2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        print(f"PSNR between clean and poisoned frames: {psnr:.2f} dB")
        print()

        if psnr > 40:
            print("✅ Frames are visually identical (PSNR > 40 dB)")
            print("   Poisoning is imperceptible - good for stealth")
        elif psnr > 30:
            print("⚠️  Frames are very similar (PSNR 30-40 dB)")
            print("   Poisoning might be subtle but detectable")
        else:
            print("❌ Frames are noticeably different (PSNR < 30 dB)")
            print("   Poisoning is too strong - will be noticed")

        # Save comparison images
        output_dir = Path('/tmp/poison_comparison')
        output_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(output_dir / 'clean_frame.png'), frame_clean)
        cv2.imwrite(str(output_dir / 'poisoned_frame.png'), frame_poisoned)

        # Compute difference image
        diff = cv2.absdiff(frame_clean, frame_poisoned)
        diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(output_dir / 'difference.png'), diff_enhanced)

        print()
        print(f"Comparison images saved to {output_dir}/")
        print("  - clean_frame.png")
        print("  - poisoned_frame.png")
        print("  - difference.png (enhanced)")


def main():
    print("=" * 60)
    print("POISONING SANITY CHECK")
    print("=" * 60)
    print()
    print("This tests if optical flow poisoning actually works")
    print()

    # Install scipy if needed
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        os.system("source venv/bin/activate && pip install -q scipy")
        import scipy

    # Run tests
    test1_passed = test_poisoning_creates_difference()
    test2_passed = test_temporal_signature_preserved()
    test_visual_inspection()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Flow Difference):  {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (Temporal Pattern): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print()

    if test1_passed and test2_passed:
        print("✅✅ POISONING WORKS ✅✅")
        print()
        print("The optical flow perturbation:")
        print("  - Creates measurable differences")
        print("  - Preserves the temporal signature")
        print()
        print("The problem is in DETECTION, not poisoning.")
        print("Need to figure out why models don't learn this signal.")
    elif test1_passed and not test2_passed:
        print("⚠️  PARTIAL SUCCESS")
        print()
        print("Flow is different but temporal pattern is lost.")
        print("Possible fixes:")
        print("  - Increase epsilon (stronger perturbation)")
        print("  - Use different temporal pattern (not sine wave)")
        print("  - Change video encoding to preserve motion better")
    else:
        print("❌ POISONING DOESN'T WORK")
        print()
        print("The perturbation is too weak or not preserved.")
        print("Fundamental approach needs rethinking:")
        print("  - Use much larger epsilon (0.1+)")
        print("  - Poison pixels directly instead of flow")
        print("  - Different perturbation method entirely")


if __name__ == '__main__':
    main()
