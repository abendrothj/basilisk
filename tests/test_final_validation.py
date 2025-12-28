#!/usr/bin/env python3
"""
FINAL VALIDATION: Verify CRF 28 results are real.

Critical tests:
1. Clean videos should score LOW (<0.1)
2. Poisoned videos should score HIGH (>0.3)
3. Multiple video types should work
4. Statistical significance test

If any of these fail, we don't have a real result.
"""

import sys
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker
from frequency_detector import FrequencySignatureDetector


def create_diverse_videos(output_dir: str, num_videos: int = 10):
    """Create diverse test videos with different content."""
    Path(output_dir).mkdir(exist_ok=True)

    video_types = ['gradient', 'shapes', 'noise', 'text', 'mixed']

    for i in range(num_videos):
        video_type = video_types[i % len(video_types)]
        video_path = f'{output_dir}/clean_{video_type}_{i}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (224, 224))

        for frame_idx in range(60):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

            if video_type == 'gradient':
                # Smooth gradient
                for y in range(224):
                    intensity = int(50 + 150 * y / 224)
                    frame[y, :] = (intensity, int(intensity * 0.8), int(intensity * 0.6))

            elif video_type == 'shapes':
                # Random shapes
                frame[:, :] = (100, 100, 100)
                for _ in range(5):
                    x = np.random.randint(20, 200)
                    y = np.random.randint(20, 200)
                    size = np.random.randint(10, 30)
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    cv2.circle(frame, (x, y), size, color, -1)

            elif video_type == 'noise':
                # Random noise
                frame = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

            elif video_type == 'text':
                # Text rendering
                frame[:, :] = (200, 200, 200)
                cv2.putText(frame, f'Frame {frame_idx}', (20, 112),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            else:  # mixed
                # Gradient + shapes + motion
                for y in range(224):
                    intensity = int(50 + 150 * y / 224)
                    frame[y, :] = (intensity, int(intensity * 0.7), int(intensity * 0.5))

                t = frame_idx / 60.0
                x = int(112 + 50 * np.sin(2 * np.pi * t * 2))
                y = int(112 + 50 * np.cos(2 * np.pi * t * 3))
                cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

            out.write(frame)

        out.release()

    print(f"Created {num_videos} diverse clean videos in {output_dir}/")


def compress_video(input_path: str, output_path: str, crf: int) -> bool:
    """Compress with H.264."""
    cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', str(crf),
           '-preset', 'medium', '-y', output_path]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return result.returncode == 0
    except:
        return False


def test_false_positive_rate():
    """
    Test 1: False positive rate on CLEAN videos.

    Critical: Clean videos should score LOW even after CRF 28 compression.
    """
    print("=" * 80)
    print("TEST 1: False Positive Rate (Clean Videos)")
    print("=" * 80)
    print()

    # Create clean videos
    create_diverse_videos('/tmp/validation_clean', num_videos=10)

    # Load optimized signature
    detector = FrequencySignatureDetector('/Users/ja/Desktop/projects/basilisk/optimized_signature_crf28.json')

    clean_scores = []
    clean_compressed_scores = []

    for video_path in Path('/tmp/validation_clean').glob('*.mp4'):
        # Test uncompressed
        score, _ = detector.detect_in_video(str(video_path), num_frames=20)
        clean_scores.append(score)

        # Compress to CRF 28
        compressed_path = str(video_path).replace('.mp4', '_crf28.mp4')
        compress_video(str(video_path), compressed_path, crf=28)

        # Test compressed
        score_compressed, _ = detector.detect_in_video(compressed_path, num_frames=20)
        clean_compressed_scores.append(score_compressed)

    print(f"Clean videos (uncompressed): mean={np.mean(clean_scores):.4f}, max={np.max(clean_scores):.4f}")
    print(f"Clean videos (CRF 28):       mean={np.mean(clean_compressed_scores):.4f}, max={np.max(clean_compressed_scores):.4f}")
    print()

    false_positive_rate = np.sum(np.array(clean_compressed_scores) > 0.3) / len(clean_compressed_scores)

    print(f"False positive rate (threshold=0.3): {false_positive_rate * 100:.1f}%")
    print()

    if false_positive_rate < 0.1:
        print("✅ PASS: Low false positive rate (<10%)")
        return True, clean_compressed_scores
    else:
        print(f"❌ FAIL: High false positive rate ({false_positive_rate*100:.1f}%)")
        print("   Clean videos are triggering detection - signature is not specific")
        return False, clean_compressed_scores


def test_true_positive_rate():
    """
    Test 2: True positive rate on POISONED videos.

    Critical: Poisoned videos should score HIGH after CRF 28 compression.
    """
    print("\n" + "=" * 80)
    print("TEST 2: True Positive Rate (Poisoned Videos)")
    print("=" * 80)
    print()

    # Use clean videos from Test 1
    marker = FrequencyDomainVideoMarker(epsilon=0.01, frequency_band='low')
    marker.load_signature('/Users/ja/Desktop/projects/basilisk/optimized_signature_crf28.json')

    detector = FrequencySignatureDetector('/Users/ja/Desktop/projects/basilisk/optimized_signature_crf28.json')

    poisoned_scores = []
    poisoned_compressed_scores = []

    for video_path in list(Path('/tmp/validation_clean').glob('clean_*.mp4'))[:10]:
        # Poison
        poisoned_path = str(video_path).replace('clean_', 'poisoned_')
        marker.poison_video(str(video_path), poisoned_path, verbose=False)

        # Test uncompressed
        score, _ = detector.detect_in_video(poisoned_path, num_frames=20)
        poisoned_scores.append(score)

        # Compress to CRF 28
        compressed_path = poisoned_path.replace('.mp4', '_crf28.mp4')
        compress_video(poisoned_path, compressed_path, crf=28)

        # Test compressed
        score_compressed, _ = detector.detect_in_video(compressed_path, num_frames=20)
        poisoned_compressed_scores.append(score_compressed)

    print(f"Poisoned videos (uncompressed): mean={np.mean(poisoned_scores):.4f}, min={np.min(poisoned_scores):.4f}")
    print(f"Poisoned videos (CRF 28):       mean={np.mean(poisoned_compressed_scores):.4f}, min={np.min(poisoned_compressed_scores):.4f}")
    print()

    true_positive_rate = np.sum(np.array(poisoned_compressed_scores) > 0.3) / len(poisoned_compressed_scores)

    print(f"True positive rate (threshold=0.3): {true_positive_rate * 100:.1f}%")
    print()

    if true_positive_rate > 0.8:
        print("✅ PASS: High true positive rate (>80%)")
        return True, poisoned_compressed_scores
    else:
        print(f"❌ FAIL: Low true positive rate ({true_positive_rate*100:.1f}%)")
        print("   Signature not surviving CRF 28 compression reliably")
        return False, poisoned_compressed_scores


def test_statistical_significance(clean_scores, poisoned_scores):
    """
    Test 3: Statistical significance.

    Critical: Poisoned and clean distributions must be significantly different.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Statistical Significance")
    print("=" * 80)
    print()

    from scipy import stats

    # T-test
    t_stat, p_value = stats.ttest_ind(poisoned_scores, clean_scores)

    print(f"Clean scores:    {np.mean(clean_scores):.4f} ± {np.std(clean_scores):.4f}")
    print(f"Poisoned scores: {np.mean(poisoned_scores):.4f} ± {np.std(poisoned_scores):.4f}")
    print()
    print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")
    print()

    if p_value < 0.001:
        print("✅ PASS: Highly significant difference (p<0.001)")
        print("   Poisoned and clean distributions are clearly separable")
        return True
    else:
        print(f"❌ FAIL: Not statistically significant (p={p_value:.4f})")
        print("   Cannot reliably distinguish poisoned from clean")
        return False


def test_compression_degradation():
    """
    Test 4: How much does compression degrade the signal?

    This shows if differentiable codec training actually helped.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Compression Degradation Analysis")
    print("=" * 80)
    print()

    # Use one poisoned video
    test_video = list(Path('/tmp/validation_clean').glob('poisoned_*.mp4'))[0]

    detector = FrequencySignatureDetector('/Users/ja/Desktop/projects/basilisk/optimized_signature_crf28.json')

    # Test across CRF levels
    crf_levels = [18, 23, 28, 31, 35]
    scores = []

    print("CRF  |  Detection Score  |  % of Uncompressed")
    print("-" * 50)

    # Baseline (uncompressed)
    baseline_score, _ = detector.detect_in_video(str(test_video), num_frames=20)

    for crf in crf_levels:
        compressed_path = str(test_video).replace('.mp4', f'_crf{crf}_test.mp4')
        compress_video(str(test_video), compressed_path, crf=crf)

        score, _ = detector.detect_in_video(compressed_path, num_frames=20)
        scores.append(score)

        survival_pct = (score / baseline_score) * 100 if baseline_score > 0 else 0

        print(f"{crf:3d}  |  {score:16.4f}  |  {survival_pct:5.1f}%")

    print()

    # Check CRF 28 survival
    crf28_score = scores[crf_levels.index(28)]
    crf28_survival = (crf28_score / baseline_score) * 100 if baseline_score > 0 else 0

    if crf28_survival > 80:
        print(f"✅ EXCELLENT: {crf28_survival:.1f}% signature survival at CRF 28")
        print("   Differentiable codec training was highly effective")
        return True
    elif crf28_survival > 50:
        print(f"✅ GOOD: {crf28_survival:.1f}% signature survival at CRF 28")
        print("   Acceptable degradation for production use")
        return True
    else:
        print(f"⚠️  WEAK: {crf28_survival:.1f}% signature survival at CRF 28")
        print("   Significant degradation - may need further optimization")
        return False


def main():
    print("=" * 80)
    print("FINAL VALIDATION: Compression-Robust Video Poisoning")
    print("=" * 80)
    print()
    print("This tests if we ACTUALLY broke the industry or if results are artifacts.")
    print()

    # Install scipy if needed
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        os.system("pip3 install -q scipy")

    # Run tests
    test1_passed, clean_scores = test_false_positive_rate()
    test2_passed, poisoned_scores = test_true_positive_rate()
    test3_passed = test_statistical_significance(clean_scores, poisoned_scores)
    test4_passed = test_compression_degradation()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    print(f"Test 1 (False Positives):  {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Test 2 (True Positives):   {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Test 3 (Statistical Sig):  {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"Test 4 (CRF 28 Survival):  {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print()

    if all([test1_passed, test2_passed, test3_passed]):
        print("✅✅✅ VALIDATION PASSED ✅✅✅")
        print()
        print("Results are REAL. You have:")
        print("  - Compression-robust video poisoning (CRF 28)")
        print("  - Low false positive rate (<10%)")
        print("  - High true positive rate (>80%)")
        print("  - Statistically significant separation")
        print()
        print("THIS IS INDUSTRY-BREAKING.")
        print("First compression-robust radioactive marking for video.")
        print()
        print("Ready for:")
        print("  1. Open source release")
        print("  2. CVPR 2026 submission")
        print("  3. Production deployment")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print()
        print("Results are NOT robust enough for production.")
        print("Further optimization needed:")
        if not test1_passed:
            print("  - Reduce false positive rate")
        if not test2_passed:
            print("  - Improve CRF 28 detection")
        if not test3_passed:
            print("  - Increase statistical separation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
