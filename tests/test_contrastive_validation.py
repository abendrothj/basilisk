#!/usr/bin/env python3
"""
FINAL VALIDATION: Test contrastive signature on REAL H.264 compression.

Critical: Must test with real ffmpeg CRF 28, not differentiable approximation.
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


def create_validation_videos(output_dir: str, num_videos: int = 15):
    """Create diverse clean videos for validation."""
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
                for y in range(224):
                    intensity = int(50 + 150 * y / 224)
                    frame[y, :] = (intensity, int(intensity * 0.8), int(intensity * 0.6))

            elif video_type == 'shapes':
                frame[:, :] = (100, 100, 100)
                for _ in range(5):
                    x = np.random.randint(20, 200)
                    y = np.random.randint(20, 200)
                    size = np.random.randint(10, 30)
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    cv2.circle(frame, (x, y), size, color, -1)

            elif video_type == 'noise':
                frame = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

            elif video_type == 'text':
                frame[:, :] = (200, 200, 200)
                cv2.putText(frame, f'Frame {frame_idx}', (20, 112),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            else:  # mixed
                for y in range(224):
                    intensity = int(50 + 150 * y / 224)
                    frame[y, :] = (intensity, int(intensity * 0.7), int(intensity * 0.5))

                t = frame_idx / 60.0
                x = int(112 + 50 * np.sin(2 * np.pi * t * 2))
                y = int(112 + 50 * np.cos(2 * np.pi * t * 3))
                cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)

            out.write(frame)

        out.release()

    print(f"Created {num_videos} validation videos")


def compress_video(input_path: str, output_path: str, crf: int) -> bool:
    """Compress with H.264."""
    cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', str(crf),
           '-preset', 'medium', '-y', output_path]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return result.returncode == 0
    except:
        return False


def main():
    print("=" * 80)
    print("CONTRASTIVE SIGNATURE VALIDATION (REAL H.264 CRF 28)")
    print("=" * 80)
    print()

    # Create validation videos
    print("Creating validation videos...")
    create_validation_videos('/tmp/contrastive_validation', num_videos=15)
    print()

    # Load contrastive signature
    signature_path = '/Users/ja/Desktop/projects/basilisk/contrastive_signature_crf28.json'

    if not os.path.exists(signature_path):
        print(f"❌ Signature not found: {signature_path}")
        return 1

    marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low')
    marker.load_signature(signature_path)

    detector = FrequencySignatureDetector(signature_path)

    print("Testing on REAL H.264 compression (CRF 28)...")
    print()

    clean_scores = []
    poisoned_scores = []

    for video_path in Path('/tmp/contrastive_validation').glob('*.mp4'):
        # Test clean video (compressed)
        clean_compressed_path = str(video_path).replace('.mp4', '_clean_crf28.mp4')
        compress_video(str(video_path), clean_compressed_path, crf=28)

        clean_score, _ = detector.detect_in_video(clean_compressed_path, num_frames=20)
        clean_scores.append(clean_score)

        # Poison video
        poisoned_path = str(video_path).replace('clean_', 'poisoned_')
        marker.poison_video(str(video_path), poisoned_path, verbose=False)

        # Compress poisoned
        poisoned_compressed_path = poisoned_path.replace('.mp4', '_crf28.mp4')
        compress_video(poisoned_path, poisoned_compressed_path, crf=28)

        poisoned_score, _ = detector.detect_in_video(poisoned_compressed_path, num_frames=20)
        poisoned_scores.append(poisoned_score)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    clean_mean = np.mean(clean_scores)
    clean_std = np.std(clean_scores)
    poisoned_mean = np.mean(poisoned_scores)
    poisoned_std = np.std(poisoned_scores)

    print(f"Clean videos (CRF 28):    {clean_mean:.4f} ± {clean_std:.4f}")
    print(f"Poisoned videos (CRF 28): {poisoned_mean:.4f} ± {poisoned_std:.4f}")
    print()

    # Compute separation
    separation = poisoned_mean - clean_mean
    print(f"Separation: {separation:.4f}")
    print()

    # Compute TPR/FPR at threshold 0.3
    threshold = 0.3
    fpr = np.sum(np.array(clean_scores) > threshold) / len(clean_scores)
    tpr = np.sum(np.array(poisoned_scores) > threshold) / len(poisoned_scores)

    print(f"Threshold: {threshold}")
    print(f"False Positive Rate: {fpr*100:.1f}%")
    print(f"True Positive Rate:  {tpr*100:.1f}%")
    print()

    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(poisoned_scores, clean_scores)

    print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")
    print()

    # Pass/fail criteria
    print("=" * 80)
    print("PASS/FAIL CRITERIA")
    print("=" * 80)
    print()

    test1_pass = fpr < 0.1
    test2_pass = tpr > 0.8
    test3_pass = p_value < 0.001

    print(f"1. False Positive Rate <10%:  {'✅ PASS' if test1_pass else '❌ FAIL'} ({fpr*100:.1f}%)")
    print(f"2. True Positive Rate >80%:   {'✅ PASS' if test2_pass else '❌ FAIL'} ({tpr*100:.1f}%)")
    print(f"3. Statistical Significance:  {'✅ PASS' if test3_pass else '❌ FAIL'} (p={p_value:.6f})")
    print()

    if all([test1_pass, test2_pass, test3_pass]):
        print("✅✅✅ VALIDATION PASSED ✅✅✅")
        print()
        print("Contrastive learning WORKED!")
        print("You have compression-robust video poisoning that:")
        print("  - Survives CRF 28 compression")
        print("  - Low false positive rate (<10%)")
        print("  - High true positive rate (>80%)")
        print("  - Statistically significant separation (p<0.001)")
        print()
        print("INDUSTRY-BREAKING RESULT.")
        print("First compression-robust radioactive marking for video.")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print()
        print("Further optimization needed:")
        if not test1_pass:
            print(f"  - Reduce false positive rate (currently {fpr*100:.1f}%)")
        if not test2_pass:
            print(f"  - Improve true positive rate (currently {tpr*100:.1f}%)")
        if not test3_pass:
            print(f"  - Increase statistical separation (p={p_value:.6f})")
        return 1


if __name__ == '__main__':
    sys.exit(main())
