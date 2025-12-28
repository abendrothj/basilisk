#!/usr/bin/env python3
"""
CMA-ES Signature Optimization for REAL H.264 Compression

Why CMA-ES instead of gradient descent:
- Cannot backpropagate through real ffmpeg H.264 encoding
- Differentiable codecs don't accurately model real compression
- CMA-ES is gradient-free, evaluates on REAL compression

Approach:
1. Population-based search for optimal signature (8x8 DCT pattern, epsilon)
2. Each candidate evaluated on REAL H.264 CRF 28 compression
3. Fitness = separation between clean and poisoned videos
4. Optimize to maximize separation while maintaining quality

This is SLOW (each evaluation requires ffmpeg) but ACCURATE.
"""

import sys
import os
import numpy as np
import subprocess
from pathlib import Path
import cv2
from cmaes import CMA

sys.path.append(os.path.join(os.path.dirname(__file__), 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker
from frequency_detector import FrequencySignatureDetector


def generate_test_videos(output_dir: str, num_videos: int = 5):
    """Generate small diverse test videos."""
    Path(output_dir).mkdir(exist_ok=True)

    video_types = ['gradient', 'shapes', 'noise']
    video_paths = []

    for i in range(num_videos):
        video_type = video_types[i % len(video_types)]
        video_path = f'{output_dir}/test_{video_type}_{i}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (224, 224))

        for frame_idx in range(30):  # Only 30 frames for speed
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

            if video_type == 'gradient':
                for y in range(224):
                    intensity = int(50 + 150 * y / 224)
                    frame[y, :] = (intensity, int(intensity * 0.8), int(intensity * 0.6))

            elif video_type == 'shapes':
                frame[:, :] = (100, 100, 100)
                for _ in range(3):
                    x = np.random.randint(20, 200)
                    y = np.random.randint(20, 200)
                    size = np.random.randint(10, 30)
                    color = tuple(np.random.randint(50, 255, 3).tolist())
                    cv2.circle(frame, (x, y), size, color, -1)

            else:  # noise
                frame = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

            out.write(frame)

        out.release()
        video_paths.append(video_path)

    return video_paths


def compress_video(input_path: str, output_path: str, crf: int = 28) -> bool:
    """Compress with real H.264."""
    cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', str(crf),
           '-preset', 'medium', '-y', output_path]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except:
        return False


def evaluate_signature(signature_flat: np.ndarray, test_videos: list, temp_dir: str) -> float:
    """
    Evaluate signature fitness on REAL H.264 compression.

    Args:
        signature_flat: Flattened signature vector [sig_8x8_flat (low-freq only), epsilon]

    Returns:
        Fitness score (higher = better separation)
    """
    # Decode signature
    # First 9 elements are low-freq DCT coefficients (3x3 grid)
    # Last element is epsilon
    sig_low_freq = signature_flat[:9]
    epsilon = signature_flat[9]

    # Clamp epsilon to valid range
    epsilon = np.clip(epsilon, 0.01, 0.10)

    # Create 8x8 signature (low-freq only, as in original approach)
    signature_dct = np.zeros((8, 8), dtype=np.float32)
    signature_dct[:3, :3] = sig_low_freq.reshape(3, 3)

    # Normalize
    norm = np.linalg.norm(signature_dct)
    if norm > 0:
        signature_dct = signature_dct / norm

    # Create temp marker
    marker = FrequencyDomainVideoMarker(epsilon=epsilon, frequency_band='low')
    marker.signature_dct = signature_dct

    # Save signature
    sig_path = f'{temp_dir}/temp_sig.json'
    marker.save_signature(sig_path)

    detector = FrequencySignatureDetector(sig_path)

    clean_scores = []
    poisoned_scores = []

    for video_path in test_videos:
        # Poison
        poisoned_path = f'{temp_dir}/poisoned_{Path(video_path).name}'
        marker.poison_video(video_path, poisoned_path, verbose=False)

        # Compress both
        clean_crf = f'{temp_dir}/clean_crf_{Path(video_path).name}'
        poisoned_crf = f'{temp_dir}/poisoned_crf_{Path(video_path).name}'

        if not compress_video(video_path, clean_crf, crf=28):
            return -1.0  # Penalty for failure
        if not compress_video(poisoned_path, poisoned_crf, crf=28):
            return -1.0

        # Detect
        try:
            clean_score, _ = detector.detect_in_video(clean_crf, num_frames=10)
            poisoned_score, _ = detector.detect_in_video(poisoned_crf, num_frames=10)

            clean_scores.append(clean_score)
            poisoned_scores.append(poisoned_score)
        except:
            return -1.0

    # Compute fitness
    clean_mean = np.mean(clean_scores)
    poisoned_mean = np.mean(poisoned_scores)
    separation = poisoned_mean - clean_mean

    # Penalties
    # 1. False positive penalty (clean should be <0.1)
    fpr_penalty = np.maximum(clean_mean - 0.1, 0) * 5.0

    # 2. Low TPR penalty (poisoned should be >0.3)
    tpr_penalty = np.maximum(0.3 - poisoned_mean, 0) * 5.0

    fitness = separation - fpr_penalty - tpr_penalty

    return fitness


def optimize_signature_cmaes(
    test_videos: list,
    num_iterations: int = 50,
    population_size: int = 10
):
    """
    Optimize signature using CMA-ES.

    Args:
        test_videos: List of test video paths
        num_iterations: Number of CMA-ES iterations
        population_size: Population size per iteration
    """
    print("=" * 80)
    print("CMA-ES SIGNATURE OPTIMIZATION FOR REAL H.264 CRF 28")
    print("=" * 80)
    print()
    print(f"Test videos: {len(test_videos)}")
    print(f"Iterations: {num_iterations}")
    print(f"Population size: {population_size}")
    print()
    print("This will take a while (each evaluation requires ffmpeg compression)...")
    print()

    # Create temp directory
    temp_dir = '/tmp/cmaes_opt'
    Path(temp_dir).mkdir(exist_ok=True)

    # Initial guess: random low-freq signature + epsilon=0.03
    initial_guess = np.concatenate([
        np.random.randn(9) * 0.1,  # 3x3 low-freq signature
        [0.03]  # epsilon
    ])

    # CMA-ES optimizer
    optimizer = CMA(mean=initial_guess, sigma=0.1, population_size=population_size)

    best_fitness = -np.inf
    best_signature = None

    for generation in range(num_iterations):
        print(f"Generation {generation + 1}/{num_iterations}")

        solutions = []
        fitnesses = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = evaluate_signature(x, test_videos, temp_dir)
            solutions.append(x)
            fitnesses.append(fitness)

        # Update optimizer
        optimizer.tell(solutions, fitnesses)

        # Track best
        max_fitness_idx = np.argmax(fitnesses)
        max_fitness = fitnesses[max_fitness_idx]

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_signature = solutions[max_fitness_idx]

        print(f"  Best fitness this gen: {max_fitness:.4f}")
        print(f"  Best overall: {best_fitness:.4f}")
        print(f"  Mean fitness: {np.mean(fitnesses):.4f}")
        print()

    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best fitness: {best_fitness:.4f}")
    print()

    # Decode best signature
    sig_low_freq = best_signature[:9]
    epsilon_best = np.clip(best_signature[9], 0.01, 0.10)

    signature_dct = np.zeros((8, 8), dtype=np.float32)
    signature_dct[:3, :3] = sig_low_freq.reshape(3, 3)
    norm = np.linalg.norm(signature_dct)
    if norm > 0:
        signature_dct = signature_dct / norm

    print(f"Best epsilon: {epsilon_best:.4f}")
    print()
    print("Best signature (3x3 low-freq):")
    print(signature_dct[:3, :3])
    print()

    # Save
    marker = FrequencyDomainVideoMarker(epsilon=epsilon_best, frequency_band='low')
    marker.signature_dct = signature_dct
    marker.save_signature('cmaes_signature_crf28.json')

    print("✓ Saved to cmaes_signature_crf28.json")

    return signature_dct, epsilon_best


if __name__ == '__main__':
    # Generate test videos
    print("Generating test videos...")
    test_videos = generate_test_videos('/tmp/cmaes_test_videos', num_videos=3)
    print(f"Generated {len(test_videos)} test videos")
    print()

    # Optimize
    optimize_signature_cmaes(
        test_videos=test_videos,
        num_iterations=30,  # Start with 30 (can increase if needed)
        population_size=8
    )
