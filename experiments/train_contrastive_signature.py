#!/usr/bin/env python3
"""
Contrastive Signature Training for Compression-Robust Video Poisoning

ROOT CAUSE OF PREVIOUS FAILURE:
- Detection measured correlation with random signature
- Natural videos have high DCT variation → false positives (40%)
- Poisoned videos after compression → weak signal (30% TPR)
- No statistical separation (p=0.64)

NEW APPROACH: Contrastive Learning
Train signature to MAXIMIZE separation between clean and poisoned distributions.

Key Changes:
1. Use BOTH clean and poisoned samples during training
2. Optimize for margin between distributions (not just positive correlation)
3. Use triplet loss or contrastive loss
4. Validate on held-out clean samples throughout training

Goal: Achieve >90% TPR, <10% FPR at CRF 28
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), 'poison-core'))
from differentiable_codec import DifferentiableH264
from frequency_poison import FrequencyDomainVideoMarker
from frequency_detector import FrequencySignatureDetector


def generate_diverse_videos(output_dir: str, num_videos: int = 20) -> List[str]:
    """Generate diverse clean videos for training."""
    Path(output_dir).mkdir(exist_ok=True)

    video_types = ['gradient', 'shapes', 'noise', 'text', 'mixed']
    video_paths = []

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
        video_paths.append(video_path)

    print(f"Generated {num_videos} clean videos")
    return video_paths


def load_video_frames(video_path: str, num_frames: int = 20) -> torch.Tensor:
    """Load frames from video as tensor."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    # Convert to tensor (N, C, H, W)
    frames_np = np.array(frames).transpose(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
    frames_torch = torch.from_numpy(frames_np).float()

    return frames_torch


def apply_dct_signature(
    frames: torch.Tensor,
    signature: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """Apply DCT signature to frames (differentiable)."""
    N, C, H, W = frames.shape

    # Process Y channel only
    y_channel = 0.299 * frames[:, 2] + 0.587 * frames[:, 1] + 0.114 * frames[:, 0]
    y_channel = y_channel.unsqueeze(1)

    # Unfold into 8x8 blocks
    num_blocks_h = H // 8
    num_blocks_w = W // 8

    blocks = y_channel.unfold(2, 8, 8).unfold(3, 8, 8)
    blocks = blocks.reshape(N, num_blocks_h * num_blocks_w, 8, 8)

    # Apply DCT
    dct_basis = _get_dct_basis(8).to(frames.device)
    dct_blocks = torch.einsum('...ij,jk->...ik', blocks, dct_basis.T)
    dct_blocks = torch.einsum('ij,...jk->...ik', dct_basis, dct_blocks)

    # Add signature
    signature_expanded = signature.unsqueeze(0).unsqueeze(0)
    dct_blocks_poisoned = dct_blocks + epsilon * 255.0 * signature_expanded

    # Inverse DCT
    blocks_poisoned = torch.einsum('...ij,kj->...ik', dct_blocks_poisoned, dct_basis)
    blocks_poisoned = torch.einsum('jk,...jl->...kl', dct_basis.T, blocks_poisoned)

    # Fold back
    blocks_poisoned = blocks_poisoned.reshape(N, 1, num_blocks_h, num_blocks_w, 8, 8)
    blocks_poisoned = blocks_poisoned.permute(0, 1, 2, 4, 3, 5).contiguous()
    y_poisoned = blocks_poisoned.reshape(N, 1, H, W)

    # Replace Y channel
    y_delta = y_poisoned.squeeze(1) - y_channel.squeeze(1)
    frames_poisoned = frames.clone()
    frames_poisoned[:, 0] += y_delta * 0.114
    frames_poisoned[:, 1] += y_delta * 0.587
    frames_poisoned[:, 2] += y_delta * 0.299

    frames_poisoned = torch.clamp(frames_poisoned, 0, 255)

    return frames_poisoned


def _get_dct_basis(N: int) -> torch.Tensor:
    """DCT basis matrix."""
    basis = torch.zeros(N, N)
    for k in range(N):
        for n in range(N):
            if k == 0:
                basis[k, n] = np.sqrt(1.0 / N)
            else:
                basis[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2*n + 1) / (2*N))
    return basis


def compute_detection_score_differentiable(
    frames: torch.Tensor,
    signature: torch.Tensor
) -> torch.Tensor:
    """
    Differentiable detection score.

    Measures mean absolute correlation between AC coefficients and signature.
    """
    N, C, H, W = frames.shape

    # Extract Y channel
    y_channel = 0.299 * frames[:, 2] + 0.587 * frames[:, 1] + 0.114 * frames[:, 0]
    y_channel = y_channel.unsqueeze(1)

    # Unfold into blocks
    num_blocks_h = H // 8
    num_blocks_w = W // 8
    blocks = y_channel.unfold(2, 8, 8).unfold(3, 8, 8)
    blocks = blocks.reshape(N, num_blocks_h * num_blocks_w, 8, 8)

    # DCT
    dct_basis = _get_dct_basis(8).to(frames.device)
    dct_blocks = torch.einsum('...ij,jk->...ik', blocks, dct_basis.T)
    dct_blocks = torch.einsum('ij,...jk->...ik', dct_basis, dct_blocks)

    # Zero out DC
    dct_blocks_ac = dct_blocks.clone()
    dct_blocks_ac[:, :, 0, 0] = 0

    # Normalize
    dct_blocks_ac_flat = dct_blocks_ac.reshape(N * num_blocks_h * num_blocks_w, 64)
    dct_blocks_ac_norm = dct_blocks_ac_flat / (torch.norm(dct_blocks_ac_flat, dim=1, keepdim=True) + 1e-8)

    # Signature AC (zero DC)
    signature_ac = signature.clone()
    signature_ac[0, 0] = 0
    signature_ac_norm = signature_ac / (torch.norm(signature_ac) + 1e-8)

    # Correlation
    correlations = torch.matmul(dct_blocks_ac_norm, signature_ac_norm.flatten())

    # Detection score: mean absolute correlation
    detection_score = torch.mean(torch.abs(correlations))

    return detection_score


def contrastive_loss(
    poisoned_score: torch.Tensor,
    clean_score: torch.Tensor,
    margin: float = 0.3
) -> torch.Tensor:
    """
    Contrastive loss: maximize separation between poisoned and clean.

    Goal:
    - Poisoned videos should have HIGH score (>0.5)
    - Clean videos should have LOW score (<0.1)
    - Margin: minimum separation (0.3 means 30% difference)

    Loss components:
    1. Push poisoned score up (maximize)
    2. Push clean score down (minimize)
    3. Enforce margin between them
    """
    # Loss 1: Poisoned should score high
    poisoned_loss = torch.relu(0.5 - poisoned_score)

    # Loss 2: Clean should score low
    clean_loss = torch.relu(clean_score - 0.1)

    # Loss 3: Margin between poisoned and clean
    margin_loss = torch.relu(margin - (poisoned_score - clean_score))

    total_loss = poisoned_loss + clean_loss + margin_loss

    return total_loss


def train_contrastive_signature(
    target_crf: int = 28,
    num_iterations: int = 500,
    learning_rate: float = 0.005,
    num_train_videos: int = 10,
    num_val_videos: int = 5
):
    """
    Main contrastive training loop.

    Key difference from previous approach:
    - Uses BOTH clean and poisoned samples
    - Optimizes for SEPARATION (not just positive correlation)
    - Validates on held-out clean samples
    """
    print("=" * 80)
    print("CONTRASTIVE SIGNATURE TRAINING FOR CRF", target_crf)
    print("=" * 80)
    print()
    print("This training optimizes for SEPARATION between clean and poisoned.")
    print("Previous failure: optimized for correlation (picked up noise).")
    print()

    # Generate clean videos
    print("Generating training videos...")
    train_clean_paths = generate_diverse_videos('/tmp/contrastive_train_clean', num_train_videos)
    val_clean_paths = generate_diverse_videos('/tmp/contrastive_val_clean', num_val_videos)
    print()

    # Initialize signature (random, low-freq only)
    signature = torch.randn(8, 8)
    signature[3:, :] = 0  # Zero high frequencies
    signature[:, 3:] = 0
    signature[0, 0] = 0  # Zero DC
    signature = signature / (torch.norm(signature) + 1e-8)
    signature = nn.Parameter(signature)

    # Initialize epsilon
    epsilon = nn.Parameter(torch.tensor(0.02))

    # Optimizer
    optimizer = torch.optim.Adam([signature, epsilon], lr=learning_rate)

    # Differentiable codec (with straight-through estimator)
    codec = DifferentiableH264(quality_factor=target_crf, temperature=0.1)
    codec.train()  # Enable training mode for straight-through estimator

    print("Training...")
    print()

    best_separation = 0.0
    best_signature = None
    best_epsilon = None

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Sample random training video
        clean_path = np.random.choice(train_clean_paths)
        clean_frames = load_video_frames(clean_path, num_frames=10)

        # Poison frames
        poisoned_frames = apply_dct_signature(clean_frames, signature, epsilon)

        # Compress both clean and poisoned
        clean_compressed = codec(clean_frames)
        poisoned_compressed = codec(poisoned_frames)

        # Compute detection scores
        clean_score = compute_detection_score_differentiable(clean_compressed, signature)
        poisoned_score = compute_detection_score_differentiable(poisoned_compressed, signature)

        # Contrastive loss
        loss = contrastive_loss(poisoned_score, clean_score, margin=0.3)

        # Visual quality constraint
        mse = F.mse_loss(poisoned_frames, clean_frames)
        psnr = 10 * torch.log10(255.0**2 / (mse + 1e-8))
        quality_loss = torch.relu(35.0 - psnr) * 0.5

        total_loss = loss + quality_loss

        # Backprop
        total_loss.backward()
        optimizer.step()

        # Project constraints
        with torch.no_grad():
            epsilon.data.clamp_(0.005, 0.05)
            signature.data = signature.data / (torch.norm(signature.data) + 1e-8)
            signature.data[3:, :] = 0
            signature.data[:, 3:] = 0
            signature.data[0, 0] = 0

        # Track best separation
        separation = poisoned_score.item() - clean_score.item()
        if separation > best_separation:
            best_separation = separation
            best_signature = signature.data.clone()
            best_epsilon = epsilon.item()

        # Print progress
        if (iteration + 1) % 50 == 0:
            print(f"Iter {iteration+1:3d}: "
                  f"Poisoned={poisoned_score.item():.4f}, "
                  f"Clean={clean_score.item():.4f}, "
                  f"Separation={separation:.4f}, "
                  f"PSNR={psnr.item():.2f} dB, "
                  f"Epsilon={epsilon.item():.4f}")

        # Validation every 100 iterations
        if (iteration + 1) % 100 == 0:
            print()
            print(f"--- Validation at iteration {iteration+1} ---")

            codec.eval()  # Use hard quantization for validation

            with torch.no_grad():
                val_clean_scores = []
                val_poisoned_scores = []

                for val_path in val_clean_paths:
                    val_frames = load_video_frames(val_path, num_frames=10)
                    val_poisoned = apply_dct_signature(val_frames, signature, epsilon)

                    val_clean_compressed = codec(val_frames)
                    val_poisoned_compressed = codec(val_poisoned)

                    val_clean_score = compute_detection_score_differentiable(val_clean_compressed, signature)
                    val_poisoned_score = compute_detection_score_differentiable(val_poisoned_compressed, signature)

                    val_clean_scores.append(val_clean_score.item())
                    val_poisoned_scores.append(val_poisoned_score.item())

                val_clean_mean = np.mean(val_clean_scores)
                val_poisoned_mean = np.mean(val_poisoned_scores)
                val_separation = val_poisoned_mean - val_clean_mean

                # Estimate TPR/FPR (threshold=0.3)
                fpr = np.sum(np.array(val_clean_scores) > 0.3) / len(val_clean_scores)
                tpr = np.sum(np.array(val_poisoned_scores) > 0.3) / len(val_poisoned_scores)

                print(f"[Diff Codec] Val Clean: {val_clean_mean:.4f}")
                print(f"[Diff Codec] Val Poisoned: {val_poisoned_mean:.4f}")
                print(f"[Diff Codec] Val Separation: {val_separation:.4f}")
                print(f"[Diff Codec] Estimated FPR: {fpr*100:.1f}%")
                print(f"[Diff Codec] Estimated TPR: {tpr*100:.1f}%")

            # CRITICAL: Validate on REAL H.264 every 100 iterations
            if (iteration + 1) % 100 == 0:
                print()
                print("[REAL H.264] Testing on actual ffmpeg compression...")

                # Save current signature
                temp_sig_path = '/tmp/temp_signature.json'
                temp_marker = FrequencyDomainVideoMarker(epsilon=epsilon.item(), frequency_band='low')
                temp_marker.signature_dct = signature.detach().cpu().numpy()
                temp_marker.save_signature(temp_sig_path)

                from frequency_detector import FrequencySignatureDetector
                temp_detector = FrequencySignatureDetector(temp_sig_path)

                # Test on one validation video
                test_video = val_clean_paths[0]

                # Poison and compress with real H.264
                poisoned_test = test_video.replace('clean_', 'poisoned_test_')
                temp_marker.poison_video(test_video, poisoned_test, verbose=False)

                # Compress both clean and poisoned
                clean_crf28 = test_video.replace('.mp4', '_real_crf28.mp4')
                poisoned_crf28 = poisoned_test.replace('.mp4', '_real_crf28.mp4')

                subprocess.run(['ffmpeg', '-i', test_video, '-c:v', 'libx264', '-crf', '28',
                               '-preset', 'medium', '-y', clean_crf28],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run(['ffmpeg', '-i', poisoned_test, '-c:v', 'libx264', '-crf', '28',
                               '-preset', 'medium', '-y', poisoned_crf28],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Detect
                real_clean_score, _ = temp_detector.detect_in_video(clean_crf28, num_frames=10)
                real_poisoned_score, _ = temp_detector.detect_in_video(poisoned_crf28, num_frames=10)
                real_separation = real_poisoned_score - real_clean_score

                print(f"[REAL H.264] Clean: {real_clean_score:.4f}")
                print(f"[REAL H.264] Poisoned: {real_poisoned_score:.4f}")
                print(f"[REAL H.264] Separation: {real_separation:.4f}")

                if real_separation < 0.05:
                    print("⚠️  WARNING: Low separation on real H.264! Codec mismatch detected.")
                else:
                    print("✓ Real H.264 separation looks good!")

                print()

            codec.train()  # Back to training mode

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best separation: {best_separation:.4f}")
    print(f"Best epsilon: {best_epsilon:.4f}")
    print()

    # Save optimized signature
    marker = FrequencyDomainVideoMarker(epsilon=best_epsilon, frequency_band='low')
    marker.signature_dct = best_signature.cpu().numpy()
    marker.save_signature(f'contrastive_signature_crf{target_crf}.json')

    print(f"✓ Saved to contrastive_signature_crf{target_crf}.json")
    print()

    return best_signature, best_epsilon


if __name__ == '__main__':
    train_contrastive_signature(
        target_crf=28,
        num_iterations=500,
        learning_rate=0.005,
        num_train_videos=10,
        num_val_videos=5
    )
