#!/usr/bin/env python3
"""
Train compression-robust signature via differentiable H.264 proxy.

Goal: Find signature that maximizes detection score at CRF 28 while maintaining visual quality.

Strategy:
1. Start with baseline signature (random, low-freq only)
2. Generate synthetic video frames
3. Apply signature → compress with differentiable H.264 → measure detection
4. Optimize signature via gradient descent to maximize detection post-compression
5. Constrain: PSNR > 35 dB (visual quality)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'poison-core'))
from differentiable_codec import DifferentiableH264
from frequency_poison import FrequencyDomainVideoMarker
from frequency_detector import FrequencySignatureDetector


def generate_training_frames(num_frames: int = 50, resolution: int = 224) -> torch.Tensor:
    """
    Generate diverse synthetic frames for training.

    Returns: (num_frames, 3, H, W) tensor in range [0, 255]
    """
    frames = []

    for i in range(num_frames):
        # Create varied content
        frame = np.zeros((resolution, resolution, 3), dtype=np.float32)

        # Random gradient background
        for y in range(resolution):
            intensity = 50 + 150 * y / resolution + np.random.randn() * 10
            frame[y, :] = np.clip([intensity, intensity * 0.7, intensity * 0.5], 0, 255)

        # Random shapes
        num_shapes = np.random.randint(2, 5)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rect'])
            x = np.random.randint(20, resolution - 20)
            y = np.random.randint(20, resolution - 20)
            size = np.random.randint(20, 50)
            color = tuple(np.random.randint(0, 255, 3).tolist())

            if shape_type == 'circle':
                cv2.circle(frame, (x, y), size, color, -1)
            else:
                cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)

        frames.append(frame)

    # Convert to torch tensor (num_frames, 3, H, W)
    frames_np = np.array(frames).transpose(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
    frames_torch = torch.from_numpy(frames_np).float()

    return frames_torch


def apply_dct_signature(
    frames: torch.Tensor,
    signature: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Apply DCT signature to frames.

    Args:
        frames: (N, C, H, W) in range [0, 255]
        signature: (8, 8) DCT pattern
        epsilon: Perturbation strength

    Returns:
        Poisoned frames (N, C, H, W)
    """
    N, C, H, W = frames.shape

    # Process Y channel only (simplified)
    # Full version would do proper YCbCr conversion
    y_channel = 0.299 * frames[:, 2] + 0.587 * frames[:, 1] + 0.114 * frames[:, 0]  # (N, H, W)
    y_channel = y_channel.unsqueeze(1)  # (N, 1, H, W)

    # Unfold into 8x8 blocks
    num_blocks_h = H // 8
    num_blocks_w = W // 8

    blocks = y_channel.unfold(2, 8, 8).unfold(3, 8, 8)  # (N, 1, num_blocks_h, num_blocks_w, 8, 8)
    blocks = blocks.reshape(N, num_blocks_h * num_blocks_w, 8, 8)

    # Apply manual DCT (using cosine basis)
    dct_basis = _get_dct_basis(8).to(frames.device)
    dct_blocks = torch.einsum('...ij,jk->...ik', blocks, dct_basis.T)
    dct_blocks = torch.einsum('ij,...jk->...ik', dct_basis, dct_blocks)

    # Add signature
    signature_expanded = signature.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
    dct_blocks_poisoned = dct_blocks + epsilon * 255.0 * signature_expanded

    # Inverse DCT
    blocks_poisoned = torch.einsum('...ij,kj->...ik', dct_blocks_poisoned, dct_basis)
    blocks_poisoned = torch.einsum('jk,...jl->...kl', dct_basis.T, blocks_poisoned)

    # Fold back
    blocks_poisoned = blocks_poisoned.reshape(N, 1, num_blocks_h, num_blocks_w, 8, 8)
    blocks_poisoned = blocks_poisoned.permute(0, 1, 2, 4, 3, 5).contiguous()
    y_poisoned = blocks_poisoned.reshape(N, 1, H, W)

    # Replace Y channel in original frames (simplified)
    y_delta = y_poisoned.squeeze(1) - y_channel.squeeze(1)
    frames_poisoned = frames.clone()
    frames_poisoned[:, 0] += y_delta * 0.114  # B
    frames_poisoned[:, 1] += y_delta * 0.587  # G
    frames_poisoned[:, 2] += y_delta * 0.299  # R

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
    Differentiable detection score computation.

    Simplified: measures correlation between frame DCT AC coefficients and signature.
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


def train_adaptive_signature(
    target_crf: int = 28,
    num_iterations: int = 200,
    learning_rate: float = 0.01
):
    """
    Main training loop.

    Optimizes signature to maximize detection at target CRF while maintaining visual quality.
    """
    print("=" * 80)
    print("ADAPTIVE SIGNATURE TRAINING FOR CRF", target_crf)
    print("=" * 80)
    print()

    # Initialize signature (random, low-freq only)
    signature = torch.randn(8, 8)
    signature[3:, :] = 0  # Zero high frequencies
    signature[:, 3:] = 0
    signature[0, 0] = 0  # Zero DC (not used in detection)
    signature = signature / (torch.norm(signature) + 1e-8)
    signature = nn.Parameter(signature)

    # Initialize epsilon
    epsilon = nn.Parameter(torch.tensor(0.05))

    # Optimizer
    optimizer = torch.optim.Adam([signature, epsilon], lr=learning_rate)

    # Differentiable codec
    codec = DifferentiableH264(quality_factor=target_crf, temperature=0.1)

    # Training frames
    print("Generating training frames...")
    train_frames = generate_training_frames(num_frames=50, resolution=224)
    print(f"Generated {len(train_frames)} frames")
    print()

    print("Training...")
    print()

    best_score = 0.0
    best_signature = None
    best_epsilon = None

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Apply signature
        poisoned_frames = apply_dct_signature(train_frames, signature, epsilon)

        # Compress
        compressed_frames = codec(poisoned_frames)

        # Compute detection score on compressed frames
        detection_score = compute_detection_score_differentiable(compressed_frames, signature)

        # Visual quality loss
        mse = F.mse_loss(compressed_frames, train_frames)
        psnr = 10 * torch.log10(255.0**2 / (mse + 1e-8))

        # Combined loss
        # Maximize detection (minimize -detection)
        # Maintain quality (penalize if PSNR < 35)
        detection_loss = -detection_score
        quality_loss = torch.relu(35.0 - psnr) * 0.1  # Penalty if PSNR < 35

        total_loss = detection_loss + quality_loss

        # Backprop
        total_loss.backward()
        optimizer.step()

        # Project constraints
        with torch.no_grad():
            # Clamp epsilon
            epsilon.data.clamp_(0.01, 0.15)

            # Normalize signature
            signature.data = signature.data / (torch.norm(signature.data) + 1e-8)

            # Keep high frequencies zero
            signature.data[3:, :] = 0
            signature.data[:, 3:] = 0
            signature.data[0, 0] = 0  # Keep DC zero

        # Track best
        if detection_score.item() > best_score:
            best_score = detection_score.item()
            best_signature = signature.data.clone()
            best_epsilon = epsilon.item()

        # Print progress
        if (iteration + 1) % 20 == 0:
            print(f"Iter {iteration+1:3d}: Detection={detection_score.item():.4f}, "
                  f"PSNR={psnr.item():.2f} dB, Epsilon={epsilon.item():.4f}, "
                  f"Loss={total_loss.item():.4f}")

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best detection score: {best_score:.4f}")
    print(f"Best epsilon: {best_epsilon:.4f}")
    print()

    # Save optimized signature
    marker = FrequencyDomainVideoMarker(epsilon=best_epsilon, frequency_band='low')
    marker.signature_dct = best_signature.cpu().numpy()
    marker.save_signature(f'optimized_signature_crf{target_crf}.json')

    print(f"✓ Saved to optimized_signature_crf{target_crf}.json")
    print()

    return best_signature, best_epsilon


if __name__ == '__main__':
    train_adaptive_signature(target_crf=28, num_iterations=200, learning_rate=0.01)
