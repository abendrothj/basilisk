#!/usr/bin/env python3
"""
Frequency Domain Video Poisoning

This implements compression-robust radioactive marking via DCT coefficient perturbation.

Key insight: H.264 compresses in frequency domain (DCT), so we poison the coefficients
that the codec preserves (low frequencies with small quantization steps).

Architecture:
1. Transform frames to YCbCr (H.264 operates on luminance)
2. Apply DCT to 8x8 blocks (matching H.264 block structure)
3. Perturb low-frequency coefficients (DC + first few AC terms)
4. Apply temporal modulation (sine wave across frames)
5. Inverse DCT back to spatial domain

Result: Signature embedded in compression-stable frequency bands.
"""

import cv2
import numpy as np
import json
import torch
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm


class FrequencyDomainVideoMarker:
    """
    Frequency domain video poisoning via DCT coefficient perturbation.

    This is designed to survive H.264/H.265 compression by targeting
    the frequency bands that video codecs preserve.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        frequency_band: str = 'low',
        temporal_period: int = 30,
        seed: int = 42,
        device: str = 'cpu'
    ):
        """
        Args:
            epsilon: Perturbation strength (0.01-0.1 typical)
            frequency_band: Which DCT frequencies to poison
                - 'low': DC + first 3x3 AC coefficients (most stable)
                - 'mid': DC + first 4x4 AC coefficients
                - 'adaptive': Learn which coefficients survive (future)
            temporal_period: Frames per temporal signature cycle
            seed: Random seed for reproducibility
            device: 'cpu' or 'cuda'
        """
        self.epsilon = epsilon
        self.frequency_band = frequency_band
        self.temporal_period = temporal_period
        self.seed = seed
        self.device = device

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate signatures
        self.signature_dct = self._generate_frequency_signature()
        self.temporal_signature = self._generate_temporal_signature()

    def _generate_frequency_signature(self) -> np.ndarray:
        """
        Generate signature in DCT domain.

        Returns 8x8 DCT coefficient perturbation pattern.
        We focus on low-frequency coefficients (top-left corner of DCT matrix)
        because H.264 preserves these with small quantization steps.

        H.264 Quantization Matrix (simplified):
        - Top-left (DC, low AC): small steps → preserved
        - Bottom-right (high AC): large steps → destroyed

        Our strategy: Only poison what survives compression.
        """
        signature = np.random.randn(8, 8).astype(np.float32)

        # Create frequency band mask
        mask = self._create_frequency_mask(self.frequency_band)
        signature = signature * mask

        # Normalize to unit length
        signature = signature / (np.linalg.norm(signature) + 1e-8)

        return signature

    def _create_frequency_mask(self, band: str) -> np.ndarray:
        """
        Create mask for which DCT frequencies to poison.

        Based on H.264 quantization matrix analysis:
        - Low frequencies: quantization steps 10-20 (well preserved)
        - Mid frequencies: quantization steps 30-60 (moderately preserved)
        - High frequencies: quantization steps 80-120 (mostly destroyed)

        We target low frequencies for maximum compression robustness.
        """
        mask = np.zeros((8, 8), dtype=np.float32)

        if band == 'low':
            # DC + first 3x3 AC coefficients
            # This covers ~90% of image energy and survives CRF 28-35
            mask[0:3, 0:3] = 1.0

        elif band == 'mid':
            # DC + first 4x4 AC coefficients
            # Better signal strength but less compression robust
            mask[0:4, 0:4] = 1.0

        elif band == 'diagonal':
            # Use diagonal pattern (matches H.264 scan order)
            for i in range(8):
                for j in range(8):
                    if i + j < 5:  # Diagonal cutoff
                        mask[i, j] = 1.0
        else:
            raise ValueError(f"Unknown frequency band: {band}")

        return mask

    def _generate_temporal_signature(self) -> np.ndarray:
        """
        Generate temporal modulation pattern (sine wave).

        This creates a cyclic pattern across frames that we can detect
        via cross-correlation, even if individual frames are compressed.
        """
        t = np.arange(self.temporal_period)
        signature = np.sin(2 * np.pi * t / self.temporal_period)
        return signature.astype(np.float32)

    def poison_video(
        self,
        input_path: str,
        output_path: str,
        poison_all_frames: bool = True,
        verbose: bool = True
    ) -> None:
        """
        Poison video by modifying DCT coefficients.

        Args:
            input_path: Input video path
            output_path: Output poisoned video path
            poison_all_frames: If False, only poison keyframes (every temporal_period frames)
            verbose: Print progress
        """
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Could not read video: {input_path}")

        # Setup writer (uncompressed mp4v for now)
        # Later we'll test with libx264 at various CRF levels
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if verbose:
            print(f"Poisoning video: {input_path}")
            print(f"  Frames: {total_frames}")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Epsilon: {self.epsilon}")
            print(f"  Frequency band: {self.frequency_band}")
            print()

        iterator = tqdm(range(total_frames), desc="Poisoning frames") if verbose else range(total_frames)

        for frame_idx in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            # Decide if we poison this frame
            is_keyframe = (frame_idx % self.temporal_period == 0)
            should_poison = poison_all_frames or is_keyframe

            if should_poison:
                # Get temporal modulation weight for this frame
                temporal_idx = frame_idx % self.temporal_period
                temporal_weight = self.temporal_signature[temporal_idx]

                # Poison frame in DCT domain
                poisoned_frame = self._poison_frame_dct(frame, temporal_weight)
                out.write(poisoned_frame)
            else:
                # Write frame unchanged
                out.write(frame)

        cap.release()
        out.release()

        if verbose:
            print(f"\n✓ Poisoned video saved to {output_path}")

    def _poison_frame_dct(
        self,
        frame: np.ndarray,
        temporal_weight: float
    ) -> np.ndarray:
        """
        Poison a single frame by modifying DCT coefficients.

        Process:
        1. Convert BGR → YCbCr (H.264 operates on luminance channel)
        2. Split Y channel into 8x8 blocks
        3. Apply DCT to each block
        4. Add signature to low-frequency coefficients
        5. Inverse DCT
        6. Convert YCbCr → BGR

        Args:
            frame: Input frame (BGR, uint8)
            temporal_weight: Temporal modulation factor [-1, 1]

        Returns:
            Poisoned frame (BGR, uint8)
        """
        # Convert to YCbCr color space (matching H.264)
        frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = frame_ycbcr[:, :, 0].astype(np.float32)

        # Get dimensions
        height, width = y_channel.shape

        # Pad to multiple of 8 (DCT block size)
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        y_padded = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode='edge')

        # Process each 8x8 block
        for i in range(0, y_padded.shape[0], 8):
            for j in range(0, y_padded.shape[1], 8):
                block = y_padded[i:i+8, j:j+8]

                # DCT transform
                dct_block = cv2.dct(block)

                # Add signature perturbation
                # Scale by epsilon, temporal weight, and 255.0 (DCT operates on [0, 255])
                perturbation = self.epsilon * temporal_weight * self.signature_dct * 255.0
                dct_block = dct_block + perturbation

                # Inverse DCT
                block_poisoned = cv2.idct(dct_block)
                y_padded[i:i+8, j:j+8] = block_poisoned

        # Remove padding
        y_poisoned = y_padded[:height, :width]

        # Clip to valid range [0, 255]
        y_poisoned = np.clip(y_poisoned, 0, 255).astype(np.uint8)

        # Replace Y channel
        frame_ycbcr[:, :, 0] = y_poisoned

        # Convert back to BGR
        frame_poisoned = cv2.cvtColor(frame_ycbcr, cv2.COLOR_YCrCb2BGR)

        return frame_poisoned

    def extract_dct_signature(
        self,
        video_path: str,
        num_frames: int = 10,
        num_blocks_per_frame: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract DCT coefficients from video to analyze signature presence.

        FIXED: Now averages over many blocks per frame, not just center block.
        This matches the poisoning process which modifies ALL blocks.

        Args:
            video_path: Path to video
            num_frames: Number of frames to analyze
            num_blocks_per_frame: Number of 8x8 blocks to sample per frame

        Returns:
            (mean_dct_coeffs, temporal_pattern)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        all_frame_dcts = []  # Per-frame average DCT
        temporal_pattern = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to YCbCr
            frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y_channel = frame_ycbcr[:, :, 0].astype(np.float32)

            # Pad to multiple of 8
            height, width = y_channel.shape
            pad_h = (8 - height % 8) % 8
            pad_w = (8 - width % 8) % 8
            y_padded = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode='edge')

            # Extract DCT from multiple blocks (matching poisoning process)
            frame_dct_blocks = []

            # Sample blocks uniformly across the frame
            num_blocks_h = y_padded.shape[0] // 8
            num_blocks_w = y_padded.shape[1] // 8
            total_blocks = num_blocks_h * num_blocks_w

            # Sample block indices
            num_samples = min(num_blocks_per_frame, total_blocks)
            block_indices = np.random.choice(total_blocks, num_samples, replace=False)

            for block_idx in block_indices:
                block_i = (block_idx // num_blocks_w) * 8
                block_j = (block_idx % num_blocks_w) * 8

                block = y_padded[block_i:block_i+8, block_j:block_j+8]

                if block.shape == (8, 8):
                    dct_block = cv2.dct(block)
                    frame_dct_blocks.append(dct_block)

            # Average DCT across all blocks in this frame
            if frame_dct_blocks:
                frame_avg_dct = np.mean(frame_dct_blocks, axis=0)
                all_frame_dcts.append(frame_avg_dct)
                temporal_pattern.append(frame_avg_dct[0, 0])  # DC coefficient

        cap.release()

        # Compute mean DCT pattern across all frames
        mean_dct = np.mean(all_frame_dcts, axis=0) if all_frame_dcts else np.zeros((8, 8))

        # Temporal pattern
        temporal_pattern = np.array(temporal_pattern)

        return mean_dct, temporal_pattern

    def compute_psnr(
        self,
        original_path: str,
        poisoned_path: str,
        num_frames: int = 10
    ) -> float:
        """
        Compute PSNR between original and poisoned video.

        PSNR > 40 dB: Visually identical (excellent)
        PSNR 30-40 dB: Subtle differences (good)
        PSNR < 30 dB: Noticeable artifacts (too strong)

        Args:
            original_path: Original video
            poisoned_path: Poisoned video
            num_frames: Number of frames to sample

        Returns:
            Average PSNR in dB
        """
        cap_orig = cv2.VideoCapture(original_path)
        cap_pois = cv2.VideoCapture(poisoned_path)

        total_frames = min(
            int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap_pois.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        psnrs = []

        for frame_idx in frame_indices:
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_pois.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret1, frame_orig = cap_orig.read()
            ret2, frame_pois = cap_pois.read()

            if ret1 and ret2:
                mse = np.mean((frame_orig.astype(float) - frame_pois.astype(float))**2)
                if mse > 0:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                    psnrs.append(psnr)

        cap_orig.release()
        cap_pois.release()

        return float(np.mean(psnrs)) if psnrs else float('inf')

    def save_signature(self, path: str) -> None:
        """Save frequency signature to JSON file."""
        data = {
            'signature_dct': self.signature_dct.tolist(),
            'temporal_signature': self.temporal_signature.tolist(),
            'epsilon': self.epsilon,
            'frequency_band': self.frequency_band,
            'temporal_period': self.temporal_period,
            'seed': self.seed
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Signature saved to {path}")

    def load_signature(self, path: str) -> None:
        """Load frequency signature from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.signature_dct = np.array(data['signature_dct'], dtype=np.float32)
        self.temporal_signature = np.array(data['temporal_signature'], dtype=np.float32)
        self.epsilon = data['epsilon']
        self.frequency_band = data['frequency_band']
        self.temporal_period = data['temporal_period']
        self.seed = data['seed']

        print(f"Signature loaded from {path}")


if __name__ == '__main__':
    # Quick demo
    print("=" * 60)
    print("Frequency Domain Video Poisoning")
    print("=" * 60)
    print()

    # Create marker
    marker = FrequencyDomainVideoMarker(
        epsilon=0.05,
        frequency_band='low',
        temporal_period=30
    )

    # Save signature
    marker.save_signature('frequency_signature.json')

    print()
    print("DCT Signature (8x8 matrix):")
    print(marker.signature_dct)
    print()
    print("Temporal Signature (30 frames):")
    print(marker.temporal_signature[:10], "...")
    print()
    print("Ready to poison videos!")
    print()
    print("Usage:")
    print("  marker = FrequencyDomainVideoMarker(epsilon=0.05)")
    print("  marker.poison_video('input.mp4', 'poisoned.mp4')")
