#!/usr/bin/env python3
"""
Frequency Domain Signature Detector

CORRECT implementation: Extracts AC coefficient patterns, not absolute values.
"""

import cv2
import numpy as np
import json
from typing import Tuple


class FrequencySignatureDetector:
    """
    Detects frequency domain signatures in videos.

    Key insight: The signature is in the AC coefficient PATTERN, not absolute values.
    We need to extract the pattern of AC coefficients and correlate with our signature.
    """

    def __init__(self, signature_path: str):
        """Load signature from JSON."""
        with open(signature_path, 'r') as f:
            data = json.load(f)

        self.signature_dct = np.array(data['signature_dct'], dtype=np.float32)
        self.temporal_signature = np.array(data['temporal_signature'], dtype=np.float32)
        self.epsilon = data['epsilon']
        self.frequency_band = data['frequency_band']
        self.temporal_period = data['temporal_period']

        # Extract AC coefficients from signature (ignore DC at [0,0])
        self.signature_ac = self.signature_dct.copy()
        self.signature_ac[0, 0] = 0  # Zero out DC
        self.signature_ac_norm = self.signature_ac / (np.linalg.norm(self.signature_ac) + 1e-8)

    def detect_in_video(
        self,
        video_path: str,
        num_frames: int = 30,
        num_blocks_per_frame: int = 100
    ) -> Tuple[float, dict]:
        """
        Detect signature in a video.

        Returns:
            (detection_score, debug_info)

        Detection score > 0.3: Likely poisoned
        Detection score < 0.1: Likely clean
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return 0.0, {'error': 'Could not read video'}

        # Sample frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        all_ac_patterns = []  # Normalized AC patterns from all blocks

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

            # Extract AC patterns from random blocks
            num_blocks_h = y_padded.shape[0] // 8
            num_blocks_w = y_padded.shape[1] // 8
            total_blocks = num_blocks_h * num_blocks_w

            num_samples = min(num_blocks_per_frame, total_blocks)
            block_indices = np.random.choice(total_blocks, num_samples, replace=False)

            for block_idx in block_indices:
                block_i = (block_idx // num_blocks_w) * 8
                block_j = (block_idx % num_blocks_w) * 8

                block = y_padded[block_i:block_i+8, block_j:block_j+8]

                if block.shape == (8, 8):
                    dct_block = cv2.dct(block)

                    # Extract AC coefficients (zero out DC)
                    ac_coeffs = dct_block.copy()
                    ac_coeffs[0, 0] = 0

                    # Normalize
                    ac_norm = ac_coeffs / (np.linalg.norm(ac_coeffs) + 1e-8)

                    all_ac_patterns.append(ac_norm)

        cap.release()

        if not all_ac_patterns:
            return 0.0, {'error': 'No blocks extracted'}

        # Compute correlation of each AC pattern with signature
        correlations = []
        for ac_pattern in all_ac_patterns:
            corr = np.dot(ac_pattern.flatten(), self.signature_ac_norm.flatten())
            correlations.append(corr)

        # Detection score: mean absolute correlation
        # (signature could be negated due to temporal modulation)
        detection_score = float(np.mean(np.abs(correlations)))

        debug_info = {
            'num_blocks': len(all_ac_patterns),
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'max_correlation': float(np.max(np.abs(correlations))),
            'positive_ratio': float(np.sum(np.array(correlations) > 0) / len(correlations))
        }

        return detection_score, debug_info


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python frequency_detector.py <signature.json> <video.mp4>")
        sys.exit(1)

    detector = FrequencySignatureDetector(sys.argv[1])
    score, info = detector.detect_in_video(sys.argv[2])

    print(f"Detection score: {score:.4f}")
    print(f"Details: {info}")

    if score > 0.3:
        print("✅ POISONED")
    elif score > 0.15:
        print("⚠️  SUSPICIOUS")
    else:
        print("❌ CLEAN")
