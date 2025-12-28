#!/usr/bin/env python3
"""
Debug: Why does our differentiable codec not match real H.264?

Test hypothesis: Real H.264 does more than just DCT + quantization:
1. Chroma subsampling (4:2:0)
2. Deblocking filter
3. Motion compensation (B/P frames)
4. Different quantization matrix
"""

import sys
import os
import cv2
import numpy as np
import subprocess
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from differentiable_codec import DifferentiableH264


def create_test_frame():
    """Create simple test frame (solid color + gradient)."""
    frame = np.ones((224, 224, 3), dtype=np.uint8) * 128

    # Add gradient
    for y in range(224):
        intensity = int(100 + 50 * y / 224)
        frame[y, :] = (intensity, intensity, intensity)

    return frame


def compress_with_differentiable_codec(frame: np.ndarray) -> np.ndarray:
    """Compress with our differentiable codec."""
    # Convert to tensor
    frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float()  # (1, 3, H, W)

    # Compress
    codec = DifferentiableH264(quality_factor=28, temperature=0.1)
    codec.eval()  # Use hard quantization

    with torch.no_grad():
        compressed_tensor = codec(frame_tensor)

    # Convert back
    compressed = compressed_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    return compressed


def compress_with_real_h264(frame: np.ndarray, output_path: str = '/tmp/test_frame.mp4'):
    """Compress with real H.264 (ffmpeg)."""
    # Save frame as video (1 frame)
    input_path = '/tmp/test_frame_input.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(input_path, fourcc, 30, (224, 224))
    out.write(frame)
    out.release()

    # Compress with H.264 CRF 28
    subprocess.run(['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', '28',
                   '-preset', 'medium', '-y', output_path],
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read back
    cap = cv2.VideoCapture(output_path)
    ret, compressed = cap.read()
    cap.release()

    return compressed if ret else None


def compare_dct_coefficients(frame1: np.ndarray, frame2: np.ndarray, label1: str, label2: str):
    """Compare DCT coefficients of two frames."""
    # Extract Y channel
    y1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    y2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

    # Extract center 8x8 block
    block1 = y1[112:120, 112:120]
    block2 = y2[112:120, 112:120]

    # DCT
    dct1 = cv2.dct(block1)
    dct2 = cv2.dct(block2)

    print(f"\n{label1} DCT coefficients (center block):")
    print(dct1)
    print(f"\n{label2} DCT coefficients (center block):")
    print(dct2)
    print(f"\nDifference ({label2} - {label1}):")
    print(dct2 - dct1)

    # Quantization check
    q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    # Scale for CRF 28
    scale = 200.0 - 2 * 28
    q_matrix_crf28 = q_matrix * scale / 50.0
    q_matrix_crf28 = np.maximum(q_matrix_crf28, 1.0)

    print(f"\nQuantization matrix (CRF 28):")
    print(q_matrix_crf28)

    # Check which coefficients are zeroed
    print(f"\n{label1} DCT / Q (should be quantized indices):")
    print(np.round(dct1 / q_matrix_crf28))

    print(f"\n{label2} DCT / Q:")
    print(np.round(dct2 / q_matrix_crf28))


def main():
    print("=" * 80)
    print("CODEC MISMATCH DEBUGGING")
    print("=" * 80)
    print()

    # Create test frame
    frame = create_test_frame()
    print("Test frame: 224x224 gradient (100-150 intensity)")
    print()

    # Compress with differentiable codec
    print("Compressing with differentiable codec...")
    diff_compressed = compress_with_differentiable_codec(frame)
    print("✓ Done")
    print()

    # Compress with real H.264
    print("Compressing with real H.264 (ffmpeg)...")
    real_compressed = compress_with_real_h264(frame)
    print("✓ Done")
    print()

    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # PSNR
    mse_diff = np.mean((frame.astype(float) - diff_compressed.astype(float))**2)
    psnr_diff = 20 * np.log10(255.0 / np.sqrt(mse_diff)) if mse_diff > 0 else float('inf')

    mse_real = np.mean((frame.astype(float) - real_compressed.astype(float))**2)
    psnr_real = 20 * np.log10(255.0 / np.sqrt(mse_real)) if mse_real > 0 else float('inf')

    print(f"\nPSNR (Differentiable): {psnr_diff:.2f} dB")
    print(f"PSNR (Real H.264):     {psnr_real:.2f} dB")

    # DCT comparison
    compare_dct_coefficients(frame, diff_compressed, "Original", "Diff Codec")
    compare_dct_coefficients(frame, real_compressed, "Original", "Real H.264")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    if abs(psnr_diff - psnr_real) < 3:
        print("✓ PSNRs are similar - compression strength matches")
    else:
        print(f"⚠️  PSNRs differ by {abs(psnr_diff - psnr_real):.2f} dB")

    # Check DCT difference
    diff_diff = np.mean(np.abs(diff_compressed.astype(float) - frame.astype(float)))
    real_diff = np.mean(np.abs(real_compressed.astype(float) - frame.astype(float)))

    print(f"\nMean pixel difference (Diff Codec): {diff_diff:.2f}")
    print(f"Mean pixel difference (Real H.264):  {real_diff:.2f}")


if __name__ == '__main__':
    main()
