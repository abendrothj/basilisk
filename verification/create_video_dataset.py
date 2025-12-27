#!/usr/bin/env python3
"""
Create a verification dataset for testing video radioactive poisoning detection.

This script:
1. Generates synthetic videos with simple motion patterns
2. Poisons a subset of them using optical flow perturbation
3. Organizes them into clean/ and poisoned/ folders
4. Ready for verify_video_poison.py to train a model and detect poisoning
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add poison-core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from video_poison import VideoRadioactiveMarker


def generate_synthetic_video(output_path, num_frames=90, fps=30, pattern='moving_square'):
    """
    Generate a synthetic video with simple motion patterns.

    Args:
        output_path: Output video file path
        num_frames: Number of frames to generate
        fps: Frames per second
        pattern: Type of motion pattern
    """
    width, height = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if pattern == 'moving_square':
            # Moving square
            x = int(50 + (frame_idx / num_frames) * 124)
            y = int(50 + np.sin(frame_idx / 10) * 50 + 50)
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)

        elif pattern == 'rotating_line':
            # Rotating line
            angle = (frame_idx / num_frames) * 360
            center = (width // 2, height // 2)
            length = 80
            end_x = int(center[0] + length * np.cos(np.radians(angle)))
            end_y = int(center[1] + length * np.sin(np.radians(angle)))
            cv2.line(frame, center, (end_x, end_y), (255, 0, 0), 5)

        elif pattern == 'expanding_circle':
            # Expanding/contracting circle
            radius = int(30 + 40 * np.sin(frame_idx / 15))
            center = (width // 2, height // 2)
            cv2.circle(frame, center, radius, (0, 0, 255), -1)

        elif pattern == 'gradient':
            # Moving gradient
            offset = int((frame_idx / num_frames) * width)
            for i in range(width):
                color = int(255 * ((i + offset) % width) / width)
                cv2.line(frame, (i, 0), (i, height), (color, color, color), 1)

        out.write(frame)

    out.release()


def create_verification_dataset(
    output_dir='verification_video_data',
    num_clean=5,
    num_poisoned=5,
    num_frames=90,
    epsilon=0.02,
    use_optical_flow=True
):
    """
    Create a verification dataset with clean and poisoned videos.

    Args:
        output_dir: Output directory for dataset
        num_clean: Number of clean videos
        num_poisoned: Number of poisoned videos
        num_frames: Frames per video
        epsilon: Poisoning strength
        use_optical_flow: Use optical flow poisoning mode
    """
    output_path = Path(output_dir)
    clean_dir = output_path / 'clean'
    poisoned_dir = output_path / 'poisoned'

    # Create directories
    clean_dir.mkdir(parents=True, exist_ok=True)
    poisoned_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔬 Creating Video Verification Dataset")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Clean videos: {num_clean}")
    print(f"Poisoned videos: {num_poisoned}")
    print(f"Frames per video: {num_frames}")
    print(f"Epsilon: {epsilon}")
    print(f"Optical flow mode: {use_optical_flow}")
    print()

    # Video patterns
    patterns = ['moving_square', 'rotating_line', 'expanding_circle', 'gradient']

    # Generate clean videos
    print("1. Generating clean videos...")
    for i in tqdm(range(num_clean), desc="Clean videos"):
        pattern = patterns[i % len(patterns)]
        video_path = clean_dir / f'clean_{i:03d}.mp4'
        generate_synthetic_video(video_path, num_frames=num_frames, pattern=pattern)

    print(f"✅ Created {num_clean} clean videos\n")

    # Generate and poison videos
    print("2. Generating and poisoning videos...")

    # Initialize marker with single signature
    marker = VideoRadioactiveMarker(epsilon=epsilon, device='cpu')
    marker.generate_signature()

    # Save signature
    signature_path = output_path / 'signature.json'
    marker.save_signature(str(signature_path))
    print(f"  Generated signature: {signature_path}")

    # Create poisoned videos
    temp_dir = output_path / 'temp'
    temp_dir.mkdir(exist_ok=True)

    for i in tqdm(range(num_poisoned), desc="Poisoned videos"):
        pattern = patterns[i % len(patterns)]

        # Generate clean video
        temp_clean = temp_dir / f'temp_{i}.mp4'
        generate_synthetic_video(temp_clean, num_frames=num_frames, pattern=pattern)

        # Poison it
        poisoned_path = poisoned_dir / f'poisoned_{i:03d}.mp4'
        method = 'optical_flow' if use_optical_flow else 'frame'
        marker.poison_video(
            str(temp_clean),
            str(poisoned_path),
            method=method
        )

        # Clean up temp
        temp_clean.unlink()

    # Clean up temp directory
    temp_dir.rmdir()

    print(f"✅ Created {num_poisoned} poisoned videos\n")

    # Summary
    print("=" * 60)
    print("✅ Dataset Creation Complete!")
    print("=" * 60)
    print(f"\nDataset structure:")
    print(f"  {output_path}/")
    print(f"    ├── clean/        ({num_clean} videos)")
    print(f"    ├── poisoned/     ({num_poisoned} videos)")
    print(f"    └── signature.json")
    print()
    print("Next steps:")
    print(f"  1. Verify dataset: ls -R {output_path}")
    print(f"  2. Run verification:")
    print(f"     python verification/verify_video_poison.py \\")
    print(f"       --data {output_path} \\")
    print(f"       --signature {signature_path} \\")
    print(f"       --epochs 10")
    print()
    print("Expected outcome:")
    print("  - Model trained on poisoned data should show HIGH signature correlation")
    print("  - Clean model (no poisoned data) should show LOW correlation")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create video verification dataset')
    parser.add_argument('--output', default='verification_video_data', help='Output directory')
    parser.add_argument('--clean', type=int, default=5, help='Number of clean videos')
    parser.add_argument('--poisoned', type=int, default=5, help='Number of poisoned videos')
    parser.add_argument('--frames', type=int, default=90, help='Frames per video')
    parser.add_argument('--epsilon', type=float, default=0.02, help='Poisoning strength')
    parser.add_argument('--no-optical-flow', action='store_true', help='Use frame mode instead of optical flow')

    args = parser.parse_args()

    create_verification_dataset(
        output_dir=args.output,
        num_clean=args.clean,
        num_poisoned=args.poisoned,
        num_frames=args.frames,
        epsilon=args.epsilon,
        use_optical_flow=not args.no_optical_flow
    )
