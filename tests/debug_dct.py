#!/usr/bin/env python3
"""Debug: What are we actually extracting from videos?"""

import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker


def create_simple_video(path: str):
    """Create very simple video - solid gray."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30, (224, 224))

    for _ in range(60):
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Solid gray
        out.write(frame)

    out.release()


# Create simple video
clean_path = '/tmp/debug_clean.mp4'
poisoned_path = '/tmp/debug_poisoned.mp4'

create_simple_video(clean_path)

# Poison it
marker = FrequencyDomainVideoMarker(epsilon=0.1, frequency_band='low')  # Higher epsilon
print("Signature DCT pattern (what we're embedding):")
print(marker.signature_dct)
print()

marker.poison_video(clean_path, poisoned_path, verbose=False)

# Extract from both
clean_dct, _ = marker.extract_dct_signature(clean_path, num_frames=5)
poisoned_dct, _ = marker.extract_dct_signature(poisoned_path, num_frames=5)

print("Extracted from CLEAN video:")
print(clean_dct)
print()

print("Extracted from POISONED video:")
print(poisoned_dct)
print()

print("Difference:")
print(poisoned_dct - clean_dct)
print()

# Check if signature is there
low_freq_clean = clean_dct[0:3, 0:3].flatten()
low_freq_poisoned = poisoned_dct[0:3, 0:3].flatten()

correlation = np.dot(
    low_freq_clean / (np.linalg.norm(low_freq_clean) + 1e-8),
    low_freq_poisoned / (np.linalg.norm(low_freq_poisoned) + 1e-8)
)

print(f"Correlation between clean and poisoned low-freq: {correlation:.4f}")
print("(Should be < 1.0 if signature changed them)")
