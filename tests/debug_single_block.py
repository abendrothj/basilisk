#!/usr/bin/env python3
"""Debug: Check a single 8x8 block before and after poisoning."""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'poison-core'))
from frequency_poison import FrequencyDomainVideoMarker

# Create simple frame - solid gray
frame = np.ones((224, 224, 3), dtype=np.uint8) * 128

# Convert to YCbCr
frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
y_channel_clean = frame_ycbcr[:, :, 0].astype(np.float32)

# Extract clean block (center)
clean_block = y_channel_clean[112:120, 112:120]
print("Clean block (spatial):")
print(clean_block)
print()

# DCT of clean block
clean_dct = cv2.dct(clean_block)
print("Clean block (DCT):")
print(clean_dct)
print()

# Now poison it
marker = FrequencyDomainVideoMarker(epsilon=0.1, frequency_band='low')
print("Signature we're adding (epsilon=0.1, temporal_weight=1.0):")
perturbation = 0.1 * 1.0 * marker.signature_dct * 255.0
print(perturbation)
print()

# Add perturbation
poisoned_dct = clean_dct + perturbation
print("Poisoned block (DCT):")
print(poisoned_dct)
print()

# Inverse DCT
poisoned_block = cv2.idct(poisoned_dct)
print("Poisoned block (spatial, after iDCT):")
print(poisoned_block)
print()

# Re-extract DCT from poisoned spatial
reextracted_dct = cv2.dct(poisoned_block)
print("Re-extracted DCT from poisoned spatial:")
print(reextracted_dct)
print()

print("Difference (poisoned - clean DCT):")
print(reextracted_dct - clean_dct)
