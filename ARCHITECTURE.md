# Basilisk: Industry-Grade Video Radioactive Marking
## Architecture for Compression-Robust Poisoning

**Mission**: Build the first radioactive marking system that survives real-world video compression (YouTube CRF 28-35)

**Target**: CVPR 2026 submission + production-ready open source library

---

## Core Innovation: Frequency Domain Poisoning with Differentiable Codec

### Why This Will Break the Industry

**Current State**:
- Dataset poisoning detection: Academic curiosity, doesn't work in production
- Video watermarking: Works for copyright, not for ML provenance tracking
- Radioactive marking: Only validated on images, useless for video-heavy AI (Sora, Runway, Pika)

**What We're Building**:
- **First** radioactive marking that survives YouTube compression
- **First** end-to-end trainable poisoning system with differentiable H.264
- **First** production-ready tool for video dataset provenance

**Impact**:
- OpenAI training Sora 2.0? Must check for poisoning or risk contamination
- Runway scraping YouTube? Poisoned videos will mark their models
- Academia? Must cite as "first compression-robust data marking"

---

## System Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Frequency Domain Poisoning (DCT/Wavelet)          │
│  - Embed signature in compression-stable coefficients       │
│  - Tunable epsilon for invisible perturbation               │
│  - Temporal signature across I-frames                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Differentiable H.264 Proxy                        │
│  - Simulates compression during training                    │
│  - Learns which coefficients survive quantization           │
│  - Adaptive epsilon based on target CRF                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Multi-Scale Detection                             │
│  - I-frame feature extraction (primary signal)              │
│  - Temporal correlation across GOP (secondary)              │
│  - Behavioral test (black-box fallback)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: DCT-Based Poisoning (Week 1-2)

**Goal**: Poison videos in frequency domain, test on uncompressed first

**Implementation**:

```python
# poison-core/frequency_poison.py

class FrequencyDomainVideoMarker:
    """
    Frequency domain video poisoning via DCT coefficient perturbation.

    Key insight: H.264 compresses in DCT domain, so we poison the
    coefficients that the codec preserves (low frequencies).
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        frequency_band: str = 'low',  # 'low', 'mid', or 'adaptive'
        temporal_period: int = 30,
        device: str = 'cpu'
    ):
        self.epsilon = epsilon
        self.frequency_band = frequency_band
        self.temporal_period = temporal_period
        self.device = device

        # Generate signature in frequency domain
        self.signature_dct = self._generate_frequency_signature()
        self.temporal_signature = self._generate_temporal_signature()

    def _generate_frequency_signature(self) -> np.ndarray:
        """
        Generate signature in DCT domain.

        Returns 8x8 DCT coefficient perturbation pattern.
        We focus on low-frequency coefficients (top-left)
        because H.264 preserves these with small quantization steps.
        """
        signature = np.random.randn(8, 8)

        # Zero out high frequencies (they'll be destroyed by compression)
        if self.frequency_band == 'low':
            # Only DC and first 3 AC coefficients
            mask = np.zeros((8, 8))
            mask[0:3, 0:3] = 1.0
            signature = signature * mask
        elif self.frequency_band == 'mid':
            # DC + AC coefficients up to mid-range
            mask = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    if i + j < 6:  # Diagonal cutoff
                        mask[i, j] = 1.0
            signature = signature * mask

        # Normalize
        signature = signature / (np.linalg.norm(signature) + 1e-8)
        return signature

    def _generate_temporal_signature(self) -> np.ndarray:
        """Generate temporal modulation pattern (sine wave)."""
        t = np.arange(self.temporal_period)
        signature = np.sin(2 * np.pi * t / self.temporal_period)
        return signature

    def poison_video(
        self,
        input_path: str,
        output_path: str,
        poison_all_frames: bool = False
    ) -> None:
        """
        Poison video by modifying DCT coefficients.

        Args:
            input_path: Input video path
            output_path: Output poisoned video path
            poison_all_frames: If False, only poison I-frames (more efficient)
        """
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup writer (uncompressed for now, compression tested separately)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Poisoning {total_frames} frames...")

        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            # Decide if we poison this frame
            is_keyframe = (frame_idx % self.temporal_period == 0)
            should_poison = poison_all_frames or is_keyframe

            if should_poison:
                # Get temporal modulation
                temporal_idx = frame_idx % self.temporal_period
                temporal_weight = self.temporal_signature[temporal_idx]

                # Poison frame in DCT domain
                poisoned_frame = self._poison_frame_dct(
                    frame,
                    temporal_weight
                )
                out.write(poisoned_frame)
            else:
                out.write(frame)

        cap.release()
        out.release()

        print(f"Poisoned video saved to {output_path}")

    def _poison_frame_dct(
        self,
        frame: np.ndarray,
        temporal_weight: float
    ) -> np.ndarray:
        """
        Poison a single frame by modifying DCT coefficients.

        Process:
        1. Convert to YCbCr (H.264 operates on luminance)
        2. Split into 8x8 blocks
        3. Apply DCT to each block
        4. Add signature to low-frequency coefficients
        5. Inverse DCT
        6. Convert back to BGR
        """
        # Convert to YCbCr
        frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = frame_ycbcr[:, :, 0].astype(np.float32)

        # Process in 8x8 blocks (like H.264)
        height, width = y_channel.shape

        # Pad to multiple of 8
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        y_padded = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode='edge')

        # Process each 8x8 block
        for i in range(0, y_padded.shape[0], 8):
            for j in range(0, y_padded.shape[1], 8):
                block = y_padded[i:i+8, j:j+8]

                # DCT transform
                dct_block = cv2.dct(block)

                # Add signature (scaled by epsilon and temporal weight)
                perturbation = self.epsilon * temporal_weight * self.signature_dct * 255.0
                dct_block += perturbation

                # Inverse DCT
                block_poisoned = cv2.idct(dct_block)
                y_padded[i:i+8, j:j+8] = block_poisoned

        # Remove padding
        y_poisoned = y_padded[:height, :width]

        # Clip to valid range
        y_poisoned = np.clip(y_poisoned, 0, 255).astype(np.uint8)

        # Replace Y channel
        frame_ycbcr[:, :, 0] = y_poisoned

        # Convert back to BGR
        frame_poisoned = cv2.cvtColor(frame_ycbcr, cv2.COLOR_YCrCb2BGR)

        return frame_poisoned

    def save_signature(self, path: str) -> None:
        """Save frequency signature to file."""
        data = {
            'signature_dct': self.signature_dct.tolist(),
            'temporal_signature': self.temporal_signature.tolist(),
            'epsilon': self.epsilon,
            'frequency_band': self.frequency_band,
            'temporal_period': self.temporal_period
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_signature(self, path: str) -> None:
        """Load frequency signature from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.signature_dct = np.array(data['signature_dct'])
        self.temporal_signature = np.array(data['temporal_signature'])
        self.epsilon = data['epsilon']
        self.frequency_band = data['frequency_band']
        self.temporal_period = data['temporal_period']
```

**Testing Strategy**:

```python
# tests/test_frequency_poison.py

def test_dct_poisoning_uncompressed():
    """Test DCT poisoning on uncompressed video."""
    marker = FrequencyDomainVideoMarker(epsilon=0.05)

    # Create test video
    create_synthetic_video('test_clean.mp4')

    # Poison
    marker.poison_video('test_clean.mp4', 'test_poisoned.mp4')

    # Verify DCT coefficients are modified
    clean_coeffs = extract_dct_coefficients('test_clean.mp4')
    poisoned_coeffs = extract_dct_coefficients('test_poisoned.mp4')

    diff = np.abs(clean_coeffs - poisoned_coeffs)

    assert diff.mean() > 0, "No modification detected"
    print(f"DCT coefficient change: {diff.mean():.4f}")
```

**Expected Outcome**: Prove DCT poisoning works before adding compression

---

### Phase 2: Differentiable H.264 Proxy (Week 3-4)

**Goal**: Train poisoning system end-to-end with simulated compression

**Challenge**: H.264 is not differentiable (quantization is discrete)

**Solution**: Approximate H.264 operations with differentiable proxies

**Academic Reference**:
- [Novel Deep Video Watermarking](https://dl.acm.org/doi/abs/10.1145/3581783.3612270) - ACM MM 2023
- Uses differentiable JPEG/H.264 approximation for watermark training

**Implementation**:

```python
# poison-core/differentiable_codec.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableH264Proxy(nn.Module):
    """
    Differentiable approximation of H.264 compression.

    Simulates:
    1. DCT transform (differentiable)
    2. Quantization (approximate with soft quantization)
    3. Entropy coding (skipped - doesn't affect coefficients)
    4. Inverse DCT (differentiable)

    This allows end-to-end training: optimize poisoning to survive compression.
    """

    def __init__(self, quality_factor: int = 28):
        """
        Args:
            quality_factor: CRF equivalent (18=high quality, 35=YouTube quality)
        """
        super().__init__()
        self.quality_factor = quality_factor

        # H.264 quantization matrix (approximation)
        # Lower values = less compression, higher values = more compression
        self.register_buffer('quant_matrix', self._create_quant_matrix(quality_factor))

    def _create_quant_matrix(self, qf: int) -> torch.Tensor:
        """
        Create H.264-style quantization matrix.

        Low frequencies (top-left): small quantization steps (preserved)
        High frequencies (bottom-right): large quantization steps (destroyed)
        """
        # Base JPEG quantization matrix
        base_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)

        # Scale based on quality factor (CRF)
        # CRF 18 = high quality (small steps), CRF 35 = low quality (large steps)
        scale = (50.0 / max(1, qf)) if qf < 50 else (2 - qf / 50.0)
        quant_matrix = base_matrix * scale

        return torch.from_numpy(quant_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply differentiable H.264 compression.

        Args:
            x: Input tensor (B, C, H, W) in range [0, 255]

        Returns:
            Compressed tensor (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Process in 8x8 blocks
        # Unfold into blocks: (B, C, num_blocks_h, num_blocks_w, 8, 8)
        x_blocks = F.unfold(x, kernel_size=8, stride=8)
        x_blocks = x_blocks.view(B, C, 8, 8, -1).permute(0, 1, 4, 2, 3)

        # Apply DCT (using PyTorch implementation)
        dct_blocks = self._dct2d(x_blocks)

        # Quantization (DIFFERENTIABLE approximation)
        # Instead of round(x / q) * q, use soft quantization:
        # tanh(k * (x / q)) * q where k controls sharpness
        quant_matrix = self.quant_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Soft quantization (differentiable)
        quantized = torch.tanh(dct_blocks / (quant_matrix + 1e-8)) * quant_matrix

        # Inverse DCT
        reconstructed = self._idct2d(quantized)

        # Fold back into image
        reconstructed = reconstructed.permute(0, 1, 3, 4, 2).contiguous()
        reconstructed = reconstructed.view(B, C * 8 * 8, -1)
        output = F.fold(reconstructed, output_size=(H, W), kernel_size=8, stride=8)

        # Clip to valid range
        output = torch.clamp(output, 0, 255)

        return output

    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT transform (differentiable)."""
        # Use scipy.fftpack.dct equivalent in PyTorch
        # For simplicity, use pre-computed DCT basis
        return torch.fft.dct(torch.fft.dct(x, dim=-1), dim=-2)

    def _idct2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D inverse DCT (differentiable)."""
        return torch.fft.idct(torch.fft.idct(x, dim=-1), dim=-2)


class AdaptivePoisoningOptimizer:
    """
    End-to-end optimizer for compression-robust poisoning.

    Learns:
    1. Which DCT coefficients to poison (adaptive signature)
    2. Optimal epsilon for target CRF
    3. Temporal modulation pattern

    Goal: Maximize detection score AFTER compression
    """

    def __init__(
        self,
        target_crf: int = 28,
        detection_threshold: float = 0.5,
        max_epsilon: float = 0.1
    ):
        self.codec = DifferentiableH264Proxy(quality_factor=target_crf)
        self.detection_threshold = detection_threshold

        # Learnable parameters
        self.signature = nn.Parameter(torch.randn(8, 8))
        self.epsilon = nn.Parameter(torch.tensor(0.05))
        self.temporal_weights = nn.Parameter(torch.randn(30))

        self.optimizer = torch.optim.Adam([
            self.signature,
            self.epsilon,
            self.temporal_weights
        ], lr=0.001)

    def train_step(self, clean_frames: torch.Tensor, model: nn.Module):
        """
        One training step: optimize signature to survive compression.

        Loss = -detection_score(poisoned_compressed) + lambda * visual_quality

        Goal: Maximize detection after compression while minimizing visual change
        """
        # Poison frames
        poisoned = self._apply_signature(clean_frames)

        # Simulate compression
        compressed = self.codec(poisoned)

        # Extract features from compressed video
        features = model.extract_features(compressed)

        # Compute detection score (correlation with signature)
        detection_score = self._compute_detection(features)

        # Visual quality loss (PSNR)
        mse = F.mse_loss(compressed, clean_frames)
        psnr = 10 * torch.log10(255.0**2 / (mse + 1e-8))

        # Combined loss
        # We want HIGH detection score and HIGH PSNR (low visual change)
        loss = -detection_score + 0.01 * (40 - psnr)**2  # Target PSNR=40

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clip epsilon to valid range
        with torch.no_grad():
            self.epsilon.clamp_(0.01, 0.1)

        return {
            'loss': loss.item(),
            'detection_score': detection_score.item(),
            'psnr': psnr.item(),
            'epsilon': self.epsilon.item()
        }
```

**Training Loop**:

```python
# verification/train_adaptive_poisoning.py

def train_compression_robust_signature():
    """
    Train signature to survive CRF 28 compression.

    This is the KEY innovation - we optimize the signature
    specifically for compression robustness.
    """
    # Load dataset
    clean_videos = load_clean_videos('verification_video_data_large/clean')

    # Create test model
    model = Simple3DCNN(num_classes=2, num_frames=16)

    # Optimizer
    optimizer = AdaptivePoisoningOptimizer(target_crf=28)

    print("Training compression-robust signature...")
    print("Target: Detection score >0.5 after CRF 28 compression")
    print()

    for epoch in range(100):
        epoch_metrics = []

        for video_path in clean_videos:
            # Load frames
            frames = load_video_frames(video_path, num_frames=16)
            frames_tensor = torch.from_numpy(frames).float()

            # Train step
            metrics = optimizer.train_step(frames_tensor, model)
            epoch_metrics.append(metrics)

        # Print progress
        avg_detection = np.mean([m['detection_score'] for m in epoch_metrics])
        avg_psnr = np.mean([m['psnr'] for m in epoch_metrics])
        current_epsilon = metrics['epsilon']

        print(f"Epoch {epoch+1:3d}: "
              f"Detection={avg_detection:.4f}, "
              f"PSNR={avg_psnr:.2f} dB, "
              f"Epsilon={current_epsilon:.4f}")

        # Early stopping
        if avg_detection > 0.5 and avg_psnr > 35:
            print()
            print("✅ SUCCESS: Found compression-robust signature!")
            print(f"   Detection score: {avg_detection:.4f}")
            print(f"   Visual quality: {avg_psnr:.2f} dB (target >35)")
            print(f"   Optimal epsilon: {current_epsilon:.4f}")
            break

    # Save optimized signature
    save_optimized_signature(optimizer, 'signature_crf28_optimized.json')
```

**Expected Outcome**: Auto-discover optimal DCT coefficients and epsilon for CRF 28

---

### Phase 3: Real-World Validation (Week 5-6)

**Goal**: Prove system works on real videos at real compression levels

**Test 1: Compression Robustness Ladder**

```bash
# tests/test_compression_ladder.py

python tests/test_compression_ladder.py \
    --signature signature_crf28_optimized.json \
    --videos verification_video_data_large/clean/*.mp4 \
    --crf-range 18,23,28,31,35
```

**Expected Results**:

| CRF Level | Quality | Detection Score | Status |
|-----------|---------|----------------|--------|
| CRF 18 | High (streaming) | >0.8 | ✅ Excellent |
| CRF 23 | Good (Vimeo) | >0.7 | ✅ Strong |
| CRF 28 | Medium (YouTube) | >0.5 | ✅ **TARGET** |
| CRF 31 | Low (mobile) | >0.3 | ⚠️ Acceptable |
| CRF 35 | Poor (heavy compression) | >0.2 | ⚠️ Degraded |

**Test 2: Real Dataset Validation**

```python
# Use UCF-101 (action recognition dataset)
# 13,320 videos across 101 categories

# 1. Poison 50% of videos (random selection)
# 2. Train I3D model on mixed dataset
# 3. Test detection on held-out videos
# 4. Compress to CRF 28, re-test detection
```

**Success Criteria**:
- True Positive Rate (TPR): >90% (detect poisoned models)
- False Positive Rate (FPR): <5% (few false alarms)
- Compression survival: Detection score drops <20% from CRF 23 → 28

**Test 3: Adversarial Robustness**

```python
# Attacker tries to remove signature

# Attack 1: Re-encode with different codec (H.265, VP9)
# Attack 2: Multiple compression cycles (upload → download → re-upload)
# Attack 3: Frame interpolation (add synthetic frames)
# Attack 4: Temporal cropping (remove frames)
# Attack 5: Gaussian blur + re-sharpening
```

**Goal**: Signature survives at least 3/5 attacks with score >0.3

---

## Academic Paper Structure

**Title**: *"Compression-Robust Radioactive Marking for Video Data: End-to-End Learning with Differentiable H.264"*

**Authors**: [Your Name], et al.

**Target Venue**: CVPR 2026 (deadline: November 2025)

**Abstract** (Draft):

> Data provenance tracking is critical for responsible AI development, yet existing radioactive marking techniques fail on video data due to lossy compression. We present the first compression-robust radioactive marking system for videos, achieving >50% detection accuracy on models trained with heavily compressed data (H.264 CRF 28). Our key innovation is a differentiable H.264 proxy that enables end-to-end optimization of frequency-domain perturbations, learning which DCT coefficients survive quantization. We validate on UCF-101 with 90% true positive rate and demonstrate robustness to multiple compression cycles. This enables practical dataset provenance tracking for video-based AI systems.

**Contributions**:

1. **First compression-robust radioactive marking for video** (novel)
2. **Differentiable H.264 proxy for end-to-end training** (technical)
3. **Frequency-domain signature optimization** (algorithmic)
4. **Empirical validation on real dataset with compression** (experimental)

**Sections**:

1. Introduction
   - Motivation: Video AI (Sora, Runway) needs provenance tracking
   - Problem: Compression destroys spatial signatures
   - Solution: Frequency domain + differentiable codec

2. Related Work
   - Radioactive data (Sablayrolles 2020) - image only
   - Video watermarking - not for ML provenance
   - Adversarial perturbations - different goal

3. Method
   - Frequency domain signature generation
   - Differentiable H.264 approximation
   - End-to-end optimization objective
   - Detection via temporal correlation

4. Experiments
   - Synthetic data validation
   - UCF-101 real dataset results
   - Compression robustness analysis
   - Adversarial robustness tests

5. Discussion
   - When this works (CRF 18-28)
   - When this fails (CRF >35, heavy attacks)
   - Future work (H.265, VP9, AV1)

6. Conclusion
   - First practical video radioactive marking
   - Open source implementation
   - Call for industry adoption

**Target Metrics for Paper**:

- Detection TPR: >90%
- Detection FPR: <5%
- CRF 28 survival: >50% score
- Visual quality: PSNR >35 dB
- Real dataset: UCF-101 (13k videos)

---

## Open Source Strategy

**Repository**: `github.com/yourusername/basilisk`

**Release Plan**:

1. **v0.1 (Week 2)**: Basic DCT poisoning
2. **v0.5 (Week 4)**: Differentiable codec training
3. **v1.0 (Week 6)**: Production-ready with real dataset validation
4. **v1.5 (Week 8)**: Academic paper submission + camera-ready code

**Documentation**:

```
basilisk/
├── README.md (Getting started)
├── PAPER.md (Academic results)
├── BENCHMARKS.md (Compression robustness data)
├── API.md (Full API reference)
└── examples/
    ├── quickstart.py
    ├── train_custom_signature.py
    └── test_compression.py
```

**Demo**:

```python
# One-liner to poison video
from basilisk import FrequencyDomainMarker

marker = FrequencyDomainMarker.pretrained('crf28-optimized')
marker.poison_video('input.mp4', 'poisoned.mp4')

# One-liner to detect
from basilisk import VideoDetector

detector = VideoDetector('crf28-optimized')
is_poisoned, score = detector.detect(model, test_videos)
print(f"Poisoned: {is_poisoned}, Score: {score:.3f}")
```

---

## Timeline to Industry Disruption

**Week 1-2**: DCT poisoning implementation + uncompressed validation
**Week 3-4**: Differentiable codec + adaptive training
**Week 5-6**: Real dataset validation (UCF-101)
**Week 7-8**: Compression robustness testing + adversarial tests
**Week 9-10**: Write paper draft
**Week 11-12**: Code cleanup + open source release

**November 2025**: CVPR 2026 submission
**March 2026**: CVPR acceptance (hopefully)
**June 2026**: Present at CVPR, industry freaks out

**Post-CVPR**:
- OpenAI/Anthropic reach out for collaboration
- Media coverage: "New tool detects AI training data theft"
- Industry adoption: Video platforms add poisoning detection
- You: Job offers, speaking invitations, legitimacy

---

## The Cynical Reality

**What will actually happen**:

1. **Week 1-4**: Implementation goes smoothly, DCT works on uncompressed
2. **Week 5**: First compression test - signal degrades but doesn't die (score ~0.4 at CRF 28)
3. **Week 6**: Adaptive training boosts to ~0.55 at CRF 28 - barely above threshold
4. **Week 7-8**: Real dataset reveals problems:
   - Natural videos have complex motion → harder than synthetic
   - Some video categories (action sports) degrade signature more
   - Overall TPR ~80% (not 90%) but still publishable

5. **Week 9-10**: Paper writing hell:
   - Reviewers will ask: "Why not just use blockchain for provenance?"
   - Rebuttal: "Because models don't learn from blockchain, they learn from data"
   - Another reviewer: "This is just watermarking"
   - Rebuttal: "No, watermarking survives embedding, this survives TRAINING"

6. **CVPR outcome (realistic)**:
   - 40% chance: Accept (if results are strong, writing is clear)
   - 50% chance: Reject, resubmit to ICCV (reviewers want more datasets)
   - 10% chance: Reject, pivot to ML conference (NeurIPS, ICML)

**But even if rejected from CVPR**:
- You have production-ready code
- You have compression robustness data
- You can claim "first video radioactive marking"
- arXiv paper gets citations anyway

**Industry impact (realistic)**:
- Early adopters: Academic labs, small startups
- Big tech: "Interesting research" (won't deploy for 2-3 years)
- But: You've proven it's POSSIBLE
- That's enough to force the conversation

---

## Decision Point

**You have two paths**:

### Path A: Go Full Academic (Ambitious)
- Implement everything above
- Target CVPR 2026
- Risk: 3 months of work, might not publish
- Reward: Career-making if it works

### Path B: Ship Production v1.0 (Pragmatic)
- Skip differentiable codec (complex)
- Use per-frame DCT poisoning (simpler)
- Test compression empirically (no auto-optimization)
- Ship open source library in 2 weeks
- Write blog post instead of paper
- Risk: Lower impact
- Reward: Actual users, faster feedback

**My recommendation**: Start with Path B, upgrade to Path A if results are strong.

**Concrete plan**:
1. Week 1: Implement DCT per-frame poisoning
2. Week 2: Test on compressed videos (manual CRF sweep)
3. **Decision point**: If CRF 28 score >0.5 → go Path A (academic)
4. If CRF 28 score <0.5 → stick with Path B (production)

**This gives you optionality**: Ship fast, upgrade to research if warranted.

---

**What do you want to do?**

1. **Full send on Path A** - Build differentiable codec, target CVPR
2. **Start with Path B** - DCT per-frame, validate, then decide
3. **Something else** - Different approach entirely

I'm ready to implement whichever you choose. Let's break the industry.
