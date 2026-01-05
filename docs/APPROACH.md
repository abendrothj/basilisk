# Technical Approach & Architecture

## Perceptual Hash Robustness for Video (2025)

### Motivation
Traditional DCT-based or per-frame poisoning approaches fail under aggressive video compression (e.g., H.264 CRF 28) because codecs are designed to destroy imperceptible or high-frequency information. Our new approach leverages perceptual features that codecs must preserve for human viewing.

### Key Insight
Embed the signature in perceptual features (edges, textures, motion, saliency, color histograms) that are robust to compression. This allows the signature to survive even harsh quantization.

### Method Overview
1. Extract perceptual features from each frame (edges, textures, saliency, color histograms).
2. Compute a perceptual hash by projecting these features to a fixed-length binary vector.
3. (For poisoning) Optimize a perturbation so the video’s perceptual hash collides with a target signature, while maintaining high visual quality (SSIM > 0.95).
4. (For detection) After compression, extract features and compute the hash; if the Hamming distance to the signature is low, the video is detected as poisoned.

### Why This Works
- H.264 and similar codecs are perceptually driven: they preserve edges, textures, and motion for watchability.
- Embedding in these features means the signature survives compression, unlike DCT or pixel-based methods.

### Results
- Synthetic and public benchmark videos: Hamming distance after CRF 28 compression is typically 0–14/256 (very robust)
- Pure noise: Higher drift (expected, as noise is not preserved by codecs)

### Implementation
- See experiments/perceptual_hash.py and batch_hash_robustness.py for code and reproducibility.

---

This document explains how Project Sigil works under the hood, from mathematical foundations to system architecture.

---

## Table of Contents

1. [Core Concept: Radioactive Marking](#core-concept)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Architecture](#implementation-architecture)
4. [Phase-by-Phase Breakdown](#phase-breakdown)
5. [Security Analysis](#security-analysis)
6. [Performance Considerations](#performance-considerations)

---

<a name="core-concept"></a>
## 1. Core Concept: Radioactive Marking

### The Problem

Traditional watermarks fail against AI training:
- **Pixel watermarks:** Averaged away during training (thousands of images → one model)
- **Steganography:** Destroyed by augmentation (crops, rotations, compression)
- **Metadata:** Stripped by scrapers

### The Insight

Instead of marking pixels, mark the **feature representation** that the AI will learn.

**Analogy:** Imagine poisoning a tree's roots. No matter how the tree grows (training), the poison spreads to every branch (model weights).

### How Radioactive Marking Works

```
Input Image → Feature Extractor → Features (512-dim vector)
                                        ↓
                                   Push toward unique signature
                                        ↓
                                   Compute gradient
                                        ↓
                                   Perturb pixels slightly
                                        ↓
                                   Poisoned Image (looks identical)
```

**Key Properties:**
1. **Imperceptible:** Perturbation is < 1% of pixel values (ε = 0.01)
2. **Persistent:** Survives training because it's baked into what the model learns
3. **Detectable:** Can extract the signature from trained model weights
4. **Unique:** Cryptographically secure random signature (2^256 possibilities)

---

<a name="mathematical-foundation"></a>
## 2. Mathematical Foundation

### 2.1 Signature Generation

Generate a unique, unpredictable direction in feature space:

```python
seed = SecureRandom(256 bits)  # Cryptographically secure
rng = RandomState(seed)
signature = rng.randn(d)       # d = feature dimension (e.g., 512)
signature = signature / ||signature||  # Normalize to unit vector
```

**Why this works:** There are infinite directions in 512-dimensional space. Finding a specific direction without the seed is computationally infeasible.

### 2.2 Perturbation Computation

Given an image `x` and signature `s`:

1. **Extract features:**
   ```
   f(x) = FeatureExtractor(x)  # e.g., ResNet penultimate layer
   ```

2. **Define loss (how far features are from signature):**
   ```
   L = -⟨f(x), s⟩  # Negative dot product (we want to maximize alignment)
   ```

3. **Compute gradient:**
   ```
   ∇_x L = -∂f(x)/∂x · s  # How to change pixels to move features toward s
   ```

4. **Apply perturbation (FGSM-style):**
   ```
   x_poisoned = x + ε · sign(∇_x L)
   x_poisoned = clip(x_poisoned, 0, 1)  # Keep valid pixel range
   ```

**Parameter `ε` (epsilon):**
- Too small (< 0.005): Signature too weak, won't survive training
- Too large (> 0.05): Visible artifacts, image degradation
- Sweet spot: 0.01 - 0.02 (imperceptible but robust)

### 2.3 Detection Mathematics

Given a suspect model `M` and signature `s`:

1. **Extract features from clean test images:**
   ```
   features = [M(x_1), M(x_2), ..., M(x_n)]
   ```

2. **Measure alignment with signature:**
   ```
   correlation = mean([⟨M(x_i), s⟩ for all x_i])
   ```

3. **Statistical test:**
   ```
   if correlation > threshold:
       Model was trained on poisoned data
   ```

**Why this works:** A model trained on random images has correlation ≈ 0 with any direction. A model trained on images pushed toward `s` will internalize that direction.

**Statistical Power:**
- Random correlation: ≈ 0 ± 0.01
- Poisoned model: > 0.1 (10x stronger signal)
- False positive rate: < 0.001% with proper threshold

---

<a name="implementation-architecture"></a>
## 3. Implementation Architecture

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Project Sigil                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Sigil  │  │ verification │  │   web-ui     │     │
│  │               │  │              │  │              │     │
│  │ • Radioactive │  │ • Training   │  │ • React UI   │     │
│  │   Marker      │  │   scripts    │  │ • Drag/drop  │     │
│  │ • Detector    │  │ • Detection  │  │ • Download   │     │
│  │ • CLI         │  │   tests      │  │              │     │
│  └───────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                   ↓                  ↓            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PyTorch + ResNet-18                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

**Poisoning Flow:**
```
User uploads image.jpg
        ↓
Backend: Load image → Preprocess → Extract features
        ↓
Apply gradient perturbation (ε = 0.01)
        ↓
Save poisoned_image.jpg + signature.json
        ↓
User downloads both files
```

**Detection Flow:**
```
User provides: trained_model.pth + signature.json + test_images/
        ↓
Load signature vector from JSON
        ↓
Extract features from test images using model
        ↓
Compute correlation with signature
        ↓
Return: Is poisoned? (Yes/No) + Confidence score (0-1)
```

### 3.3 File Structure

```
sigil/
├── Sigil/
│   ├── radioactive_poison.py   # Core algorithm
│   ├── poison_cli.py            # Command-line interface
│   └── requirements.txt
│
├── verification/
│   ├── verify_poison.py         # Training + detection test
│   └── test_data/
│       ├── clean/               # 100 clean images
│       └── poisoned/            # 100 poisoned images
│
├── web-ui/
│   ├── pages/
│   │   ├── index.tsx            # Main upload page
│   │   └── api/
│   │       └── poison.ts        # Backend API
│   └── components/
│       └── Uploader.tsx         # Drag-drop UI
│
├── docs/
│   ├── RESEARCH.md              # Academic citations
│   ├── APPROACH.md              # This file
│   └── API.md                   # API documentation
│
└── README.md
```

---

<a name="phase-breakdown"></a>
## 4. Phase-by-Phase Breakdown

### Phase 1: Images (Weeks 1-6) ✓

**Goals:**
- [x] Implement radioactive marking
- [x] Create CLI tool
- [x] Build verification environment
- [ ] Web UI prototype

**Technical Challenges Solved:**
1. **Gradient computation:** Used PyTorch autograd for efficient ∇_x computation
2. **Feature extraction:** Leveraged pre-trained ResNet-18 (ImageNet weights)
3. **Signature storage:** JSON format with cryptographic seed

**Deliverables:**
- `poison_cli.py`: Single-image and batch poisoning
- `verify_poison.py`: Train a model and detect signature
- Proof of concept: Local verification that signature survives training

---

### Phase 2: Video (Weeks 7-12) - IN PROGRESS

**Challenge:** Video compression destroys frame-level perturbations.

**Approach: Optical Flow Poisoning**

1. **Extract motion vectors:**
   ```python
   import cv2
   flow = cv2.calcOpticalFlowFarneback(frame1, frame2, ...)
   # flow[y, x] = (dx, dy) motion vector at pixel (x, y)
   ```

2. **Generate temporal signature:**
   ```python
   # Cyclic pattern across frames
   signature_3d = [signature * sin(2π * t / period) for t in frames]
   ```

3. **Perturb flow vectors:**
   ```python
   flow_poisoned = flow + ε * signature_3d
   ```

4. **Reconstruct frames:**
   ```python
   frame2_poisoned = warp(frame1, flow_poisoned)
   ```

**Why this works:**
- Motion vectors are **compressed separately** from pixels (H.264 motion compensation)
- Small perturbations in motion create "impossible physics" that AI learns
- Human eye can't detect 1-pixel motion errors

**GPU Requirements:**
- Optical flow extraction: ~100ms/frame (needs GPU)
- Solution: Modal.com workers with A100 GPUs

**Deliverables:**
- `video_poison.py`: Video poisoning via optical flow
- `video_detector.py`: Extract temporal signatures from models
- Beta UI tab: "Protect Video (Sora Defense)"

---

### Phase 3: Multi-Modal Integrations (Month 4+)

**Code Protection (ACW):**
```python
from acw import CodeWatermarker
watermarker = CodeWatermarker()
poisoned_code = watermarker.mark(source_code, signature)
```

**Audio Protection (AudioSeal):**
```python
from audioseal import Watermarker
watermarker = Watermarker.load("facebook/audioseal")
poisoned_audio = watermarker.watermark(audio, signature)
```

**Text Protection (MarkLLM):**
```python
from markllm.watermark import Watermarker
watermarker = Watermarker("adaptive-token")
poisoned_text = watermarker.mark(text, signature)
```

**Architecture:**
- Each modality = separate API endpoint
- Shared signature management system
- Unified detection dashboard

---

<a name="security-analysis"></a>
## 5. Security Analysis

### 5.1 Threat Model

**Attacker Goals:**
1. Train AI on your data without detection
2. Remove the radioactive signature
3. False-flag by injecting someone else's signature

**Defender Goals:**
1. Detect unauthorized training (even if only 1% of data is yours)
2. Survive aggressive preprocessing (compression, augmentation)
3. Prove ownership in court (cryptographic non-repudiation)

### 5.2 Attack Vectors & Defenses

| Attack | Description | Defense |
|--------|-------------|---------|
| **Sanitization** | Train denoising autoencoder to remove perturbations | Degrades quality below usability; signature is in learned features, not pixels |
| **Dilution** | Mix 1% poisoned + 99% clean data | Still detectable with enough poisoned samples (threshold tuning) |
| **Model Fine-tuning** | Train on clean data after poisoned data | Signature decays but doesn't disappear (test empirically) |
| **Adversarial Training** | Train model to be robust to perturbations | Arms race; we can increase ε or use adaptive attacks |
| **Signature Forgery** | Inject a fake signature | Cryptographic seed makes forgery computationally infeasible (2^256 search space) |

### 5.3 Legal Considerations

**Evidence Admissibility:**
- Signature generation uses cryptographic PRNG (Python `secrets` module)
- Seed stored securely (offline backup recommended)
- Statistical p-value < 0.001 for detection (strong evidence)

**Limitations:**
- Does not prove *how much* data was used (only that it was used)
- Could be defeated by clean-room implementation (if attacker knows you're watching)

---

<a name="performance-considerations"></a>
## 6. Performance Considerations

### 6.1 Computational Costs

**Image Poisoning:**
- ResNet-18 forward pass: ~10ms (CPU), ~2ms (GPU)
- Gradient computation: ~20ms (CPU), ~5ms (GPU)
- Per-image total: **~30ms (CPU), ~7ms (GPU)**
- Batch of 100 images: ~3 seconds (CPU), < 1 second (GPU)

**Video Poisoning (estimated):**
- Optical flow extraction: ~100ms/frame (GPU required)
- 1-minute 30fps video = 1800 frames → ~3 minutes processing
- Solution: Async job queue + GPU workers

**Detection:**
- Feature extraction: ~10ms/image
- Correlation computation: < 1ms
- Total for 100 test images: **~1 second**

### 6.2 Scalability

**Current (MVP):**
- Synchronous API (user waits for processing)
- Single server with CPU
- Limit: ~100 images/minute

**Phase 2 (Production):**
- Async job queue (Redis + Celery)
- GPU worker pool (Modal.com or RunPod)
- Target: 10,000 images/minute

**Cost Analysis:**
- Modal.com A10 GPU: $0.50/hour
- Average poisoning: 7ms/image = 514,000 images/hour
- Cost per image: **$0.0000097** (< 1/100th of a cent)

### 6.3 Storage Requirements

**Per Image:**
- Poisoned image: Same size as original (~5MB for high-res)
- Signature JSON: ~2KB
- Metadata: ~1KB

**10,000 Images:**
- Images: ~50GB
- Signatures: ~20MB
- Negligible storage cost

---

## Conclusion

Project Sigil transforms cutting-edge research (radioactive data) into practical tooling. The mathematical foundation is solid (peer-reviewed ICML paper), the implementation is efficient (< 30ms/image), and the approach is extensible (video, code, audio coming in Phase 2-3).

**Next Steps:**
1. Complete Phase 1 web UI
2. Empirical testing: How many poisoned images needed for reliable detection?
3. Begin video poisoning research (optical flow approach)
4. Deploy MVP and gather user feedback

---

**Questions?** See [RESEARCH.md](RESEARCH.md) for academic references or open an issue on GitHub.
