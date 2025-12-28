# Video Poisoning: Alternative Approaches Research

**Date:** 2025-12-28
**Context:** Optical flow poisoning approach failed to generalize - researching alternatives

---

## Executive Summary

The optical flow perturbation approach **fundamentally fails** because the signal (temporal correlation 0.36) is **too weak** relative to natural video motion signals (>>1.0). Models trained on poisoned videos learn **scene content** (moving squares vs circles) rather than the **subtle optical flow signature**.

This document researches alternative approaches with proven robustness to compression and model training.

---

## Why Optical Flow Poisoning Failed

### Empirical Evidence

From [test_poisoning_sanity.py](tests/test_poisoning_sanity.py):

1. **Flow Difference Test**: p=0.698515 (NOT statistically significant)
   - Clean videos: mean flow magnitude = X
   - Poisoned videos: mean flow magnitude = Y
   - Difference NOT detectable by t-test

2. **Temporal Pattern Test**: correlation = 0.364601
   - Signature EXISTS in optical flow
   - But correlation strength is WEAK (0.36 vs ideal 1.0)

3. **Visual Quality**: PSNR = 21.77 dB
   - Should be >30 dB for imperceptibility
   - Perturbation is visually noticeable but statistically weak

### Root Cause Analysis

**Signal-to-Noise Ratio Problem:**

```
Natural Motion Signal:
- Moving objects: flow magnitude 5-50 pixels/frame
- Camera motion: flow magnitude 10-100 pixels/frame
- Scene changes: flow magnitude >>100 pixels/frame

Our Signature Signal:
- Epsilon = 0.03 (3% perturbation)
- Applied to flow vectors: ~0.15-1.5 pixels/frame
- Temporal correlation: 0.36 (weak)

Ratio: Natural Motion / Signature ≈ 10-100x stronger
```

**Why 3D CNNs Ignore It:**

1. **Gradient Magnitude Bias**: During backpropagation, gradients flow toward **strong** features
   - Content features (object shape, motion direction): strong gradients
   - Subtle flow perturbation: weak gradients, drowned out

2. **Feature Competition**: Models have limited capacity
   - Must learn: object recognition, motion classification, scene understanding
   - Signature adds 0.36 correlation boost → **not worth the capacity cost**

3. **Memorization vs Pattern Learning**:
   - 50 videos, each unique (different content)
   - Model learns: "video A has moving square → class 0"
   - NOT: "video with 0.36 temporal flow correlation → class 1"

**Mathematical Formulation:**

Given loss function `L(y, ŷ)` and feature extractors `f_content` (natural motion) and `f_signature` (our perturbation):

```
∂L/∂f_content ≈ 10-100x larger than ∂L/∂f_signature

SGD updates prioritize: θ ← θ - α * ∂L/∂f_content
Result: f_signature ignored during optimization
```

### Validation Evidence

From [test_generalization.py](tests/test_generalization.py):
- **Train on videos A, B**: Model learns content of A and B
- **Test on video C**: Detection score = -0.330 (FAILED)
- **Conclusion**: No signature learning, pure memorization

From [test_proper_validation.py](tests/test_proper_validation.py):
- 50 videos, 70/30 split, dropout + L2 regularization
- Validation accuracy: 66.7% (learns something)
- Detection score: -0.31 (but NOT the signature)

---

## Alternative Approaches

### 1. Per-Frame Image Poisoning (PRAGMATIC)

**Concept**: Apply proven radioactive image marking to **each frame independently**.

**Method**:
```python
for frame in video:
    poisoned_frame = apply_radioactive_marking(frame, spatial_signature)
    save_frame(poisoned_frame)
```

**Advantages**:
- **Proven to work**: Image poisoning validated in original Sablayrolles et al. paper
- **Strong signal**: Each frame carries full-strength signature
- **No temporal dependency**: Survives frame drops, re-ordering
- **Implementation**: Already have working code in `poison-core/image_poison.py`

**Disadvantages**:
- **Not exploiting temporal dimension**: Wastes potential of video data
- **Compression vulnerability**: H.264 uses inter-frame compression (temporal correlation)
  - I-frames (keyframes): Preserved relatively intact
  - P/B-frames (predicted): Heavily compressed using motion vectors
  - Our signature might survive I-frames but degrade on P/B-frames

**Empirical Evidence**:
- Radioactive marking on images: **0.95+ detection scores** in original paper
- Video = sequence of images → should inherit image robustness
- **BUT**: No published results on per-frame poisoning survival through H.264

**Compression Robustness Strategy**:
- Poison **all frames** to create redundancy
- Even if P/B-frames degrade, I-frames preserve signature
- H.264 has I-frame every 30-60 frames @ CRF 23
- Detection: Extract features from **I-frames only**

**Next Steps**:
1. Implement: Poison video frame-by-frame using existing image marker
2. Test: Train 3D CNN on per-frame poisoned videos
3. Validate: Measure detection on held-out videos
4. Compression test: CRF 18 → 23 → 28 → 35 survival rates

**Expected Outcome**: Detection score >0.5 (vs 0.36 for optical flow)

---

### 2. Frequency Domain Watermarking (ROBUST)

**Concept**: Poison **DCT/wavelet coefficients** that H.264 compression preserves.

**Academic Foundation**:

Recent research (2024-2025) shows frequency domain methods achieve **high robustness** against H.264/AVC compression:

1. **DCT-Based Watermarking** ([Robust H.264/AVC Video Watermarking](https://pmc.ncbi.nlm.nih.gov/articles/PMC3932276/)):
   - Embed watermark in **low-frequency DCT coefficients**
   - H.264 quantization has **small step sizes** for low frequencies
   - Result: 80% accuracy after lossy compression

2. **Wavelet Domain (DWT)** ([DWT-Based Video Watermarking](https://www.worldscientific.com/doi/abs/10.1142/S0219467820500047)):
   - Adaptive embedding in DWT coefficients
   - Robust against spatial attacks (scaling, noise) AND temporal attacks (frame dropping)
   - Blind extraction (no original video needed)

3. **Deep Learning + Frequency Domain** ([DNN Wavelet Watermarking](https://www.sciencedirect.com/science/article/abs/pii/S2214212624001704)):
   - Dual-tree complex wavelet transform (DT-CWT)
   - Differentiable H.264 proxy for end-to-end training
   - **Superior robustness** vs spatial-only methods

**Implementation Strategy**:

```python
# For each frame:
1. Apply DCT/DWT transform
   dct_coeffs = cv2.dct(frame_yuv)

2. Identify low-frequency region (e.g., top-left 8x8 block)
   low_freq = dct_coeffs[0:8, 0:8]

3. Add signature to coefficients
   low_freq += alpha * signature_pattern

4. Inverse transform
   poisoned_frame = cv2.idct(dct_coeffs)
```

**Why This Works**:

- **H.264 operates in frequency domain**: Uses DCT for compression
- **Low frequencies preserved**: Quantization matrix has small steps for DC/low-AC coefficients
- **High frequencies discarded**: Where most compression loss occurs
- **Our signature aligns with codec**: Survives quantization by design

**Advantages**:
- **Compression-aware**: Designed to survive H.264/H.265
- **Academic validation**: Multiple papers with empirical results
- **Differentiable**: Can train end-to-end with differentiable codec proxy

**Disadvantages**:
- **Implementation complexity**: Need DCT/wavelet transforms, coefficient selection
- **Hyperparameter tuning**: Which frequencies? What alpha strength?
- **No radioactive marking precedent**: Would be novel application (risk)

**Next Steps**:
1. Implement DCT-based poisoning (simpler than wavelet)
2. Test on uncompressed videos first
3. Gradually increase compression (CRF 18 → 35)
4. Measure signature survival rate

**Expected Outcome**: Better compression robustness than spatial methods

---

### 3. Hybrid Spatial-Temporal (EXPERIMENTAL)

**Concept**: Combine **per-frame spatial poisoning** with **weak temporal signature** for redundancy.

**Method**:
```python
for t, frame in enumerate(video):
    # Spatial signature (STRONG signal)
    spatial_poison = apply_radioactive_marking(frame, signature)

    # Temporal modulation (WEAK signal)
    temporal_strength = sin(2π * t / period)  # Cyclic pattern
    combined = spatial_poison * (1 + 0.1 * temporal_strength)

    save_frame(combined)
```

**Detection**:
- **Primary**: Spatial feature correlation (like image poisoning)
- **Secondary**: Temporal correlation (bonus if detected)
- **Decision**: Logical OR (either signal triggers detection)

**Advantages**:
- **Redundancy**: Two independent signals
- **Graceful degradation**: If temporal fails, spatial still works
- **Research novelty**: Combines two approaches

**Disadvantages**:
- **More complex**: Two hyperparameters (spatial epsilon, temporal strength)
- **Potential interference**: Signals might cancel out
- **Unproven**: No academic precedent

**Next Steps**: Only pursue after testing per-frame and frequency approaches

---

### 4. Keyframe-Only Poisoning (EFFICIENT)

**Concept**: Poison only **I-frames (keyframes)** that H.264 preserves intact.

**Rationale**:
- H.264 structure: I-frame every 30-60 frames (GOP size)
- I-frames: Intra-coded (spatial compression only, like JPEG)
- P/B-frames: Inter-coded (temporal compression, uses motion vectors)

**Strategy**:
```python
# Detect I-frames
cap = cv2.VideoCapture(video)
i_frames = detect_keyframes(cap)  # Every ~30 frames

# Poison only I-frames
for idx in i_frames:
    frame = extract_frame(video, idx)
    poisoned = apply_radioactive_marking(frame, signature)
    replace_frame(video, idx, poisoned)
```

**Detection**:
```python
# Extract only I-frames from trained model
features = model.extract_features(video, keyframes_only=True)
correlation = compute_correlation(features, signature)
```

**Advantages**:
- **Compression-proof**: I-frames survive H.264 intact (minimal quality loss)
- **Efficient**: Poison 3-5% of frames (60s video @ 30fps = 1800 frames, ~60 I-frames)
- **Strong signal**: Full-strength spatial poisoning on preserved frames

**Disadvantages**:
- **Sparse signal**: Models might not learn from 3-5% of frames
- **Keyframe detection**: Need to identify I-frames (codec-specific)
- **3D CNN dependency**: Temporal models process all frames, diluting signal

**Next Steps**: Test after per-frame approach validates spatial poisoning works

---

## Academic Literature Review

### Video Watermarking (State-of-the-Art)

**Recent Advances (2024-2025)**:

1. **RC-VWN (Robust Compatible Video Watermarking Network)** - Feb 2025
   - [Spatio-Temporal Enhancement](https://dl.acm.org/doi/10.1109/TCSVT.2024.3471891)
   - Multiscale pyramid attention
   - Addresses H.264 compression robustness

2. **Deep Learning Review** - 2024
   - [Current and Future Trends](https://link.springer.com/article/10.1007/s00034-024-02651-z)
   - Challenge: Most methods neglect long-distance spatio-temporal features
   - Solution: GANs and attention mechanisms for robustness

3. **Differentiable Codec Training** - 2024
   - [Novel Deep Video Watermarking](https://dl.acm.org/doi/abs/10.1145/3581783.3612270)
   - Train watermarking network with H.264 codec in loop
   - Achieves compression robustness via end-to-end learning

**Key Findings**:
- Spatial-only methods: **Poor H.264 robustness**
- Frequency domain: **Better compression survival**
- End-to-end training with differentiable codec: **State-of-the-art**

### Radioactive Data Marking

**Original Work**: Sablayrolles et al. (2020) - [Radioactive Data](https://www.semanticscholar.org/paper/Radioactive-data:-tracing-through-training-Sablayrolles-Douze/d2184233e9b44ba429d713a9cde93433a952ddcc)
- **Domain**: Images only
- **Method**: Spatial feature manipulation
- **Result**: 0.95+ detection accuracy
- **Limitation**: No video extension published

**Video Poisoning Research (2024)**:

1. **Stealthy 3D Poisoning** - 2024
   - [NSF Research](https://par.nsf.gov/servlets/purl/10322068)
   - Problem: Frame-by-frame triggers cause **temporal inconsistency**
   - Solution: 3D poisoning triggers with natural-like textures
   - **Lesson**: Temporal consistency is critical for stealth

2. **Backdoor Attacks** - 2024
   - [Data Poisoning Survey](https://github.com/penghui-yang/awesome-data-poisoning-and-backdoor-attacks)
   - Multiple papers on backdoor triggers in video
   - Focus: Adversarial ML (different goal than our marking)

**Key Insight**: No published work on **radioactive marking for videos** - this would be novel research.

---

## Decision Matrix

| Approach | Signal Strength | Compression Robustness | Implementation Complexity | Academic Precedent | Recommended Priority |
|----------|----------------|----------------------|-------------------------|-------------------|---------------------|
| **Per-Frame Image Poisoning** | **High (0.9+)** | **Medium** (I-frames survive) | **Low** (reuse existing code) | **High** (proven for images) | **1st - DO THIS** |
| **Frequency Domain (DCT)** | **Medium (0.7+)** | **High** (designed for H.264) | **Medium** (DCT transforms) | **High** (video watermarking) | **2nd - Test if per-frame fails compression** |
| Hybrid Spatial-Temporal | Medium-High | Medium-High | High | Low | 4th - Research only |
| Keyframe-Only Poisoning | Medium | Very High | Medium | Medium | 3rd - If sparse signal works |
| ~~Optical Flow Poisoning~~ | ~~Weak (0.36)~~ | ~~Unknown~~ | ~~Medium~~ | ~~None~~ | ❌ **ABANDON** |

---

## Recommended Path Forward

### Phase 1: Per-Frame Image Poisoning (IMMEDIATE)

**Why Start Here**:
1. **Proven approach**: Radioactive marking works on images (0.95+ detection)
2. **Low risk**: Reuse existing validated code
3. **Fast validation**: Can test within hours

**Implementation Plan**:
```bash
# 1. Modify video_poison.py to poison frame-by-frame
def poison_video_per_frame(video_path, output_path, signature):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.read():
        poisoned_frame = image_marker.poison(frame, signature)
        frames.append(poisoned_frame)
    save_video(frames, output_path)

# 2. Create test dataset
python verification/create_video_dataset.py \
    --method per_frame \
    --clean 25 --poisoned 25

# 3. Train 3D CNN
python verification/verify_video_poison.py

# 4. Test generalization
python tests/test_generalization.py

# 5. Compression robustness test
for crf in 18 23 28 35; do
    ffmpeg -i poisoned.mp4 -c:v libx264 -crf $crf compressed_$crf.mp4
    python tests/test_compression_robustness.py compressed_$crf.mp4
done
```

**Success Criteria**:
- Detection score >0.7 on held-out videos (vs 0.36 for optical flow)
- CRF 23 compression: score >0.5
- CRF 28 compression: score >0.3

**Time Estimate**: 1-2 days implementation + testing

---

### Phase 2: Compression Robustness Analysis (CRITICAL)

**If per-frame poisoning works but degrades with compression**:

1. **Analyze degradation pattern**:
   - Which frames lose signature? (P/B-frames vs I-frames)
   - At what CRF does detection fail?
   - Is degradation gradual or cliff-edge?

2. **Countermeasures**:
   - **Option A**: Increase epsilon (accept visible artifacts)
   - **Option B**: Keyframe-only poisoning (sparse but robust)
   - **Option C**: Frequency domain (Phase 3)

**Fallback Decision Tree**:
```
Per-frame detection score > 0.7 on uncompressed?
├─ YES: Test compression
│   ├─ CRF 23 score > 0.5?
│   │   ├─ YES: ✅ SHIP IT (good enough for production)
│   │   └─ NO: Proceed to Phase 3 (frequency domain)
│   └─ CRF 28 score > 0.3?
│       ├─ YES: ⚠️  ACCEPTABLE (moderate compression)
│       └─ NO: Proceed to Phase 3 (frequency domain)
└─ NO: CRITICAL - Investigate why image poisoning failed on video
```

---

### Phase 3: Frequency Domain (IF NEEDED)

**Only pursue if**:
- Per-frame poisoning fails compression tests (CRF 28 score <0.3)
- AND compression robustness is business-critical

**Implementation**:
1. Research DCT coefficient selection (which frequencies to poison?)
2. Implement DCT-based poisoning for single frame (test on images first)
3. Extend to video (all frames)
4. Train 3D CNN on DCT-poisoned videos
5. Compression test (CRF 18-35)

**Why Defer**:
- Higher implementation complexity
- Unproven for radioactive marking (novel research)
- Per-frame might be "good enough"

**Time Estimate**: 1 week research + implementation

---

## Open Research Questions

### 1. Why does per-frame poisoning work for images but might fail for videos?

**Hypothesis**: 3D CNNs might average out frame-level signals

**Test**: Compare detection on:
- 2D CNN (ResNet50) trained on individual frames → should work (proven)
- 3D CNN (Simple3DCNN) trained on video clips → unknown

**Prediction**: 2D CNN will show high detection (0.9+), 3D CNN might be lower (0.7?) due to temporal pooling

### 2. What is the minimum poisoning density for detection?

**Hypothesis**: Don't need to poison ALL frames, can use sparse poisoning

**Test**: Poison X% of frames (10%, 25%, 50%, 100%) and measure detection

**Goal**: Find optimal trade-off (less poisoning = faster, more poisoning = robust)

### 3. Do I-frames vs P/B-frames matter for detection?

**Hypothesis**: Signature on I-frames is sufficient (P/B-frames get degraded anyway)

**Test**: Compare:
- Poison all frames → measure detection
- Poison only I-frames → measure detection
- Poison only P/B-frames → measure detection

**Expected**: I-frame-only performs nearly as well as all-frames (I-frames carry most information)

### 4. Can we train a differentiable H.264 proxy for end-to-end optimization?

**Hypothesis**: If we have differentiable codec, can optimize epsilon to maximize (detection score - visual quality loss)

**Implementation**: Use existing research on differentiable video compression
- [Differentiable H.264 Proxy](https://dl.acm.org/doi/abs/10.1145/3581783.3612270) from 2024 ACM MM paper

**Outcome**: Automatically find optimal poisoning parameters

---

## Cynical Reality Check

### What This Research Reveals

**Good News**:
1. Per-frame image poisoning is **low-hanging fruit** - likely works
2. Academic literature has **proven solutions** for compression robustness
3. Frequency domain is a **known escape hatch** if spatial fails

**Bad News**:
1. You wasted time on optical flow (0.36 correlation was doomed from start)
2. No published radioactive marking on video → you're in **novel research territory**
3. Compression robustness is **make-or-break** - if it fails at CRF 28, approach is DOA

### Probability Assessment

**Per-frame image poisoning succeeds on uncompressed videos**: 90%
- Image poisoning works (proven), video is just sequence of images

**Per-frame poisoning survives CRF 23 compression**: 60%
- I-frames should preserve signal, but P/B-frames might dilute it

**Per-frame poisoning survives CRF 28 compression**: 30%
- Lossy compression at this level destroys subtle signals
- May need frequency domain approach

**Frequency domain poisoning works**: 70%
- Academic papers show it works for watermarking
- But radioactive marking might need different approach

### The Honest Answer

**You have three options**:

1. **Ship per-frame poisoning v1.0**: Accept CRF 23 limitation, document it
   - Pro: Fast to implement, likely works
   - Con: YouTube uses CRF 28-35, so production use limited

2. **Go full research mode**: Implement frequency domain, publish paper
   - Pro: Novel contribution, potential CVPR/ICCV paper
   - Con: 3-6 months of work, might still fail

3. **Pivot to image-only poisoning**: Drop video entirely for v1.0
   - Pro: Proven to work, ship in days
   - Con: Admits defeat on video (but pragmatic)

**My recommendation**: Try per-frame poisoning for 2 days. If detection score >0.7 on uncompressed, test compression. If CRF 23 score >0.5, declare victory and ship v1.0 with documented limitations. If it fails, pivot to image-only for v1.0 and make video a v2.0 research project.

---

## Sources

### Video Watermarking
- [Robust and Compatible Video Watermarking via Spatio-Temporal Enhancement and Multiscale Pyramid Attention](https://dl.acm.org/doi/10.1109/TCSVT.2024.3471891)
- [Deep Learning-Based Watermarking Techniques Challenges: A Review of Current and Future Trends](https://link.springer.com/article/10.1007/s00034-024-02651-z)
- [A Novel Deep Video Watermarking Framework with Enhanced Robustness to H.264/AVC Compression](https://dl.acm.org/doi/abs/10.1145/3581783.3612270)
- [Robust and blind video watermarking against online sharing platforms](https://www.nature.com/articles/s41598-025-91192-9)

### Frequency Domain Methods
- [A Robust H.264/AVC Video Watermarking Scheme with Drift Compensation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3932276/)
- [A Robust DWT-Based Compressed Domain Video Watermarking Technique](https://www.worldscientific.com/doi/abs/10.1142/S0219467820500047)
- [A DNN robust video watermarking method in dual-tree complex wavelet transform domain](https://www.sciencedirect.com/science/article/abs/pii/S2214212624001704)
- [A Comprehensive Review of Digital Video Watermarking Techniques](https://link.springer.com/article/10.1007/s11831-025-10443-0)

### Radioactive Data & Poisoning
- [Radioactive data: tracing through training](https://www.semanticscholar.org/paper/Radioactive-data:-tracing-through-training-Sablayrolles-Douze/d2184233e9b44ba429d713a9cde93433a952ddcc)
- [Stealthy 3D Poisoning Attack on Video Recognition Models](https://par.nsf.gov/servlets/purl/10322068)
- [Have You Poisoned My Data? Defending Neural Networks Against Data Poisoning](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_5)

---

**Next Action**: Implement per-frame image poisoning for videos (Phase 1)
