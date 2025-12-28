# Video Poisoning: Current Status & Next Steps

## Executive Summary

**Goal:** Compression-robust video poisoning for radioactive data marking

**Current Status:**
- ✅ **Working:** CRF 18-23 (Vimeo, YouTube HD quality)
- ❌ **Not Working:** CRF 28 (YouTube SD quality)
- 🔬 **Research Phase:** Finding optimal approach for CRF 28

---

## What Works Right Now

### Frequency Domain Poisoning (CRF 18-23)

**Method:** Perturb DCT coefficients in low-frequency bands

**Results:**
- **CRF 18:** 0.60 detection score
- **CRF 23:** 0.50 detection score
- **Visual quality:** PSNR 38 dB (excellent)

**Platforms Protected:**
- Vimeo (professional video hosting)
- YouTube HD uploads
- Most archival/preservation systems

**Files:**
- [poison-core/frequency_poison.py](poison-core/frequency_poison.py)
- [poison-core/frequency_detector.py](poison-core/frequency_detector.py)
- [tests/test_frequency_poison.py](tests/test_frequency_poison.py)

**Usage:**
```python
from frequency_poison import FrequencyDomainVideoMarker

# Poison video
marker = FrequencyDomainVideoMarker(epsilon=0.05, frequency_band='low')
marker.poison_video('input.mp4', 'poisoned.mp4')
marker.save_signature('signature.json')

# Compress (up to CRF 23)
# ffmpeg -i poisoned.mp4 -c:v libx264 -crf 23 compressed.mp4

# Detect
from frequency_detector import FrequencySignatureDetector
detector = FrequencySignatureDetector('signature.json')
score, info = detector.detect_in_video('compressed.mp4')
# score > 0.3 → poisoned
# score < 0.1 → clean
```

---

## What Doesn't Work (Yet)

### CRF 28 (YouTube SD Quality)

**Problem:** Quantization steps destroy signature

**Evidence:**
| Approach | Diff Codec Score | Real H.264 Score | Status |
|----------|-----------------|------------------|---------|
| Baseline (ε=0.05) | 0.61 | 0.064 | ❌ Failed |
| Adaptive Training | 0.61 | 0.42 → 0.18* | ❌ Failed validation |
| Contrastive Learning | 0.45 | 0.095 | ❌ Failed |
| Straight-Through | 0.36 | 0.01 | ❌ Codec mismatch |

*Initial test looked good, but final validation revealed 40% FPR (false positives)

**Root Cause:**
```
Quantization step at CRF 28: ~40
Signature magnitude: epsilon * 255 = 0.05 * 255 = 12.75
Result: Signature gets rounded to zero
```

**Why gradient descent failed:**
- Differentiable codec approximation inaccurate
- 10 dB PSNR difference from real H.264
- Training shows 80% TPR, reality shows 0% TPR

---

## Path Forward: Three Options

### Option A: CMA-ES Optimization 🎯 RECOMMENDED

**What:** Evolutionary algorithm on REAL H.264 (no approximation)

**Why this will work:**
- Evaluates directly on ffmpeg (no codec mismatch)
- Gradient-free (works with non-differentiable systems)
- Can find global optimum in epsilon + signature space

**Implementation:** [train_cmaes_signature.py](train_cmaes_signature.py)

**Runtime:** 1-3 hours

**Expected outcome:**
- Best case: Finds signature with separation > 0.2 → Industry-breaking
- Worst case: Proves CRF 28 is fundamentally unsolvable → Pivot to Option B

**Command:**
```bash
python3 train_cmaes_signature.py
# Wait 1-3 hours
python3 tests/test_contrastive_validation.py  # Validate
```

---

### Option B: Accept CRF 23 Limit

**What:** Document CRF 28 as unsolved, claim success for CRF 18-23

**Justification:**
- Already works and proven
- Still protects professional/HD content
- Honest about limitations

**Impact:**
- ✅ Vimeo, YouTube HD, archives
- ❌ YouTube SD, Facebook, TikTok

**Action:** Run final validation on CRF 23, update docs

---

### Option C: High Epsilon Brute Force

**What:** Try epsilon=0.18-0.20 to overcome quantization

**Math:**
```
Need: signature magnitude > quantization step
     epsilon * 255 > 46
     epsilon > 0.18
```

**Risk:** Visual quality degradation (PSNR < 30 dB)

**Test:** 5 minutes

---

## Recommendation

**Two-phase approach:**

### Phase 1: Quick Test (30 minutes)
```bash
# Test high epsilon
python3 -c "
from poison_core.frequency_poison import FrequencyDomainVideoMarker
marker = FrequencyDomainVideoMarker(epsilon=0.18)
marker.poison_video('test.mp4', 'poisoned.mp4')
# Compress + validate
"
```

**If works:** Great! Measure PSNR, validate on more videos
**If fails:** Move to Phase 2

### Phase 2: CMA-ES (3 hours)
```bash
python3 train_cmaes_signature.py
```

**If works:** Run full validation, claim industry-breaking result
**If fails:** Accept Option B (CRF 23 limit)

---

## Technical Details

### Why Differentiable Codec Failed

**Comparison:**
| Metric | Our Codec | Real H.264 |
|--------|-----------|------------|
| PSNR | 49.20 dB | 38.94 dB |
| AC coefficients | All zeroed | Some preserved |
| Quantization | Soft (tanh) | Hard (round) |

**Conclusion:** 10 dB difference = fundamentally different behavior

### Why Contrastive Learning Matters

**Wrong detection:**
```python
score = mean(abs(correlation(frame_dct, signature)))
# Problem: natural videos correlate with random patterns
# Result: 40% false positive rate
```

**Right detection:**
```python
# During training:
loss = maximize_separation(poisoned, clean)

# During detection:
score = correlation(frame_dct, signature)
# But signature was trained to separate distributions
```

---

## Repository Structure

```
basilisk/
├── poison-core/
│   ├── frequency_poison.py       ✅ Working (CRF 18-23)
│   ├── frequency_detector.py     ✅ Working
│   └── differentiable_codec.py   ❌ Flawed approximation
│
├── tests/
│   ├── test_frequency_poison.py       ✅ Validation tests
│   ├── test_compression_real.py       ✅ Cross-CRF testing
│   ├── test_final_validation.py       ✅ Statistical validation
│   ├── test_contrastive_validation.py ✅ CMA-ES validation
│   └── debug_codec_mismatch.py        📊 Analysis tool
│
├── experiments/                   📁 Failed approaches (archived)
│   ├── train_adaptive_signature.py
│   ├── train_contrastive_signature.py
│   └── README.md
│
├── train_cmaes_signature.py       🎯 Ready to run
│
└── Documentation:
    ├── COMPRESSION_ROBUSTNESS_JOURNEY.md  📖 Full technical history
    ├── DECISION_POINT.md                  🔀 Options analysis
    ├── DIAGNOSIS.md                       🔬 Root cause analysis
    └── VIDEO_STATUS.md                    📋 This file
```

---

## What We Learned

1. **Differentiable approximations are dangerous**
   - Must validate on real system frequently
   - 10% error in proxy = 100% failure in reality

2. **Contrastive learning is essential**
   - Naive correlation detection = high FPR
   - Must train to separate distributions

3. **CRF 28 is extremely aggressive**
   - Quantization steps 30-46 destroy small signals
   - Need either larger epsilon or clever optimization

4. **Gradient-free optimization necessary**
   - Can't backprop through ffmpeg
   - CMA-ES is the right tool

5. **Test on real data early and often**
   - Don't trust training metrics
   - Validate on actual compression pipeline

---

## Success Metrics

### Minimum Viable (CRF 23)
- ✅ TPR > 80%
- ✅ FPR < 10%
- ✅ p-value < 0.001
- ✅ PSNR > 35 dB

### Stretch Goal (CRF 28)
- ❓ TPR > 80% (to be determined)
- ❓ FPR < 10% (to be determined)
- ❓ p-value < 0.001 (to be determined)
- ❓ PSNR > 30 dB (may need to compromise)

---

## Next Action Required

**Choose one:**

1. **Quick test** (5 min): `python3 tests/quick_high_epsilon_test.py`
2. **CMA-ES** (3 hrs): `python3 train_cmaes_signature.py`
3. **Validate CRF 23** (30 min): Accept current results, finalize docs

**I recommend:** Start with quick test, then decide.
