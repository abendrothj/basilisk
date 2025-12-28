# Decision Point: Compression-Robust Video Poisoning

## Current Status

**Goal:** Compression-robust video poisoning that survives YouTube-quality H.264 (CRF 28)

**Results So Far:**
- ✅ **CRF 18-23**: Works (0.50-0.60 detection score)
- ❌ **CRF 28**: Failed (0.06 detection score, 0% TPR)

**Why CRF 28 Failed:**
- Quantization steps (30-46) destroy small AC coefficients
- Our signature magnitude (12.75) too small
- Differentiable codec approximation inaccurate (10 dB PSNR difference)

---

## The Four Options

### Option A: CMA-ES Optimization 🎯 **RECOMMENDED**

**What:** Evolutionary algorithm that evaluates on REAL H.264 (no approximation)

**Pros:**
- Guaranteed accurate (tests on real ffmpeg)
- Already implemented (`train_cmaes_signature.py`)
- Will give definitive answer on CRF 28 feasibility

**Cons:**
- Slow (1-3 hours compute)
- No guarantee of success

**Timeline:**
- Run overnight: 1-3 hours
- Validation: 30 minutes
- Total: ~4 hours

**Expected Outcome:**
- Best case: Separation > 0.2 (TPR >80%, FPR <10%) → Industry-breaking result
- Worst case: Separation < 0.1 → Proven that CRF 28 is unsolvable → Pivot to Option B

**Command:**
```bash
python3 train_cmaes_signature.py  # ~3 hours
python3 tests/test_contrastive_validation.py  # Validate
```

---

### Option B: Target CRF 18-23 (Vimeo Quality)

**What:** Accept limitation, claim success for professional platforms

**Pros:**
- Already works (proven)
- No additional research needed
- Still valuable (protects Vimeo, professional content)

**Cons:**
- Doesn't protect against most aggressive compression
- Misses YouTube SD (billions of videos)

**Platforms Covered:**
- ✅ Vimeo Pro (CRF 18-20)
- ✅ YouTube HD (CRF 23)
- ❌ YouTube SD (CRF 28)
- ❌ Facebook (CRF 30+)

**Timeline:** Immediate

**Action:**
```bash
# Run validation on CRF 23
python3 tests/test_final_validation.py --crf 23
# Update docs to specify CRF 18-23 target
```

---

### Option C: Increase Epsilon (Brute Force)

**What:** Use much larger perturbation (epsilon=0.15-0.20) to overcome quantization

**Hypothesis:** If signature magnitude > quantization step, it survives

**Math:**
```
Current: epsilon * 255 = 0.05 * 255 = 12.75
Needed:  epsilon * 255 > 46.0 (largest quant step)
        epsilon > 0.18
```

**Pros:**
- Simple to test
- May work without complex optimization

**Cons:**
- Visual quality degradation (PSNR < 30 dB)
- May still fail (quantization is nonlinear)

**Timeline:** 1 hour

**Action:**
```bash
# Quick test with high epsilon
marker = FrequencyDomainVideoMarker(epsilon=0.18)
marker.poison_video('test.mp4', 'poisoned.mp4')
# Compress + validate
```

---

### Option D: Hybrid CMA-ES + High Epsilon

**What:** Run CMA-ES with wider epsilon search space [0.05, 0.25]

**Goal:** Find optimal trade-off between detection and quality

**Pros:**
- Explores full solution space
- May find sweet spot we missed

**Cons:**
- Same time cost as Option A
- Risk of poor visual quality

**Timeline:** 3-4 hours

---

## Recommendation Matrix

| Priority | Option | When to Choose |
|----------|--------|----------------|
| 🥇 | **Option A (CMA-ES)** | You want the definitive answer on CRF 28 |
| 🥈 | **Option B (CRF 23)** | Time-constrained, want guaranteed success |
| 🥉 | **Option C (High Epsilon)** | Quick test before committing to CMA-ES |
| 4️⃣ | **Option D (Hybrid)** | Option A fails, willing to trade quality |

---

## My Recommendation

**Run Option C (30 min), then Option A (3 hours)**

### Phase 1: Quick Validation (Option C)
```bash
# Test if high epsilon solves it
python3 -c "
from poison_core.frequency_poison import FrequencyDomainVideoMarker
marker = FrequencyDomainVideoMarker(epsilon=0.18)
# ... quick test on 1-2 videos
"
```

**If Phase 1 works (separation >0.2):**
→ Great! Test on more videos, measure PSNR
→ If PSNR >30 dB, we're done
→ If PSNR <30 dB, run Option D to optimize

**If Phase 1 fails (separation <0.1):**
→ Run Option A (CMA-ES) to exhaustively search
→ If CMA-ES also fails, pivot to Option B

---

## Decision Criteria

### Declare Success If:
- TPR > 80% (poisoned videos detected)
- FPR < 10% (clean videos pass)
- p-value < 0.001 (statistically significant)
- PSNR > 30 dB (acceptable quality)

### Pivot to CRF 23 If:
- CMA-ES fails after 30 generations
- No epsilon achieves both detection + quality
- Separation plateaus < 0.15

---

## What This Means for the Project

### If CRF 28 Works:
- **Industry-breaking result** ✅
- First compression-robust video marking
- Open source immediately
- CVPR 2026 submission
- Major media coverage potential

### If CRF 23 is the Limit:
- **Still valuable** ✅
- Protects professional content (Vimeo, YouTube HD)
- Honest about limitations
- Document CRF 28 as open research problem
- Open source with clear documentation of constraints

### Either Way:
- Significant contribution to field
- Comprehensive testing methodology
- Open source tools for community
- Clear understanding of compression robustness limits

---

## Next Action

**Choose one:**

```bash
# Option A: CMA-ES (recommended)
python3 train_cmaes_signature.py

# Option B: Validate CRF 23
python3 tests/test_final_validation.py --crf 23

# Option C: Quick high-epsilon test
python3 -c "
import sys; sys.path.append('poison-core')
from frequency_poison import FrequencyDomainVideoMarker
from frequency_detector import FrequencySignatureDetector
import subprocess

# Test epsilon=0.18
marker = FrequencyDomainVideoMarker(epsilon=0.18)
marker.poison_video('test_video.mp4', 'poisoned_high_eps.mp4')
subprocess.run(['ffmpeg', '-i', 'poisoned_high_eps.mp4', '-c:v',
                'libx264', '-crf', '28', '-y', 'compressed.mp4'])
# Then detect...
"

# Option D: Hybrid
# (Modify train_cmaes_signature.py epsilon bounds to [0.05, 0.25])
python3 train_cmaes_signature.py
```

**My suggestion:** Start with Option C (5 min test), then decide whether to run Option A or accept Option B.
