# Project Basilisk: Current State Summary

**Date:** December 28, 2025
**Status:** Research Phase - Video Compression Robustness

---

## Quick Summary

**What works:**
- ✅ Image poisoning (production-ready)
- ✅ Video poisoning for CRF 18-23 (Vimeo/YouTube HD quality)

**What doesn't work yet:**
- ❌ Video poisoning for CRF 28 (YouTube SD quality)

**Current focus:**
- Finding optimal approach for CRF 28 compression robustness

---

## Image Poisoning: Production Ready ✅

**Status:** Fully validated and working

**Features:**
- Radioactive data marking based on Sablayrolles et al. (ICML 2020)
- Detection confidence > 0.1 for poisoned models
- PSNR > 40 dB (visually imperceptible)
- CLI + Web UI + API

**Usage:**
```bash
python poison-core/poison_cli.py poison input.jpg output.jpg
```

**Documentation:** [README.md](README.md)

---

## Video Poisoning: Partial Success 🟡

### What Works (CRF 18-23)

**Platforms:**
- Vimeo (professional hosting)
- YouTube HD uploads
- Archival systems

**Performance:**
- CRF 18: 0.60 detection score
- CRF 23: 0.50 detection score
- PSNR: 38 dB (excellent quality)

**Implementation:**
- Method: DCT coefficient perturbation in low-frequency bands
- Files: `poison-core/frequency_poison.py`, `poison-core/frequency_detector.py`

### What Doesn't Work (CRF 28)

**Problem:** Quantization destroys signature

**Attempts made:**
1. Baseline (epsilon=0.05): 0.06 detection score ❌
2. Adaptive training with differentiable codec: Failed validation (40% FPR) ❌
3. Contrastive learning: 0% TPR on real H.264 ❌
4. Straight-through estimator: Codec mismatch detected ❌

**Root cause:** Differentiable H.264 approximation doesn't match real compression (10 dB PSNR difference)

---

## The Decision Point

### Three Options Available

**Option A: CMA-ES Optimization** 🎯 Recommended
- Evolutionary algorithm on REAL H.264
- Runtime: 1-3 hours
- Guaranteed accurate (no approximation)
- File: `train_cmaes_signature.py` (ready to run)

**Option B: Accept CRF 23 Limit**
- Document as limitation
- Still valuable for professional platforms
- Ready to ship now

**Option C: High Epsilon Test**
- Quick test (5 minutes)
- epsilon=0.18 to overcome quantization
- Risk: visual quality degradation

### Recommendation

**Two-phase approach:**
1. Quick high-epsilon test (5 min) → If works, great!
2. If fails → CMA-ES (3 hrs) → If works, industry-breaking; If fails, accept Option B

---

## Documentation Map

### For Understanding Current State
- **[VIDEO_STATUS.md](VIDEO_STATUS.md)** - Detailed technical status
- **[DECISION_POINT.md](DECISION_POINT.md)** - Options analysis
- **This file** - Quick overview

### For Technical Deep Dive
- **[COMPRESSION_ROBUSTNESS_JOURNEY.md](COMPRESSION_ROBUSTNESS_JOURNEY.md)** - Complete history of all approaches tried
- **[DIAGNOSIS.md](DIAGNOSIS.md)** - Root cause analysis of failures
- **[experiments/README.md](experiments/README.md)** - Failed approaches archived

### For Understanding the Approach
- **[ALTERNATIVE_APPROACHES.md](ALTERNATIVE_APPROACHES.md)** - Research on different methods
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design (aspirational)

### For Using What Works
- **[README.md](README.md)** - Main project README (images work!)
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Test results

---

## Key Files by Purpose

### Production Code (Images)
```
poison-core/
├── radioactive_poison.py      ✅ Core algorithm (works)
├── poison_cli.py              ✅ CLI tool (works)
└── requirements.txt
```

### Working Video Code (CRF 18-23)
```
poison-core/
├── frequency_poison.py        ✅ DCT poisoning (works for CRF 18-23)
└── frequency_detector.py      ✅ Detection (contrastive-aware)
```

### Research Code (CRF 28)
```
train_cmaes_signature.py       🎯 Next to try (CMA-ES)
experiments/
├── train_adaptive_signature.py      ❌ Failed (codec mismatch)
└── train_contrastive_signature.py   ❌ Failed (codec mismatch)
```

### Validation & Testing
```
tests/
├── test_frequency_poison.py         ✅ Unit tests (passing)
├── test_compression_real.py         📊 Cross-CRF analysis
├── test_final_validation.py         📊 Statistical validation
├── test_contrastive_validation.py   📊 For CMA-ES results
└── debug_codec_mismatch.py          🔬 Diagnostic tool
```

---

## Technical Lessons Learned

1. **Differentiable approximations are dangerous**
   - Our codec: PSNR 49 dB (barely compresses)
   - Real H.264: PSNR 39 dB (moderate compression)
   - Result: Training shows 80% TPR, reality shows 0%

2. **Must validate on real systems frequently**
   - Added real H.264 validation every 100 training iterations
   - Caught codec mismatch immediately

3. **Contrastive learning essential for detection**
   - Naive correlation → 40% FPR (false positives)
   - Contrastive (maximize separation) → proper distributions

4. **Gradient-free optimization needed**
   - Can't backprop through ffmpeg
   - CMA-ES is the right tool for black-box optimization

---

## What This Means for Open Source Release

### Can Release Now (Option B)
- Working image poisoning
- Working video poisoning (CRF 18-23)
- Document CRF 28 as open research problem
- Still valuable contribution

### If CMA-ES Succeeds (Option A)
- Industry-breaking result
- First compression-robust video marking at YouTube quality
- Major impact
- CVPR 2026 submission

### Either Way
- Comprehensive testing methodology
- Clear documentation of limits
- Reproducible research
- Value to community

---

## Immediate Next Steps

**For you to decide:**

```bash
# Option 1: Quick test high epsilon (5 minutes)
# Create tests/quick_high_epsilon_test.py and run

# Option 2: Run CMA-ES (3 hours)
python3 train_cmaes_signature.py

# Option 3: Accept CRF 23, finalize docs
# Run final validation, update README
```

**My recommendation:** Try quick high-epsilon test first. If it works, amazing. If not, run CMA-ES overnight and we'll have a definitive answer tomorrow.

---

## Repository Stats

**Lines of code:**
- Core implementation: ~2,500 lines
- Tests: ~1,200 lines
- Documentation: ~15,000 words

**Test coverage:**
- Images: 55 tests passing ✅
- Video: 4/5 tests passing (CRF 28 pending)

**Documentation:**
- 15 markdown files
- Complete technical history
- Clear decision points

---

## Questions to Answer

1. **Can we solve CRF 28?**
   - Run CMA-ES to find out definitively
   - If yes → industry-breaking
   - If no → accept CRF 23 limit

2. **What quality trade-off is acceptable?**
   - PSNR > 30 dB minimum
   - May need epsilon=0.15-0.20 for CRF 28

3. **When to release?**
   - Now: Works for images + video (CRF 18-23)
   - After CMA-ES: Potentially works for CRF 28 too

---

## Bottom Line

**Images:** ✅ Done. Ship it.

**Video (CRF 18-23):** ✅ Done. Ship it.

**Video (CRF 28):** ⏳ One more experiment away from definitive answer (CMA-ES)

**Time to decision:** 3-4 hours of compute
