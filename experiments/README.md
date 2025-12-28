# Experiments Directory

This directory contains experimental approaches and intermediate results from developing compression-robust video poisoning.

## Contents

### Training Scripts (Failed Approaches)
- `train_adaptive_signature.py` - Gradient descent with differentiable H.264 codec
  - **Result:** FAILED - codec mismatch (diff codec PSNR 49 dB vs real 39 dB)
  - **Lesson:** Differentiable approximations don't match real compression

- `train_contrastive_signature.py` - Contrastive learning with straight-through estimator
  - **Result:** FAILED - 80% TPR on diff codec, 0% TPR on real H.264
  - **Lesson:** Must validate on real compression during training

### Intermediate Signatures
- `optimized_signature_crf28.json` - From adaptive training
  - Detection on real H.264: 0.42 (seemed good)
  - Final validation: 30% TPR, 40% FPR (failed)

- `contrastive_signature_crf28.json` - From contrastive learning
  - Detection on diff codec: 0.45
  - Detection on real H.264: 0.095 (failed)

### Log Files
- `contrastive_training.log` - First contrastive training run
- `contrastive_training_v2.log` - Training with straight-through estimator
- `cmaes_training.log` - CMA-ES optimization (if run)

## Why These Failed

See [../COMPRESSION_ROBUSTNESS_JOURNEY.md](../COMPRESSION_ROBUSTNESS_JOURNEY.md) for detailed analysis.

**TL;DR:**
1. Cannot accurately model real H.264 with differentiable approximation
2. Soft quantization (tanh) doesn't match hard quantization (round)
3. Straight-through estimator still has 10 dB PSNR mismatch
4. Must use gradient-free optimization on REAL H.264

## What Worked

See parent directory for working implementations:
- `poison-core/frequency_poison.py` - DCT poisoning (works for CRF 18-23)
- `poison-core/frequency_detector.py` - Detection with contrastive validation
- `train_cmaes_signature.py` - CMA-ES optimization (to be run)
