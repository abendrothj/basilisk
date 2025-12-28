# Diagnosis: Why Contrastive Training Failed

## Test Results

### Differentiable Codec (Training)
- Clean: 0.13 ± 0.02
- Poisoned: 0.45 ± 0.03
- Separation: 0.33
- TPR: 80%, FPR: 0%

### Real H.264 CRF 28 (Validation)
- Clean: 0.090 ± 0.046
- Poisoned: 0.095 ± 0.046
- Separation: 0.005
- TPR: 0%, FPR: 0%
- p-value: 0.77 (NOT significant)

## Root Cause: Differentiable Codec Mismatch

The differentiable codec does NOT accurately model real H.264 quantization.

### Differentiable Codec (Soft Quantization)
```python
normalized = dct / (q * temperature)
quantized = tanh(normalized) * q
```

**Problem:** This is a smooth, differentiable approximation. It preserves gradients but doesn't accurately model the **hard clipping** that real quantization does.

### Real H.264 Quantization (Hard)
```
quantized = round(dct / q) * q
```

**Reality:** This is a hard step function. Small DCT coefficients (<q/2) get **completely zeroed**. The signature we're adding may be getting zeroed entirely.

## Why Previous "Adaptive Training" Seemed to Work

The adaptive training showed 0.42 detection at CRF 28. But this was still on differentiable codec during training, with only final test on real H.264. The signature wasn't optimized for REAL quantization behavior.

## The Fundamental Challenge

**You cannot backpropagate through real H.264 compression.**

Options:
1. **Fix the differentiable codec** - Make it more accurately model hard quantization
2. **Use zeroth-order optimization** - Optimize signature without gradients (evolutionary algorithms, black-box optimization)
3. **Hybrid approach** - Train with differentiable codec, fine-tune with real compression feedback

## Recommended Fix: Zeroth-Order Optimization

Since we can't backprop through real H.264, we need gradient-free optimization:

### Approach: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
1. Population-based search for optimal signature
2. Each candidate evaluated on REAL H.264 compression
3. No gradients needed - just fitness scores
4. Proven for black-box optimization

### Fitness Function
```python
def fitness(signature, epsilon):
    # Generate poisoned videos with this signature
    # Compress with REAL ffmpeg CRF 28
    # Measure:
    #   - clean_score (should be low)
    #   - poisoned_score (should be high)
    #   - separation = poisoned_score - clean_score
    # Return: separation (maximize)
    return separation
```

### Why This Will Work
- Directly optimizes for REAL H.264 behavior
- No approximation error
- Slower (no gradients) but accurate
- Can run in parallel (evaluate many candidates simultaneously)

## Alternative: Better Differentiable Codec

Improve the quantization approximation:

### Straight-Through Estimator
```python
# Forward: Hard quantization (like real H.264)
quantized = torch.round(dct / q) * q

# Backward: Pretend it's identity (straight-through)
quantized_ste = dct + (quantized - dct).detach()
```

This gives gradients (for training) while using hard quantization (like real codec).

### Better Soft Quantization
Current: `tanh(x / (q * temp)) * q`

Better: Use sigmoid-based soft rounding:
```python
def soft_round(x, temperature=0.1):
    # Approximate round() with sigmoid
    frac = x - torch.floor(x)
    soft_frac = torch.sigmoid((frac - 0.5) / temperature)
    return torch.floor(x) + soft_frac

quantized = soft_round(dct / q) * q
```

As temperature → 0, this approaches hard round().

## Recommended Path Forward

**Option A: Fast but risky** - Fix differentiable codec with straight-through estimator or better soft quantization, retrain

**Option B: Slow but guaranteed** - Use CMA-ES or similar zeroth-order optimizer with REAL H.264 evaluation

**My recommendation: Start with Option A** (fix codec), validate on real H.264 frequently during training. If still fails, fall back to Option B.
