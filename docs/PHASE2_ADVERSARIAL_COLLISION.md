# Phase 2: Adversarial Perceptual Hash Collision

**Status:** ✅ **COMPLETED**
**Prerequisite:** ✅ Hash stability proven (0-14 bit drift at CRF 28)

---

## The Plan

### Proven Foundation

**Phase 1 Results:**
- Perceptual hash survives CRF 28 compression ✅
- Hamming distance: 0-14 bits out of 256 (0-5% drift)
- Real videos (UCF-101): 4-14 bit drift
- Synthetic videos: 0-6 bit drift

**This means:** If we can make a video's hash match our signature, it will still match after compression!

---

## Implementation

### Step 1: PGD Attack for Hash Collision (2-3 hours)

**Goal:** Optimize video perturbation to make perceptual hash = target signature

```python
# experiments/adversarial_perceptual_poison.py

import torch
import numpy as np
from perceptual_hash import extract_perceptual_features, compute_perceptual_hash

def poison_video_pgd(
    video_frames: list,
    target_hash: np.ndarray,
    epsilon: float = 0.05,
    num_iterations: int = 100,
    step_size: float = 0.01
):
    """
    Use PGD to perturb video frames until hash matches target.

    Args:
        video_frames: List of frames (H, W, 3) BGR
        target_hash: 256-bit target signature
        epsilon: Max L_inf perturbation (default 0.05 = 12.75/255)
        num_iterations: PGD steps
        step_size: Learning rate

    Returns:
        poisoned_frames: Perturbed frames
        final_distance: Hamming distance to target
    """
    # Convert to torch
    frames_torch = torch.tensor(
        np.array(video_frames),
        dtype=torch.float32
    ).requires_grad_(True)

    target_hash_torch = torch.tensor(target_hash, dtype=torch.float32)

    optimizer = torch.optim.Adam([frames_torch], lr=step_size)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Extract perceptual hash (differentiable approximation)
        current_hash = compute_hash_differentiable(frames_torch)

        # Loss: Hamming distance to target
        # Use BCE as differentiable approximation
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            current_hash, target_hash_torch
        )

        loss.backward()
        optimizer.step()

        # Project to epsilon ball
        with torch.no_grad():
            # Clamp perturbation
            perturbation = frames_torch - frames_original
            perturbation = torch.clamp(perturbation, -epsilon * 255, epsilon * 255)
            frames_torch.data = frames_original + perturbation
            frames_torch.data = torch.clamp(frames_torch.data, 0, 255)

        if (iteration + 1) % 10 == 0:
            dist = hamming_distance(
                (current_hash > 0).cpu().numpy(),
                target_hash
            )
            print(f"Iteration {iteration+1}: Hamming distance = {dist}/256")

    poisoned_frames = frames_torch.detach().cpu().numpy().astype(np.uint8)
    return poisoned_frames
```

**Challenge:** Need to make feature extraction differentiable

### Step 2: Differentiable Feature Extraction (1-2 hours)

**Problem:** `cv2.Canny`, `cv2.Laplacian`, `cv2.getGaborKernel` are not differentiable

**Solution:** Use PyTorch-based approximations

```python
import torch
import torch.nn.functional as F

def extract_features_differentiable(frames_torch):
    """
    Differentiable approximation of perceptual features.

    Args:
        frames_torch: (N, H, W, 3) torch tensor

    Returns:
        features: dict of differentiable features
    """
    features = {}

    for frame_idx, frame in enumerate(frames_torch):
        # Convert to grayscale for edge/saliency
        gray = 0.299*frame[:,:,2] + 0.587*frame[:,:,1] + 0.114*frame[:,:,0]

        # 1. Edges (differentiable Sobel)
        sobel_x = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            padding=1
        )
        sobel_y = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            padding=1
        )
        edges = torch.sqrt(sobel_x**2 + sobel_y**2)

        # 2. Textures (differentiable Gabor - approximate with learned filters)
        # For simplicity, use random conv filters
        textures = []
        for theta in [0, 45, 90, 135]:
            # Simplified: use Sobel at different angles
            kernel = create_gabor_kernel_torch(theta)
            texture = F.conv2d(
                gray.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=10
            )
            textures.append(texture)
        textures = torch.stack(textures)

        # 3. Saliency (Laplacian)
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        saliency = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            laplacian_kernel,
            padding=1
        )

        # 4. Color histogram (differentiable soft binning)
        hist_r = soft_histogram(frame[:,:,0], bins=32)
        hist_g = soft_histogram(frame[:,:,1], bins=32)
        hist_b = soft_histogram(frame[:,:,2], bins=32)
        color_hist = torch.cat([hist_r, hist_g, hist_b])

        features[frame_idx] = {
            'edges': edges.squeeze(),
            'textures': textures.squeeze(),
            'saliency': saliency.squeeze(),
            'color_hist': color_hist
        }

    return features

def soft_histogram(values, bins=32, sigma=1.0):
    """Differentiable soft histogram using Gaussian kernels."""
    bin_centers = torch.linspace(0, 255, bins)
    dists = (values.unsqueeze(-1) - bin_centers) ** 2
    weights = torch.exp(-dists / (2 * sigma**2))
    hist = weights.sum(dim=(0,1))
    return hist / (hist.sum() + 1e-8)
```

### Step 3: End-to-End Validation (1 hour)

**Pipeline:**
```python
# 1. Generate target signature
target_hash = np.random.randint(0, 2, 256)

# 2. Load clean video
video = load_video_frames('test.mp4', max_frames=30)

# 3. Poison to match signature
poisoned = poison_video_pgd(video, target_hash, epsilon=0.05)

# 4. Save poisoned video
save_video(poisoned, 'poisoned.mp4')

# 5. Compress with CRF 28
compress_h264('poisoned.mp4', 'compressed.mp4', crf=28)

# 6. Extract hash from compressed
compressed = load_video_frames('compressed.mp4')
features_comp = extract_perceptual_features(compressed)
hash_comp = compute_perceptual_hash(features_comp)

# 7. Check if hash survived
distance = hamming_distance(hash_comp, target_hash)

# SUCCESS if distance < 20 (based on Phase 1 results)
print(f"Hamming distance after CRF 28: {distance}/256")
print(f"Success: {distance < 20}")
```

---

## Expected Results

### Hypothesis

**If we can get Hamming distance < 10 before compression:**
- Phase 1 showed drift of 0-14 bits
- After compression: distance < 10 + 14 = 24 bits
- **Detection threshold:** 30 bits
- **TPR:** >90% (poisoned videos detected)
- **FPR:** ~0% (random videos won't match signature)

### Risk Mitigation

**If PGD doesn't converge well:**
- Increase num_iterations (100 → 500)
- Use CMA-ES instead (gradient-free, proven to work)
- Accept larger epsilon (0.05 → 0.08)

**If hash drifts too much:**
- Use error-correcting codes (BCH, Reed-Solomon)
- Increase hash size (256 → 512 bits)
- Target multiple hashes (ensemble)

---

## Timeline

| Task | Effort | Status |
|------|--------|--------|
| **Step 1:** PGD framework | 2-3 hours | Pending |
| **Step 2:** Differentiable features | 1-2 hours | Pending |
| **Step 3:** End-to-end test | 1 hour | Pending |
| **Total** | **4-6 hours** | - |

---

## Success Criteria

### Minimum Viable

- [ ] Poison video → hash matches signature (distance < 10)
- [ ] Compress CRF 28 → hash still matches (distance < 30)
- [ ] Test on 10 videos → 8+ succeed (TPR > 80%)
- [ ] 10 clean videos → 0 match (FPR = 0%)

### Stretch Goals

- [ ] Works with epsilon = 0.03 (lower perturbation)
- [ ] Hash drift < 10 bits (very stable)
- [ ] Works on UCF-101 real videos
- [ ] PSNR > 35 dB (imperceptible)

---

## Why This Will Work

**Unlike DCT approach:**
1. ✅ Not fighting codec design (perceptual features preserved)
2. ✅ Hash stability proven empirically (0-14 bit drift)
3. ✅ Standard adversarial attack (PGD is well-understood)
4. ✅ No codec approximation needed (hash is black-box)

**The breakthrough insight:**
> Codec preserves what humans see → perceptual features → hash stability

**DCT failed because:**
> Codec destroys exact coefficient values → frequency domain unstable

---

## Implementation Files

```
core/
├── adversarial.py                ✅ Done (PGD engine)
├── perceptual_hash.py            ✅ Done (hash extraction)

cli/
└── poison.py                     ✅ Done (CLI Tool)

experiments/
└── batch_hash_robustness.py      ✅ Done (stability testing)
```

---

## Next Steps

1. **Implement differentiable feature extraction** (highest priority)
2. **Implement PGD framework**
3. **Run end-to-end test on 1 video**
4. **If works:** Scale to 20+ videos, run statistics
5. **If fails:** Debug, adjust hyperparameters, iterate

**Ready to start coding!** 🚀
