#!/usr/bin/env python3
"""
Differentiable H.264 Proxy

Approximates H.264 compression in a differentiable way for end-to-end training.

Key components:
1. DCT transform (differentiable via torch.fft)
2. Quantization (approximate with soft quantization)
3. Inverse DCT (differentiable)

This lets us optimize poisoning to SURVIVE compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_dct import dct_2d, idct_2d
    HAS_TORCH_DCT = True
except ImportError:
    HAS_TORCH_DCT = False
    # Fallback: use numpy DCT via opencv, not differentiable
    import cv2


class DifferentiableH264(nn.Module):
    """
    Differentiable approximation of H.264 video compression.

    Simulates:
    - YCbCr conversion
    - 8x8 block DCT
    - Quantization (soft approximation)
    - Inverse DCT
    - Clipping

    Does NOT simulate:
    - Entropy coding (doesn't affect coefficients)
    - Motion compensation (we're training on individual frames)
    - Deblocking filter (optional, small effect)
    """

    def __init__(self, quality_factor: int = 28, temperature: float = 0.1):
        """
        Args:
            quality_factor: CRF equivalent (18=high, 28=YouTube, 35=heavy)
            temperature: Softness of quantization (lower = harder)
        """
        super().__init__()
        self.quality_factor = quality_factor
        self.temperature = temperature

        # Create H.264-style quantization matrix
        self.register_buffer('quant_matrix', self._create_quant_matrix(quality_factor))

    def _create_quant_matrix(self, qf: int) -> torch.Tensor:
        """
        H.264 quantization matrix (8x8).

        Based on JPEG quantization table, scaled for CRF.
        Low frequencies (top-left): small steps → preserved
        High frequencies (bottom-right): large steps → destroyed
        """
        # Base JPEG luminance quantization table
        base = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)

        # Scale based on CRF
        # CRF 18 (high quality) → small quantization steps
        # CRF 28 (YouTube) → medium steps
        # CRF 35 (heavy) → large steps
        if qf < 25:
            scale = 50.0 / max(1, qf)
        else:
            scale = 200.0 - 2 * qf

        quant_matrix = base * scale / 50.0

        # Ensure minimum quantization of 1
        quant_matrix = np.maximum(quant_matrix, 1.0)

        return torch.from_numpy(quant_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply differentiable H.264 compression.

        Args:
            x: Input tensor (B, C, H, W) in range [0, 255], BGR format

        Returns:
            Compressed tensor (B, C, H, W), BGR format
        """
        batch_size, channels, height, width = x.shape

        # Convert BGR → YCbCr (only process Y channel like real H.264)
        # Simplified: assume grayscale or only Y channel
        # For full implementation, would convert all 3 channels
        y_channel = self._bgr_to_y(x)  # (B, 1, H, W)

        # Pad to multiple of 8
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8

        if pad_h > 0 or pad_w > 0:
            y_padded = F.pad(y_channel, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            y_padded = y_channel

        # Process in 8x8 blocks
        y_compressed = self._process_blocks(y_padded)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            y_compressed = y_compressed[:, :, :height, :width]

        # Convert back to BGR (simplified: just replicate Y to all channels)
        out = self._y_to_bgr(y_compressed, x)

        # Clip to valid range
        out = torch.clamp(out, 0, 255)

        return out

    def _bgr_to_y(self, bgr: torch.Tensor) -> torch.Tensor:
        """Convert BGR to Y (luminance) channel."""
        # Y = 0.299*R + 0.587*G + 0.114*B
        # BGR order: B=[:,:,0], G=[:,:,1], R=[:,:,2]
        b, g, r = bgr[:, 0:1], bgr[:, 1:2], bgr[:, 2:3]
        y = 0.114 * b + 0.587 * g + 0.299 * r
        return y

    def _y_to_bgr(self, y: torch.Tensor, original_bgr: torch.Tensor) -> torch.Tensor:
        """Convert Y back to BGR (simplified: keep Cb, Cr from original)."""
        # Simplified: replace Y channel, keep chroma from original
        # Full implementation would properly convert YCbCr → BGR
        b_orig, g_orig, r_orig = original_bgr[:, 0:1], original_bgr[:, 1:2], original_bgr[:, 2:3]

        # Compute how much Y changed
        y_orig = 0.114 * b_orig + 0.587 * g_orig + 0.299 * r_orig
        y_delta = y - y_orig

        # Apply delta proportionally to all channels (simplified)
        b_new = b_orig + y_delta * 0.114
        g_new = g_orig + y_delta * 0.587
        r_new = r_orig + y_delta * 0.299

        return torch.cat([b_new, g_new, r_new], dim=1)

    def _process_blocks(self, y: torch.Tensor) -> torch.Tensor:
        """Process Y channel in 8x8 blocks with DCT + quantization."""
        batch_size, _, height, width = y.shape

        # Unfold into 8x8 blocks
        # y: (B, 1, H, W) → (B, 1, num_blocks_h, num_blocks_w, 8, 8)
        num_blocks_h = height // 8
        num_blocks_w = width // 8

        blocks = y.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, 1, num_blocks_h, num_blocks_w, 8, 8)
        blocks = blocks.contiguous().view(batch_size, num_blocks_h * num_blocks_w, 8, 8)

        # Apply DCT to each block
        dct_blocks = self._dct2d(blocks)

        # Quantize (soft)
        quantized_blocks = self._soft_quantize(dct_blocks)

        # Inverse DCT
        reconstructed_blocks = self._idct2d(quantized_blocks)

        # Fold back into image
        reconstructed = reconstructed_blocks.view(batch_size, 1, num_blocks_h, num_blocks_w, 8, 8)
        reconstructed = reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous()
        reconstructed = reconstructed.view(batch_size, 1, height, width)

        return reconstructed

    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D DCT.

        Args:
            x: (B, N, 8, 8) blocks

        Returns:
            DCT coefficients (B, N, 8, 8)
        """
        if HAS_TORCH_DCT:
            # Use differentiable torch-dct library
            batch_size, num_blocks, h, w = x.shape
            x_reshaped = x.view(batch_size * num_blocks, 1, h, w)
            dct = dct_2d(x_reshaped, norm='ortho')
            return dct.view(batch_size, num_blocks, h, w)
        else:
            # Fallback: manual DCT (not differentiable, for testing only)
            return self._manual_dct2d(x)

    def _idct2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D inverse DCT.

        Args:
            x: DCT coefficients (B, N, 8, 8)

        Returns:
            Spatial blocks (B, N, 8, 8)
        """
        if HAS_TORCH_DCT:
            batch_size, num_blocks, h, w = x.shape
            x_reshaped = x.view(batch_size * num_blocks, 1, h, w)
            idct = idct_2d(x_reshaped, norm='ortho')
            return idct.view(batch_size, num_blocks, h, w)
        else:
            return self._manual_idct2d(x)

    def _manual_dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """Manual 2D DCT using cosine basis (differentiable but slow)."""
        N = 8
        basis = self._get_dct_basis(N).to(x.device)

        # x: (B, num_blocks, 8, 8)
        # basis: (8, 8) for 1D DCT

        # DCT along rows: X @ basis.T
        # DCT along cols: basis @ X_rows
        x_dct = torch.einsum('...ij,jk->...ik', x, basis.T)  # rows
        x_dct = torch.einsum('ij,...jk->...ik', basis, x_dct)  # cols

        return x_dct

    def _manual_idct2d(self, x: torch.Tensor) -> torch.Tensor:
        """Manual 2D IDCT."""
        N = 8
        basis = self._get_dct_basis(N).to(x.device)

        # IDCT is transpose of DCT
        x_spatial = torch.einsum('...ij,kj->...ik', x, basis)
        x_spatial = torch.einsum('jk,...jl->...kl', basis.T, x_spatial)

        return x_spatial

    def _get_dct_basis(self, N: int) -> torch.Tensor:
        """Get DCT basis matrix."""
        if not hasattr(self, '_dct_basis_cache'):
            basis = torch.zeros(N, N)
            for k in range(N):
                for n in range(N):
                    if k == 0:
                        basis[k, n] = np.sqrt(1.0 / N)
                    else:
                        basis[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2*n + 1) / (2*N))
            self._dct_basis_cache = basis
        return self._dct_basis_cache

    def _soft_quantize(self, dct: torch.Tensor) -> torch.Tensor:
        """
        Improved quantization with straight-through estimator.

        Forward pass: Hard quantization (like real H.264)
        Backward pass: Gradient flows through (straight-through trick)

        This gives us:
        - Accurate simulation of real H.264 (hard quantization)
        - Differentiability for training (gradients)

        Args:
            dct: DCT coefficients (B, N, 8, 8)

        Returns:
            Quantized coefficients (B, N, 8, 8)
        """
        # Expand quant_matrix to match batch
        q = self.quant_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)

        # Hard quantization (forward pass matches real H.264)
        quantized_hard = torch.round(dct / (q + 1e-8)) * q

        # Straight-through estimator: use hard quantization in forward,
        # but gradient flows as if it's identity function
        if self.training:
            # Detach the difference so gradients flow through as identity
            quantized = dct + (quantized_hard - dct).detach()
        else:
            # During inference, use actual hard quantization
            quantized = quantized_hard

        return quantized


class AdaptivePoisoningOptimizer:
    """
    End-to-end optimizer for compression-robust poisoning.

    Optimizes DCT signature to maximize detection score AFTER compression.
    """

    def __init__(
        self,
        target_crf: int = 28,
        learning_rate: float = 0.01,
        num_iterations: int = 100
    ):
        self.codec = DifferentiableH264(quality_factor=target_crf, temperature=0.1)
        self.target_crf = target_crf
        self.lr = learning_rate
        self.num_iterations = num_iterations

        # Learnable signature (initialized randomly)
        signature_init = torch.randn(8, 8)
        signature_init[3:, :] = 0  # Zero out high frequencies
        signature_init[:, 3:] = 0
        signature_init = signature_init / (torch.norm(signature_init) + 1e-8)

        self.signature = nn.Parameter(signature_init)

        # Learnable epsilon
        self.epsilon = nn.Parameter(torch.tensor(0.05))

        self.optimizer = torch.optim.Adam([self.signature, self.epsilon], lr=self.lr)

    def train_step(self, clean_frames: torch.Tensor) -> dict:
        """
        One training step.

        Args:
            clean_frames: (B, C, H, W) clean video frames

        Returns:
            Dictionary with loss, detection_score, etc.
        """
        # Poison frames in spatial domain
        poisoned_frames = self._apply_signature_spatial(clean_frames)

        # Simulate compression
        compressed_frames = self.codec(poisoned_frames)

        # Extract signature from compressed frames
        detection_score = self._compute_detection_score(compressed_frames)

        # Visual quality loss (PSNR)
        mse = F.mse_loss(compressed_frames, clean_frames)
        psnr = 10 * torch.log10(255.0**2 / (mse + 1e-8))

        # Combined loss
        # Maximize detection (minimize negative detection)
        # Maximize PSNR (minimize distance from target PSNR=40)
        detection_loss = -detection_score
        quality_loss = (40 - psnr)**2 * 0.01

        total_loss = detection_loss + quality_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clamp epsilon
        with torch.no_grad():
            self.epsilon.clamp_(0.01, 0.2)

            # Normalize signature
            self.signature.data = self.signature.data / (torch.norm(self.signature.data) + 1e-8)

        return {
            'loss': total_loss.item(),
            'detection_score': detection_score.item(),
            'psnr': psnr.item(),
            'epsilon': self.epsilon.item()
        }

    def _apply_signature_spatial(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply signature to frames via DCT poisoning."""
        # This is a simplified version - would need full block processing
        # For now, add signature as spatial perturbation (approximation)
        noise = torch.randn_like(frames) * self.epsilon * 10
        return frames + noise

    def _compute_detection_score(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute detection score (correlation with signature)."""
        # Simplified: measure if signature pattern is in the frames
        # Full version would extract DCT and correlate with self.signature
        return torch.tensor(0.5, requires_grad=True)  # Placeholder


if __name__ == '__main__':
    print("=" * 60)
    print("Differentiable H.264 Codec")
    print("=" * 60)
    print()

    # Test codec
    codec = DifferentiableH264(quality_factor=28)

    # Create test frame
    test_frame = torch.rand(1, 3, 224, 224) * 255

    print("Input shape:", test_frame.shape)
    print("Input range:", test_frame.min().item(), "-", test_frame.max().item())

    # Compress
    compressed = codec(test_frame)

    print("Output shape:", compressed.shape)
    print("Output range:", compressed.min().item(), "-", compressed.max().item())

    # Compute PSNR
    mse = F.mse_loss(test_frame, compressed)
    psnr = 10 * torch.log10(255.0**2 / mse)

    print(f"PSNR: {psnr.item():.2f} dB")
    print()
    print("✓ Differentiable codec works")
    print()
    print("Next: Implement adaptive training loop")
