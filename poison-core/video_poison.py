#!/usr/bin/env python3
"""
Video Poisoning Implementation
Novel application of radioactive marking to video via optical flow perturbation

Key Innovation: Poison the MOTION between frames, not the pixels themselves.
This survives video compression because motion vectors are encoded separately.

Based on radioactive data marking (Sablayrolles et al., 2020)
Extended to temporal domain for video protection
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, List
import json
import secrets
import hashlib
from tqdm import tqdm

from radioactive_poison import RadioactiveMarker


class VideoRadioactiveMarker:
    """
    Poison video by injecting signature into optical flow (motion vectors).

    Why this works:
    - Video codecs (H.264, AV1) compress motion separately from pixels
    - Small perturbations in motion create "impossible physics"
    - AI models learn these motion patterns
    - Signature survives compression and frame drops
    """

    def __init__(
        self,
        epsilon: float = 0.02,
        temporal_period: int = 30,  # Frames per signature cycle
        device: str = 'cpu'
    ):
        """
        Args:
            epsilon: Perturbation strength for optical flow (higher than images)
            temporal_period: Number of frames for signature pattern to repeat
            device: 'cpu' or 'cuda'
        """
        self.epsilon = epsilon
        self.temporal_period = temporal_period
        self.device = device

        # Use image marker for spatial signature
        self.image_marker = RadioactiveMarker(epsilon=epsilon, device=device)

        # Temporal signature (cyclic pattern across frames)
        self.temporal_signature = None
        self.seed = None

    def generate_signature(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a temporal signature pattern.

        This creates a cyclic pattern that repeats every N frames,
        making it robust to frame drops and compression.
        """
        if seed is None:
            seed = secrets.randbits(256)

        self.seed = seed

        # Generate spatial signature
        self.image_marker.generate_signature(seed=seed)

        # Generate temporal modulation (sine wave pattern)
        t = np.linspace(0, 2 * np.pi, self.temporal_period)
        temporal_modulation = np.sin(t)

        self.temporal_signature = temporal_modulation

        return self.temporal_signature

    def save_signature(self, output_path: str):
        """Save video signature including temporal pattern."""
        if self.temporal_signature is None:
            raise ValueError("No signature generated. Call generate_signature() first.")

        data = {
            'seed': int(self.seed),
            'epsilon': float(self.epsilon),
            'temporal_period': int(self.temporal_period),
            'temporal_signature': self.temporal_signature.tolist(),
            'spatial_signature': self.image_marker.signature.tolist(),
            'type': 'video',
            'version': '1.0'
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_signature(self, signature_path: str):
        """Load a previously generated video signature."""
        with open(signature_path, 'r') as f:
            data = json.load(f)

            self.seed = data['seed']
            self.epsilon = data['epsilon']
            self.temporal_period = data['temporal_period']
            self.temporal_signature = np.array(data['temporal_signature'])

            # Load spatial signature into image marker
            self.image_marker.signature = np.array(data['spatial_signature'])
            self.image_marker.seed = self.seed
            self.image_marker.epsilon = self.epsilon

    def extract_optical_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Extract optical flow between two frames using Farneback algorithm.

        Returns:
            flow: (H, W, 2) array of (dx, dy) motion vectors
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        return flow

    def perturb_optical_flow(
        self,
        flow: np.ndarray,
        frame_idx: int
    ) -> np.ndarray:
        """
        Perturb optical flow with temporal signature.

        Args:
            flow: (H, W, 2) optical flow field
            frame_idx: Current frame index for temporal modulation

        Returns:
            Perturbed flow field
        """
        # Get temporal modulation for this frame
        temporal_idx = frame_idx % self.temporal_period
        temporal_weight = self.temporal_signature[temporal_idx]

        # Generate spatial pattern from signature
        # Use hash of signature to create pseudo-random but deterministic pattern
        h, w = flow.shape[:2]

        # Create spatial pattern based on signature
        # Hash full seed + frame_idx down to 32-bit for NumPy compatibility
        import hashlib
        seed_combined = str(self.seed + frame_idx).encode()
        seed_32bit = int(hashlib.sha256(seed_combined).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed_32bit)
        spatial_pattern = rng.randn(h, w, 2)

        # Normalize spatial pattern
        spatial_pattern = spatial_pattern / (np.linalg.norm(spatial_pattern) + 1e-8)

        # Apply perturbation
        perturbation = self.epsilon * temporal_weight * spatial_pattern
        flow_poisoned = flow + perturbation

        return flow_poisoned

    def reconstruct_frame_from_flow(
        self,
        frame1: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct frame2 by warping frame1 according to optical flow.

        This applies the poisoned motion to generate the output frame.
        """
        h, w = flow.shape[:2]

        # Create coordinate grid
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Apply flow to coordinates
        map_x = (x + flow[:, :, 0]).astype(np.float32)
        map_y = (y + flow[:, :, 1]).astype(np.float32)

        # Warp frame using remap - both maps must be float32
        frame2_reconstructed = cv2.remap(
            frame1,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return frame2_reconstructed

    def poison_video(
        self,
        input_video_path: str,
        output_video_path: str,
        method: str = 'optical_flow'
    ) -> Tuple[str, dict]:
        """
        Poison a video file.

        Args:
            input_video_path: Path to input video
            output_video_path: Path to save poisoned video
            method: 'optical_flow' (motion poisoning) or 'frame' (per-frame image poisoning)

        Returns:
            Tuple of (output_path, metadata)
        """
        if self.temporal_signature is None:
            raise ValueError("No signature loaded. Call generate_signature() or load_signature().")

        print(f"Poisoning video: {input_video_path}")
        print(f"Method: {method}")
        print(f"Epsilon: {self.epsilon}")

        # Open input video
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if method == 'optical_flow':
            success = self._poison_video_optical_flow(cap, out, total_frames)
        elif method == 'frame':
            success = self._poison_video_per_frame(cap, out, total_frames)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'optical_flow' or 'frame'")

        # Cleanup
        cap.release()
        out.release()

        # Generate metadata
        metadata = {
            'original': str(input_video_path),
            'poisoned': str(output_video_path),
            'method': method,
            'epsilon': self.epsilon,
            'temporal_period': self.temporal_period,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'total_frames': total_frames,
            'signature_id': hashlib.sha256(str(self.seed).encode()).hexdigest()[:16]
        }

        print(f"✅ Poisoned video saved to {output_video_path}")

        return output_video_path, metadata

    def _poison_video_optical_flow(
        self,
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        total_frames: int
    ) -> bool:
        """
        Poison video using optical flow perturbation.

        This is the novel approach - perturbs motion vectors.
        """
        ret, prev_frame = cap.read()
        if not ret:
            return False

        # Write first frame unchanged
        out.write(prev_frame)

        frame_idx = 1
        pbar = tqdm(total=total_frames - 1, desc="Poisoning frames")

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Extract optical flow
            flow = self.extract_optical_flow(prev_frame, curr_frame)

            # Perturb flow with temporal signature
            flow_poisoned = self.perturb_optical_flow(flow, frame_idx)

            # Reconstruct frame from poisoned flow
            poisoned_frame = self.reconstruct_frame_from_flow(prev_frame, flow_poisoned)

            # Blend with original to reduce visible artifacts
            alpha = 0.95  # 95% poisoned, 5% original
            final_frame = cv2.addWeighted(poisoned_frame, alpha, curr_frame, 1 - alpha, 0)

            # Write frame
            out.write(final_frame.astype(np.uint8))

            prev_frame = curr_frame
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        return True

    def _poison_video_per_frame(
        self,
        cap: cv2.VideoCapture,
        out: cv2.VideoWriter,
        total_frames: int
    ) -> bool:
        """
        Poison video by applying image poisoning to each frame.

        This is the simpler approach - uses existing image poisoning.
        Less robust to compression but easier to implement.
        """
        import tempfile
        from PIL import Image

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Poisoning frames")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Save temporarily
                temp_input = tmpdir / f"frame_{frame_idx:06d}_in.jpg"
                temp_output = tmpdir / f"frame_{frame_idx:06d}_out.jpg"
                pil_image.save(temp_input)

                # Poison using image marker
                self.image_marker.poison_image(str(temp_input), str(temp_output))

                # Load poisoned frame
                poisoned_pil = Image.open(temp_output)
                poisoned_rgb = np.array(poisoned_pil)
                poisoned_bgr = cv2.cvtColor(poisoned_rgb, cv2.COLOR_RGB2BGR)

                # Write frame
                out.write(poisoned_bgr)

                frame_idx += 1
                pbar.update(1)

        pbar.close()
        return True


class VideoRadioactiveDetector:
    """
    Detect if a video model was trained on poisoned videos.
    """

    def __init__(self, signature_path: str, device: str = 'cpu'):
        """
        Args:
            signature_path: Path to video signature JSON
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Load signature
        with open(signature_path, 'r') as f:
            data = json.load(f)

            if data.get('type') != 'video':
                raise ValueError("Signature is not for video (use RadioactiveDetector for images)")

            self.spatial_signature = np.array(data['spatial_signature'])
            self.temporal_signature = np.array(data['temporal_signature'])
            self.seed = data['seed']
            self.epsilon = data['epsilon']
            self.temporal_period = data['temporal_period']

    def detect(
        self,
        model: nn.Module,
        test_videos: List[str],
        threshold: float = 0.15,
        method: str = 'auto'
    ) -> Tuple[bool, float]:
        """
        Detect if a video model was trained on poisoned data.

        This implements THREE detection strategies:
        1. Spatial feature correlation (baseline)
        2. Temporal feature correlation (novel - tests cyclic pattern)
        3. Behavioral test (black-box - tests model response to synthetic motion)

        Args:
            model: Video model (e.g., 3D CNN, video transformer)
            test_videos: List of test video paths (clean videos)
            threshold: Detection threshold
            method: 'spatial', 'temporal', 'behavioral', or 'auto' (tries all)

        Returns:
            Tuple of (is_poisoned, confidence_score)
        """
        model.eval()
        model.to(self.device)

        if method == 'auto':
            # Try all methods and use the maximum correlation
            print("🔍 Testing multiple detection strategies...")
            spatial_poisoned, spatial_score = self._detect_spatial(model, test_videos, threshold)
            temporal_poisoned, temporal_score = self._detect_temporal(model, test_videos, threshold)

            # Use the best result
            if temporal_score > spatial_score:
                print(f"   Best method: Temporal (score={temporal_score:.6f})")
                return temporal_poisoned, temporal_score
            else:
                print(f"   Best method: Spatial (score={spatial_score:.6f})")
                return spatial_poisoned, spatial_score

        elif method == 'spatial':
            return self._detect_spatial(model, test_videos, threshold)
        elif method == 'temporal':
            return self._detect_temporal(model, test_videos, threshold)
        elif method == 'behavioral':
            return self._detect_behavioral(model, test_videos, threshold)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _detect_spatial(
        self,
        model: nn.Module,
        test_videos: List[str],
        threshold: float
    ) -> Tuple[bool, float]:
        """
        Strategy 1: Spatial Feature Correlation

        Tests if model's spatial features correlate with our spatial signature.
        This is the baseline approach from image poisoning.
        """
        correlations = []

        with torch.no_grad():
            for video_path in tqdm(test_videos, desc="[Spatial] Detecting signature"):
                # Extract features from video using the model
                features = self._extract_video_features(model, video_path)

                if features is None:
                    continue

                # Compute correlation with spatial signature
                features_np = features.cpu().numpy().flatten()
                signature_np = self.spatial_signature.flatten()

                # Normalize both vectors
                features_norm = features_np / (np.linalg.norm(features_np) + 1e-8)
                signature_norm = signature_np / (np.linalg.norm(signature_np) + 1e-8)

                # Compute correlation
                correlation = np.dot(features_norm, signature_norm)
                correlations.append(correlation)

        if not correlations:
            return False, 0.0

        # Average correlation across test videos
        avg_correlation = float(np.mean(correlations))
        is_poisoned = avg_correlation > threshold

        return is_poisoned, avg_correlation

    def _detect_temporal(
        self,
        model: nn.Module,
        test_videos: List[str],
        threshold: float
    ) -> Tuple[bool, float]:
        """
        Strategy 2: Temporal Feature Correlation (NOVEL)

        Tests if model's temporal features exhibit the cyclic pattern
        of our temporal signature. This is the key innovation for video.

        How it works:
        1. Extract per-frame features from model
        2. Compute correlation with temporal signature across time
        3. Check for cyclic pattern matching our period
        """
        temporal_correlations = []

        with torch.no_grad():
            for video_path in tqdm(test_videos, desc="[Temporal] Detecting cyclic signature"):
                # Extract features for EACH frame separately
                frame_features = self._extract_temporal_features(model, video_path)

                if frame_features is None or len(frame_features) < self.temporal_period:
                    continue

                # Compute per-frame correlations with spatial signature
                frame_correlations = []
                for feat in frame_features:
                    feat_np = feat.cpu().numpy().flatten()
                    sig_np = self.spatial_signature.flatten()

                    # Normalize
                    feat_norm = feat_np / (np.linalg.norm(feat_np) + 1e-8)
                    sig_norm = sig_np / (np.linalg.norm(sig_np) + 1e-8)

                    # Correlation
                    corr = np.dot(feat_norm, sig_norm)
                    frame_correlations.append(corr)

                # Now check if frame_correlations follow our temporal_signature pattern
                # Use cross-correlation to detect cyclic pattern
                temporal_corr = self._measure_temporal_correlation(
                    frame_correlations,
                    self.temporal_signature
                )
                temporal_correlations.append(temporal_corr)

        if not temporal_correlations:
            return False, 0.0

        # Average temporal correlation
        avg_temporal_corr = float(np.mean(temporal_correlations))
        is_poisoned = avg_temporal_corr > threshold

        return is_poisoned, avg_temporal_corr

    def _detect_behavioral(
        self,
        model: nn.Module,
        test_videos: List[str],
        threshold: float
    ) -> Tuple[bool, float]:
        """
        Strategy 3: Behavioral Test (Black-Box)

        Tests if model exhibits different behavior on videos with
        our signature motion pattern vs random motion.

        This works even if we can't extract features directly.
        """
        # Generate synthetic test videos with signature motion
        print("Generating synthetic test videos with signature motion...")
        signature_video = self._generate_synthetic_video_with_signature()
        random_video = self._generate_synthetic_video_random()

        with torch.no_grad():
            # Get model predictions
            sig_features = self._extract_video_features(model, signature_video)
            rand_features = self._extract_video_features(model, random_video)

            if sig_features is None or rand_features is None:
                return False, 0.0

            # Measure difference in model response
            sig_np = sig_features.cpu().numpy().flatten()
            rand_np = rand_features.cpu().numpy().flatten()

            # Normalize
            sig_norm = sig_np / (np.linalg.norm(sig_np) + 1e-8)
            rand_norm = rand_np / (np.linalg.norm(rand_np) + 1e-8)

            # Compute divergence
            # If model learned our signature, responses should be different
            divergence = np.linalg.norm(sig_norm - rand_norm)

        # Clean up temp videos
        import os
        os.remove(signature_video)
        os.remove(random_video)

        is_poisoned = divergence > threshold
        return is_poisoned, float(divergence)

    def _extract_temporal_features(
        self,
        model: nn.Module,
        video_path: str,
        num_frames: int = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Extract features for EACH frame separately (for temporal correlation).

        This is different from _extract_video_features which processes
        the entire video clip at once. Here we want per-frame features
        to detect temporal patterns.

        Returns:
            List of feature tensors, one per frame
        """
        if num_frames is None:
            num_frames = self.temporal_period * 2  # Extract at least 2 cycles

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return None

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frame_features = []

            # Process frames in sliding windows
            window_size = 16  # Model expects 16-frame clips
            stride = 1  # Slide by 1 frame to get per-frame features

            frames_buffer = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (112, 112))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_buffer.append(frame)

                # When buffer is full, extract features
                if len(frames_buffer) >= window_size:
                    # Convert to tensor
                    video_tensor = np.stack(frames_buffer[-window_size:], axis=0)
                    video_tensor = torch.from_numpy(video_tensor).float()
                    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
                    video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
                    video_tensor = video_tensor / 255.0
                    video_tensor = video_tensor.to(self.device)

                    # Extract features
                    with torch.no_grad():
                        if hasattr(model, 'extract_features'):
                            feat = model.extract_features(video_tensor)
                        else:
                            feat = model(video_tensor)

                        # Take the features corresponding to the center frame
                        frame_features.append(feat)

            cap.release()

            return frame_features if len(frame_features) > 0 else None

        except Exception as e:
            print(f"Error extracting temporal features from {video_path}: {e}")
            return None

    def _measure_temporal_correlation(
        self,
        frame_correlations: List[float],
        temporal_signature: np.ndarray
    ) -> float:
        """
        Measure how well frame correlations match the temporal signature pattern.

        Uses cross-correlation to detect cyclic patterns.

        Args:
            frame_correlations: List of per-frame correlation scores
            temporal_signature: Expected temporal pattern (sine wave)

        Returns:
            Correlation score (higher = better match)
        """
        # Convert to numpy array
        frame_corr_array = np.array(frame_correlations)

        # Compute cross-correlation with signature
        # This detects if the pattern repeats with our period
        period = len(temporal_signature)

        # Tile the signature to match frame_correlations length
        num_tiles = int(np.ceil(len(frame_corr_array) / period))
        signature_tiled = np.tile(temporal_signature, num_tiles)[:len(frame_corr_array)]

        # Normalize both
        frame_norm = frame_corr_array / (np.linalg.norm(frame_corr_array) + 1e-8)
        sig_norm = signature_tiled / (np.linalg.norm(signature_tiled) + 1e-8)

        # Compute correlation
        correlation = np.dot(frame_norm, sig_norm)

        return float(correlation)

    def _generate_synthetic_video_with_signature(self) -> str:
        """
        Generate a synthetic test video with our signature motion pattern.

        This creates a simple scene with moving objects that have
        motion vectors matching our temporal signature.

        Returns:
            Path to temporary video file
        """
        import tempfile

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        import os
        os.close(temp_fd)

        # Generate simple moving pattern video
        width, height = 112, 112
        fps = 30
        duration_frames = self.temporal_period * 2  # 2 cycles

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        for frame_idx in range(duration_frames):
            # Create frame with moving circle
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Get temporal modulation
            temporal_idx = frame_idx % self.temporal_period
            temporal_weight = self.temporal_signature[temporal_idx]

            # Move circle according to signature
            x = int(width / 2 + 20 * temporal_weight)
            y = int(height / 2)

            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
            out.write(frame)

        out.release()
        return temp_path

    def _generate_synthetic_video_random(self) -> str:
        """
        Generate a synthetic test video with random motion (control).

        Returns:
            Path to temporary video file
        """
        import tempfile

        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        import os
        os.close(temp_fd)

        width, height = 112, 112
        fps = 30
        duration_frames = self.temporal_period * 2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        positions = rng.randint(20, width - 20, size=duration_frames)

        for frame_idx in range(duration_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            x = int(positions[frame_idx])
            y = int(height / 2)

            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
            out.write(frame)

        out.release()
        return temp_path

    def _extract_video_features(
        self,
        model: nn.Module,
        video_path: str,
        num_frames: int = 16
    ) -> Optional[torch.Tensor]:
        """
        Extract features from a video using the provided model.

        Args:
            model: Video model (3D CNN or similar)
            video_path: Path to video file
            num_frames: Number of frames to sample

        Returns:
            Feature tensor or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return None

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to 112x112 (common for video models)
                    frame = cv2.resize(frame, (112, 112))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if len(frames) < num_frames:
                return None

            # Convert to tensor (T, H, W, C) -> (1, C, T, H, W)
            video_tensor = np.stack(frames, axis=0)  # (T, H, W, C)
            video_tensor = torch.from_numpy(video_tensor).float()
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
            video_tensor = video_tensor / 255.0  # Normalize
            video_tensor = video_tensor.to(self.device)

            # Extract features using model
            # Try to get intermediate features if available
            if hasattr(model, 'extract_features'):
                features = model.extract_features(video_tensor)
            else:
                # Use model output directly
                features = model(video_tensor)

            return features

        except Exception as e:
            print(f"Error extracting features from {video_path}: {e}")
            return None


if __name__ == "__main__":
    print("Video Radioactive Poison - Core Implementation")
    print("=" * 60)

    # Example usage
    marker = VideoRadioactiveMarker(epsilon=0.02, temporal_period=30, device='cpu')

    # Generate signature
    signature = marker.generate_signature()
    print(f"Generated temporal signature with {len(signature)} frames")

    # Save signature
    marker.save_signature("video_signature.json")
    print("Signature saved to video_signature.json")

    print("\nReady to poison videos!")
    print("Use the CLI: python video_poison_cli.py")
