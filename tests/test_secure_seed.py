
import pytest
import subprocess
import os
import sys
import tempfile
import cv2
import numpy as np
from typing import List, Union
from core.perceptual_hash import compute_perceptual_hash

# Mock features for unit testing
MOCK_FEATURES = {
    0: {
        'edges': np.ones((10, 10)),
        'textures': np.ones((4, 10, 10)),
        'saliency': np.ones((10, 10)),
        'color_hist': np.ones(96)
    }
}

class TestSecureSeedUnit:
    """Unit tests for core.perceptual_hash seed logic"""

    def test_default_seed_consistency(self):
        """Test that default seed matches implicit None"""
        h1 = compute_perceptual_hash(MOCK_FEATURES, seed=None)
        h2 = compute_perceptual_hash(MOCK_FEATURES, seed=42)
        np.testing.assert_array_equal(h1, h2, "Implicit None seed should match explicit 42")

    def test_string_integer_parity(self):
        """Test that string '42' is treated same as int 42 (The Bug Fix)"""
        h_int = compute_perceptual_hash(MOCK_FEATURES, seed=42)
        h_str = compute_perceptual_hash(MOCK_FEATURES, seed="42")
        np.testing.assert_array_equal(h_int, h_str, "String '42' should be parsed as int 42")

    def test_custom_seed_determinism(self):
        """Test that arbitrary string seeds are deterministic"""
        seed = "my-secret-password"
        h1 = compute_perceptual_hash(MOCK_FEATURES, seed=seed)
        h2 = compute_perceptual_hash(MOCK_FEATURES, seed=seed)
        np.testing.assert_array_equal(h1, h2)

    def test_seed_uniqueness(self):
        """Test that different seeds produce different hashes"""
        h_default = compute_perceptual_hash(MOCK_FEATURES, seed=42)
        h_custom = compute_perceptual_hash(MOCK_FEATURES, seed="secret")
        
        # It's statistically nearly impossible for these to match
        assert not np.array_equal(h_default, h_custom), "Custom seed should not collide with default"

class TestSecureSeedCLI:
    """Integration tests calls the CLI directly"""
    
    VIDEO_PATH = "experimental/test_videos/short_test.mp4"
    _temp_video_path = None

    @classmethod
    def setup_class(cls):
        """Ensure video file exists, create dummy if needed"""
        if not os.path.exists(cls.VIDEO_PATH):
            # Create a temporary video file
            fd, path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            cls._temp_video_path = path
            cls.VIDEO_PATH = path
            
            # Generate dummy video content
            height, width = 64, 64
            fps = 10
            # Use 'mp4v' codec which is widely supported
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                # Fallback to creating a dummy empty file if opencv fails (though it shouldn't)
                # But for the test we need valid frames, so this is critical.
                pass
                
            try:
                # 10 frames of random noise
                for _ in range(10):
                    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                    out.write(frame)
            finally:
                out.release()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup temporary video if created"""
        if cls._temp_video_path and os.path.exists(cls._temp_video_path):
            os.unlink(cls._temp_video_path)


    def run_cli(self, args: List[str]) -> str:
        cmd = [sys.executable, "-m", "cli.extract", self.VIDEO_PATH] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        return result.stdout.strip()

    def test_cli_seed_flag(self):
        """Verify --seed flag affects output"""
        # Default
        h_default = self.run_cli([])
        
        # Explicit Default
        h_explicit = self.run_cli(["--seed", "42"])
        assert h_default == h_explicit, "CLI --seed 42 should match default"

        # Custom
        h_custom = self.run_cli(["--seed", "pytest_secret"])
        assert h_default != h_custom, "CLI custom seed should differ from default"
        
        # Determinism
        h_custom_2 = self.run_cli(["--seed", "pytest_secret"])
        assert h_custom == h_custom_2, "CLI custom seed must be deterministic"
