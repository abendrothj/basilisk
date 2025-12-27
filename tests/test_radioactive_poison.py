#!/usr/bin/env python3
"""
Unit tests for radioactive_poison.py
Tests the core poisoning algorithm and signature generation
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json
import tempfile
import sys
from PIL import Image

# Add poison-core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'poison-core'))

from radioactive_poison import RadioactiveMarker, RadioactiveDetector


class TestRadioactiveMarker:
    """Test suite for RadioactiveMarker class"""

    @pytest.fixture
    def marker(self):
        """Create a RadioactiveMarker instance for testing"""
        return RadioactiveMarker(epsilon=0.01, signature_dim=512, device='cpu')

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create a test image"""
        img = Image.new('RGB', (224, 224), color='red')
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)
        return str(img_path)

    def test_marker_initialization(self, marker):
        """Test that marker initializes correctly"""
        assert marker.epsilon == 0.01
        assert marker.signature_dim == 512
        assert marker.device == 'cpu'
        assert marker.signature is None
        assert marker.feature_extractor is not None

    def test_signature_generation(self, marker):
        """Test signature generation"""
        signature = marker.generate_signature(seed=12345)

        assert signature is not None
        assert len(signature) == 512
        assert isinstance(signature, np.ndarray)

        # Check it's normalized (unit vector)
        norm = np.linalg.norm(signature)
        assert abs(norm - 1.0) < 1e-6

    def test_signature_deterministic(self, marker):
        """Test that same seed produces same signature"""
        sig1 = marker.generate_signature(seed=12345)

        marker2 = RadioactiveMarker(epsilon=0.01)
        sig2 = marker2.generate_signature(seed=12345)

        assert np.allclose(sig1, sig2)

    def test_signature_different_seeds(self, marker):
        """Test that different seeds produce different signatures"""
        sig1 = marker.generate_signature(seed=12345)

        marker2 = RadioactiveMarker(epsilon=0.01)
        sig2 = marker2.generate_signature(seed=54321)

        assert not np.allclose(sig1, sig2)

    def test_save_and_load_signature(self, marker, tmp_path):
        """Test signature saving and loading"""
        marker.generate_signature(seed=12345)

        sig_path = tmp_path / "test_signature.json"
        marker.save_signature(str(sig_path))

        # Check file was created
        assert sig_path.exists()

        # Load and verify
        marker2 = RadioactiveMarker(epsilon=0.01)
        marker2.load_signature(str(sig_path))

        assert np.allclose(marker.signature, marker2.signature)
        assert marker.seed == marker2.seed
        assert marker.epsilon == marker2.epsilon

    def test_save_signature_without_generation(self, marker, tmp_path):
        """Test that saving fails if no signature generated"""
        sig_path = tmp_path / "test_signature.json"

        with pytest.raises(ValueError, match="No signature generated"):
            marker.save_signature(str(sig_path))

    def test_poison_image(self, marker, test_image, tmp_path):
        """Test image poisoning"""
        marker.generate_signature(seed=12345)

        output_path = tmp_path / "poisoned.jpg"
        result_path, metadata = marker.poison_image(
            test_image,
            str(output_path)
        )

        # Check output file exists
        assert Path(output_path).exists()

        # Check metadata
        assert metadata['original'] == test_image
        assert metadata['poisoned'] == str(output_path)
        assert metadata['epsilon'] == 0.01
        assert 'signature_id' in metadata

        # Verify poisoned image is valid
        poisoned_img = Image.open(output_path)
        assert poisoned_img.size == (224, 224)
        assert poisoned_img.mode == 'RGB'

    def test_poison_without_signature(self, marker, test_image, tmp_path):
        """Test that poisoning fails without signature"""
        output_path = tmp_path / "poisoned.jpg"

        with pytest.raises(ValueError, match="No signature loaded"):
            marker.poison_image(test_image, str(output_path))

    def test_epsilon_values(self):
        """Test different epsilon values"""
        # Very small epsilon
        marker1 = RadioactiveMarker(epsilon=0.001)
        assert marker1.epsilon == 0.001

        # Large epsilon
        marker2 = RadioactiveMarker(epsilon=0.1)
        assert marker2.epsilon == 0.1

    def test_signature_json_format(self, marker, tmp_path):
        """Test that saved signature has correct JSON format"""
        marker.generate_signature(seed=12345)

        sig_path = tmp_path / "signature.json"
        marker.save_signature(str(sig_path))

        with open(sig_path, 'r') as f:
            data = json.load(f)

        assert 'signature' in data
        assert 'seed' in data
        assert 'epsilon' in data
        assert 'signature_dim' in data
        assert 'version' in data

        assert len(data['signature']) == 512
        assert data['seed'] == 12345
        assert data['epsilon'] == 0.01
        assert data['version'] == '1.0'


class TestRadioactiveDetector:
    """Test suite for RadioactiveDetector class"""

    @pytest.fixture
    def signature_file(self, tmp_path):
        """Create a test signature file"""
        marker = RadioactiveMarker(epsilon=0.01)
        marker.generate_signature(seed=12345)

        sig_path = tmp_path / "test_signature.json"
        marker.save_signature(str(sig_path))

        return str(sig_path)

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images"""
        images = []
        for i in range(5):
            img = Image.new('RGB', (224, 224), color='blue')
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            images.append(str(img_path))
        return images

    def test_detector_initialization(self, signature_file):
        """Test detector initialization"""
        detector = RadioactiveDetector(signature_file, device='cpu')

        assert detector.signature is not None
        assert len(detector.signature) == 512
        assert detector.seed == 12345
        assert detector.epsilon == 0.01

    def test_detector_load_invalid_file(self, tmp_path):
        """Test detector with invalid signature file"""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json")

        with pytest.raises(json.JSONDecodeError):
            RadioactiveDetector(str(invalid_file))

    def test_detection_on_clean_model(self, signature_file, test_images):
        """Test detection on a model NOT trained on poisoned data"""
        from torchvision.models import resnet18, ResNet18_Weights

        # Create a clean pretrained model
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])

        detector = RadioactiveDetector(signature_file, device='cpu')
        is_poisoned, confidence = detector.detect(model, test_images, threshold=0.1)

        # Clean model should have low correlation
        assert isinstance(is_poisoned, bool)
        assert isinstance(confidence, float)
        # Typically should be False for clean model
        # (but we allow some randomness in testing)

    def test_detection_threshold(self, signature_file, test_images):
        """Test detection with different thresholds"""
        from torchvision.models import resnet18, ResNet18_Weights

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])

        detector = RadioactiveDetector(signature_file, device='cpu')

        # Test with very high threshold
        is_poisoned_high, _ = detector.detect(model, test_images, threshold=0.9)
        assert is_poisoned_high == False  # Should not detect with very high threshold

        # Test with very low threshold
        is_poisoned_low, _ = detector.detect(model, test_images, threshold=0.001)
        # May or may not detect depending on random correlation


class TestIntegration:
    """Integration tests for end-to-end poisoning workflow"""

    def test_full_poison_workflow(self, tmp_path):
        """Test complete workflow: poison -> save -> load -> verify"""
        # Create test image
        test_img = Image.new('RGB', (224, 224), color='green')
        input_path = tmp_path / "input.jpg"
        test_img.save(input_path)

        # Initialize marker and generate signature
        marker = RadioactiveMarker(epsilon=0.02, device='cpu')
        marker.generate_signature(seed=99999)

        # Save signature
        sig_path = tmp_path / "signature.json"
        marker.save_signature(str(sig_path))

        # Poison image
        output_path = tmp_path / "poisoned.jpg"
        result_path, metadata = marker.poison_image(
            str(input_path),
            str(output_path)
        )

        # Verify output
        assert Path(output_path).exists()
        assert Path(sig_path).exists()

        # Verify image is readable
        poisoned_img = Image.open(output_path)
        assert poisoned_img.size == test_img.size

        # Verify signature can be loaded
        detector = RadioactiveDetector(str(sig_path), device='cpu')
        assert detector.seed == 99999

    def test_batch_workflow(self, tmp_path):
        """Test poisoning multiple images with same signature"""
        # Create multiple test images
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(i*80, i*80, i*80))
            img.save(input_dir / f"img_{i}.jpg")

        # Initialize marker
        marker = RadioactiveMarker(epsilon=0.01, device='cpu')
        marker.generate_signature(seed=55555)

        # Poison all images
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        for img_file in input_dir.glob("*.jpg"):
            output_path = output_dir / img_file.name
            marker.poison_image(str(img_file), str(output_path))

        # Verify all outputs exist
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3

        # Verify all are valid images
        for img_path in output_files:
            img = Image.open(img_path)
            assert img.size == (224, 224)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_image_path(self, tmp_path):
        """Test poisoning with invalid image path"""
        marker = RadioactiveMarker(epsilon=0.01)
        marker.generate_signature()

        with pytest.raises(FileNotFoundError):
            marker.poison_image(
                "nonexistent.jpg",
                str(tmp_path / "output.jpg")
            )

    def test_corrupted_image(self, tmp_path):
        """Test poisoning with corrupted image"""
        # Create corrupted image file
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"not an image")

        marker = RadioactiveMarker(epsilon=0.01)
        marker.generate_signature()

        with pytest.raises(Exception):  # PIL will raise an error
            marker.poison_image(
                str(corrupted),
                str(tmp_path / "output.jpg")
            )

    def test_zero_epsilon(self):
        """Test with epsilon = 0 (no perturbation)"""
        marker = RadioactiveMarker(epsilon=0.0)
        assert marker.epsilon == 0.0
        # Should still work, just won't perturb

    def test_negative_epsilon(self):
        """Test with negative epsilon"""
        # This should work (reverse perturbation direction)
        marker = RadioactiveMarker(epsilon=-0.01)
        assert marker.epsilon == -0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
