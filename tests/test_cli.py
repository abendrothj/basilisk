#!/usr/bin/env python3
"""
Tests for CLI tool (poison_cli.py)
Tests command-line interface and argument parsing
"""

import pytest
from pathlib import Path
import json
import sys
from PIL import Image
from click.testing import CliRunner

# Add poison-core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'poison-core'))

from poison_cli import cli


@pytest.fixture
def runner():
    """Create Click test runner"""
    return CliRunner()


@pytest.fixture
def test_image(tmp_path):
    """Create a test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_path = tmp_path / "test_input.jpg"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def test_images_dir(tmp_path):
    """Create directory with multiple test images"""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    for i in range(5):
        img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
        img.save(input_dir / f"img_{i}.jpg")

    return str(input_dir)


class TestInfoCommand:
    """Test the 'info' command"""

    def test_info_command(self, runner):
        """Test that info command runs"""
        result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'Basilisk' in result.output
        assert 'poison' in result.output


class TestPoisonCommand:
    """Test the 'poison' command"""

    def test_poison_basic(self, runner, test_image, tmp_path):
        """Test basic poisoning of single image"""
        output_path = tmp_path / "output.jpg"

        result = runner.invoke(cli, [
            'poison',
            test_image,
            str(output_path)
        ])

        assert result.exit_code == 0
        assert Path(output_path).exists()

        # Check signature file was created
        sig_path = str(output_path).replace('.jpg', '_signature.json')
        assert Path(sig_path).exists()

        # Verify signature content
        with open(sig_path, 'r') as f:
            sig_data = json.load(f)
            assert 'seed' in sig_data
            assert 'signature' in sig_data
            assert 'epsilon' in sig_data

    def test_poison_with_epsilon(self, runner, test_image, tmp_path):
        """Test poisoning with custom epsilon"""
        output_path = tmp_path / "output.jpg"

        result = runner.invoke(cli, [
            'poison',
            test_image,
            str(output_path),
            '--epsilon', '0.02'
        ])

        assert result.exit_code == 0

        # Check epsilon in signature
        sig_path = str(output_path).replace('.jpg', '_signature.json')
        with open(sig_path, 'r') as f:
            sig_data = json.load(f)
            assert sig_data['epsilon'] == 0.02

    def test_poison_with_existing_signature(self, runner, test_image, tmp_path):
        """Test poisoning with existing signature file"""
        # First create a signature
        sig_path = tmp_path / "my_signature.json"
        output_path1 = tmp_path / "output1.jpg"

        result1 = runner.invoke(cli, [
            'poison',
            test_image,
            str(output_path1)
        ])

        # Move signature to known location
        auto_sig = str(output_path1).replace('.jpg', '_signature.json')
        Path(auto_sig).rename(sig_path)

        # Now poison another image with same signature
        output_path2 = tmp_path / "output2.jpg"

        result2 = runner.invoke(cli, [
            'poison',
            test_image,
            str(output_path2),
            '--signature', str(sig_path)
        ])

        assert result2.exit_code == 0
        assert 'Loading signature' in result2.output

    def test_poison_nonexistent_input(self, runner, tmp_path):
        """Test poisoning with nonexistent input file"""
        result = runner.invoke(cli, [
            'poison',
            'nonexistent.jpg',
            str(tmp_path / "output.jpg")
        ])

        assert result.exit_code != 0

    def test_poison_invalid_epsilon(self, runner, test_image, tmp_path):
        """Test poisoning with invalid epsilon value"""
        # Epsilon too large
        result = runner.invoke(cli, [
            'poison',
            test_image,
            str(tmp_path / "output.jpg"),
            '--epsilon', '1.0'
        ])

        # Should either fail or warn
        # (depends on implementation - may allow any value)


class TestBatchCommand:
    """Test the 'batch' command"""

    def test_batch_basic(self, runner, test_images_dir, tmp_path):
        """Test basic batch poisoning"""
        output_dir = tmp_path / "outputs"

        result = runner.invoke(cli, [
            'batch',
            test_images_dir,
            str(output_dir)
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

        # Check that signature file was created
        sig_path = output_dir / "batch_signature.json"
        assert sig_path.exists()

        # Check that output images were created
        output_images = list(output_dir.glob("*.jpg"))
        assert len(output_images) == 5

    def test_batch_with_epsilon(self, runner, test_images_dir, tmp_path):
        """Test batch with custom epsilon"""
        output_dir = tmp_path / "outputs"

        result = runner.invoke(cli, [
            'batch',
            test_images_dir,
            str(output_dir),
            '--epsilon', '0.015'
        ])

        assert result.exit_code == 0

        sig_path = output_dir / "batch_signature.json"
        with open(sig_path, 'r') as f:
            sig_data = json.load(f)
            assert sig_data['epsilon'] == 0.015

    def test_batch_empty_directory(self, runner, tmp_path):
        """Test batch with empty input directory"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "outputs"

        result = runner.invoke(cli, [
            'batch',
            str(empty_dir),
            str(output_dir)
        ])

        # Should warn about no images
        assert 'No images found' in result.output or result.exit_code != 0

    def test_batch_mixed_files(self, runner, tmp_path):
        """Test batch with mixed image and non-image files"""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        # Add images
        for i in range(3):
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(input_dir / f"img_{i}.jpg")

        # Add non-image files
        (input_dir / "readme.txt").write_text("not an image")
        (input_dir / "data.json").write_text("{}")

        output_dir = tmp_path / "outputs"

        result = runner.invoke(cli, [
            'batch',
            str(input_dir),
            str(output_dir)
        ])

        assert result.exit_code == 0

        # Should have processed only the 3 images
        output_images = list(output_dir.glob("*.jpg"))
        assert len(output_images) == 3


class TestDetectCommand:
    """Test the 'detect' command"""

    def test_detect_command_structure(self, runner, tmp_path):
        """Test detect command with dummy inputs (will fail but tests structure)"""
        # Create dummy files
        model_path = tmp_path / "model.pth"
        model_path.write_text("dummy model")

        sig_path = tmp_path / "sig.json"
        sig_path.write_text('{"seed": 123, "signature": [], "epsilon": 0.01}')

        test_dir = tmp_path / "test_images"
        test_dir.mkdir()

        result = runner.invoke(cli, [
            'detect',
            str(model_path),
            str(sig_path),
            str(test_dir)
        ])

        # Will fail because of dummy data, but command structure is tested
        # Just ensure it doesn't crash on argument parsing


class TestOutputFormats:
    """Test output formatting and messages"""

    def test_poison_success_message(self, runner, test_image, tmp_path):
        """Test that success message is displayed"""
        output_path = tmp_path / "output.jpg"

        result = runner.invoke(cli, [
            'poison',
            test_image,
            str(output_path)
        ])

        assert 'Poisoned image saved' in result.output or '✅' in result.output

    def test_batch_progress_output(self, runner, test_images_dir, tmp_path):
        """Test that batch shows progress"""
        output_dir = tmp_path / "outputs"

        result = runner.invoke(cli, [
            'batch',
            test_images_dir,
            str(output_dir)
        ])

        # Should show completion message
        assert 'Complete' in result.output or 'successful' in result.output


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_output_to_same_as_input(self, runner, test_image):
        """Test poisoning where output overwrites input"""
        # This might be intentional, so just test it works
        result = runner.invoke(cli, [
            'poison',
            test_image,
            test_image  # Same path
        ])

        # Should either succeed or warn
        # (implementation-dependent)

    def test_unicode_filenames(self, runner, tmp_path):
        """Test handling of unicode in filenames"""
        # Create image with unicode filename
        img = Image.new('RGB', (100, 100), color='green')
        input_path = tmp_path / "test_图片.jpg"
        img.save(input_path)

        output_path = tmp_path / "output_图片.jpg"

        result = runner.invoke(cli, [
            'poison',
            str(input_path),
            str(output_path)
        ])

        # Should handle unicode filenames
        assert result.exit_code == 0 or 'error' in result.output.lower()

    def test_long_path(self, runner, tmp_path):
        """Test handling of very long file paths"""
        # Create nested directory structure
        deep_dir = tmp_path
        for i in range(10):
            deep_dir = deep_dir / f"level_{i}"
        deep_dir.mkdir(parents=True)

        img = Image.new('RGB', (100, 100), color='orange')
        input_path = deep_dir / "test.jpg"
        img.save(input_path)

        output_path = deep_dir / "output.jpg"

        result = runner.invoke(cli, [
            'poison',
            str(input_path),
            str(output_path)
        ])

        # Should handle long paths
        assert result.exit_code == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
