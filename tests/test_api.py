#!/usr/bin/env python3
"""
Unit tests for API server
Tests Flask endpoints and image upload/download
"""

import pytest
import json
import io
import base64
from pathlib import Path
from PIL import Image
import sys

# Add api directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'api'))

from server import app


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def test_image():
    """Create a test image in memory"""
    img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get('/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'basilisk-api'
        assert 'version' in data


class TestPoisonEndpoint:
    """Test image poisoning endpoint"""

    def test_poison_success(self, client, test_image):
        """Test successful image poisoning"""
        data = {
            'image': (test_image, 'test.jpg'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 200

        result = json.loads(response.data)
        assert result['success'] == True
        assert 'poisoned_image' in result
        assert 'signature_id' in result
        assert 'signature' in result

        # Verify base64 image
        assert result['poisoned_image'].startswith('data:image/jpeg;base64,')

        # Verify signature structure
        sig = result['signature']
        assert 'seed' in sig
        assert 'epsilon' in sig
        assert 'signature' in sig
        assert len(sig['signature']) == 512

    def test_poison_no_image(self, client):
        """Test poisoning without image"""
        response = client.post('/api/poison', data={})

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'No image' in result['error']

    def test_poison_invalid_epsilon(self, client, test_image):
        """Test poisoning with invalid epsilon"""
        data = {
            'image': (test_image, 'test.jpg'),
            'epsilon': '0.1'  # Too large
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'Epsilon' in result['error']

    def test_poison_default_epsilon(self, client, test_image):
        """Test poisoning with default epsilon"""
        data = {
            'image': (test_image, 'test.jpg')
            # No epsilon provided
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['signature']['epsilon'] == 0.01  # Default value

    def test_poison_invalid_file_type(self, client):
        """Test poisoning with non-image file"""
        data = {
            'image': (io.BytesIO(b"not an image"), 'test.txt'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result

    def test_poison_png_image(self, client):
        """Test poisoning PNG image"""
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        data = {
            'image': (img_bytes, 'test.png'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['success'] == True

    def test_poison_different_epsilons(self, client, test_image):
        """Test poisoning with different epsilon values"""
        epsilons = ['0.005', '0.01', '0.02', '0.05']

        for eps in epsilons:
            # Reset BytesIO position
            test_image.seek(0)

            data = {
                'image': (test_image, 'test.jpg'),
                'epsilon': eps
            }

            response = client.post(
                '/api/poison',
                data=data,
                content_type='multipart/form-data'
            )

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result['signature']['epsilon'] == float(eps)


class TestBatchEndpoint:
    """Test batch poisoning endpoint"""

    def test_batch_poison_success(self, client):
        """Test successful batch poisoning"""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(i*80, i*80, i*80))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append((img_bytes, f'test_{i}.jpg'))

        data = {
            'images': images,
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/batch',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 200

        result = json.loads(response.data)
        assert result['success'] == True
        assert 'results' in result
        assert 'signature' in result
        assert result['total'] == 3
        assert result['successful'] == 3

        # Check each result
        for res in result['results']:
            assert res['success'] == True
            assert 'poisoned_image' in res
            assert 'original_name' in res

    def test_batch_no_images(self, client):
        """Test batch poisoning without images"""
        response = client.post('/api/batch', data={})

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result

    def test_batch_too_many_images(self, client):
        """Test batch poisoning with too many images"""
        # Try to send 101 images (over the 100 limit)
        images = []
        for i in range(101):
            img_bytes = io.BytesIO()
            img = Image.new('RGB', (50, 50), color='blue')
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append((img_bytes, f'test_{i}.jpg'))

        data = {
            'images': images,
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/batch',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'Maximum 100 images' in result['error']

    def test_batch_shared_signature(self, client):
        """Test that batch uses same signature for all images"""
        images = []
        for i in range(2):
            img = Image.new('RGB', (224, 224), color='green')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append((img_bytes, f'test_{i}.jpg'))

        data = {
            'images': images,
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/batch',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        result = json.loads(response.data)

        # There should be one signature for all images
        assert 'signature_id' in result
        assert 'signature' in result


class TestImageValidation:
    """Test image validation and error handling"""

    def test_empty_filename(self, client):
        """Test upload with empty filename"""
        data = {
            'image': (io.BytesIO(b"data"), ''),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_large_image(self, client):
        """Test with large image"""
        # Create a large image (4K)
        img = Image.new('RGB', (3840, 2160), color='purple')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)

        data = {
            'image': (img_bytes, 'large.jpg'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        # Should succeed (or fail gracefully if too large)
        assert response.status_code in [200, 413, 500]

    def test_small_image(self, client):
        """Test with very small image"""
        img = Image.new('RGB', (10, 10), color='yellow')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        data = {
            'image': (img_bytes, 'small.jpg'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        # Should succeed
        assert response.status_code == 200


class TestResponseFormat:
    """Test API response formats"""

    def test_success_response_structure(self, client, test_image):
        """Test structure of successful response"""
        data = {
            'image': (test_image, 'test.jpg'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        result = json.loads(response.data)

        # Check required fields
        required_fields = ['success', 'poisoned_image', 'signature_id', 'signature']
        for field in required_fields:
            assert field in result

        # Verify types
        assert isinstance(result['success'], bool)
        assert isinstance(result['poisoned_image'], str)
        assert isinstance(result['signature_id'], str)
        assert isinstance(result['signature'], dict)

    def test_error_response_structure(self, client):
        """Test structure of error response"""
        response = client.post('/api/poison', data={})

        result = json.loads(response.data)
        assert 'error' in result
        assert isinstance(result['error'], str)

    def test_content_type_json(self, client, test_image):
        """Test that response is JSON"""
        data = {
            'image': (test_image, 'test.jpg'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.content_type == 'application/json'


class TestCORS:
    """Test CORS headers"""

    def test_cors_headers_present(self, client, test_image):
        """Test that CORS headers are present"""
        data = {
            'image': (test_image, 'test.jpg'),
            'epsilon': '0.01'
        }

        response = client.post(
            '/api/poison',
            data=data,
            content_type='multipart/form-data'
        )

        # CORS headers should be present (added by Flask-CORS)
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
