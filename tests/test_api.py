#!/usr/bin/env python3
"""
Unit tests for Perceptual Hash Tracking API
Tests Flask endpoints for hash extraction and comparison
"""

import pytest
import json
import io
import tempfile
from pathlib import Path
from PIL import Image
import sys
import cv2
import numpy as np
import unittest.mock

# Add api directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'api'))

from server import app


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    
    # Store original db path to restore if needed (though patch handles this)
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        temp_db_path = temp_db.name
    
    # Initialize temp database
    from core.hash_database import HashDatabase
    test_db = HashDatabase(temp_db_path)
    
    # Patch the global db object in server module
    with unittest.mock.patch('server.db', test_db):
        with app.test_client() as client:
            yield client
            
    # Cleanup
    test_db.close()
    Path(temp_db_path).unlink(missing_ok=True)


@pytest.fixture
def test_video():
    """Create a test video in memory"""
    def _create_video():
        # Create temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Create simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 30.0, (224, 224))

        for i in range(30):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

        with open(temp_path, 'rb') as f:
            video_bytes = io.BytesIO(f.read())

        Path(temp_path).unlink()
        video_bytes.seek(0)
        return video_bytes

    return _create_video


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get('/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'sigil-hash-api'
        assert 'version' in data
        assert data['feature'] == 'perceptual_hash_tracking'


class TestExtractEndpoint:
    """Test hash extraction endpoint"""

    def test_extract_success(self, client, test_video):
        """Test successful hash extraction"""
        video_data = test_video()

        data = {
            'video': (video_data, 'test.mp4'),
            'max_frames': '30'
        }

        response = client.post('/api/extract',
                              data=data,
                              content_type='multipart/form-data')

        assert response.status_code == 200
        result = json.loads(response.data)

        assert result['success'] is True
        assert 'hash' in result
        assert len(result['hash']) == 256  # 256-bit hash
        assert all(c in '01' for c in result['hash'])  # Binary string
        assert 'hash_id' in result
        assert result['num_frames'] <= 30

    def test_extract_no_file(self, client):
        """Test extraction without video file"""
        response = client.post('/api/extract',
                              data={},
                              content_type='multipart/form-data')

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result

    def test_extract_invalid_frames(self, client, test_video):
        """Test extraction with invalid max_frames parameter"""
        video_data = test_video()

        data = {
            'video': (video_data, 'test.mp4'),
            'max_frames': '500'  # Too high
        }

        response = client.post('/api/extract',
                              data=data,
                              content_type='multipart/form-data')

        assert response.status_code == 400


class TestCompareEndpoint:
    """Test hash comparison endpoint"""

    def test_compare_with_hash_string(self, client):
        """Test comparison with direct hash string"""
        # Create a dummy 256-bit hash
        test_hash = '0' * 128 + '1' * 128

        data = {
            'hash': test_hash,
            'threshold': '30'
        }

        response = client.post('/api/compare',
                              data=data,
                              content_type='multipart/form-data')

        assert response.status_code == 200
        result = json.loads(response.data)

        assert result['success'] is True
        assert 'matches' in result
        assert 'num_matches' in result
        assert result['query_hash'] == test_hash

    def test_compare_invalid_hash(self, client):
        """Test comparison with invalid hash format"""
        data = {
            'hash': 'invalid',  # Not 256 bits
            'threshold': '30'
        }

        response = client.post('/api/compare',
                              data=data,
                              content_type='multipart/form-data')

        assert response.status_code == 400

    def test_compare_no_input(self, client):
        """Test comparison without hash or video"""
        response = client.post('/api/compare',
                              data={},
                              content_type='multipart/form-data')

        assert response.status_code == 400


class TestStatsEndpoint:
    """Test statistics endpoint"""

    def test_stats(self, client):
        """Test /api/stats endpoint"""
        response = client.get('/api/stats')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'total_hashes' in data
        assert 'database_path' in data
