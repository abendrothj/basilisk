#!/usr/bin/env python3
"""
Sigil API Server
Flask backend for perceptual hash tracking service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import json
import hashlib
from pathlib import Path
import tempfile
import numpy as np

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.perceptual_hash import load_video_frames, extract_perceptual_features, compute_perceptual_hash
    from core.hash_database import HashDatabase
    from core.crypto_signatures import SignatureManager, SigilIdentity
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Make sure core/ is in the correct location")
    # Only exit if not in test mode
    if __name__ == "__main__":
        sys.exit(1)
    else:
        raise

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max for videos
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
TEMP_DIR = Path(tempfile.gettempdir()) / 'sigil'
TEMP_DIR.mkdir(exist_ok=True)

# Initialize database
DB_PATH = Path(__file__).parent.parent / 'sigil_hashes.db'
db = HashDatabase(str(DB_PATH))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'sigil-hash-api',
        'version': '1.0.0',
        'feature': 'perceptual_hash_tracking'
    })


@app.route('/api/extract', methods=['POST'])
def extract_hash():
    """
    Extract perceptual hash from uploaded video

    Request:
        - video: File upload
        - max_frames: int (optional, default 60)
        - metadata: JSON string (optional)

    Response:
        - success: bool
        - hash: 256-bit hash as string
        - hash_id: unique identifier
        - metadata: stored metadata
    """
    try:
        # Validate request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use MP4, AVI, MOV, or MKV'}), 400

        # Get parameters
        max_frames = int(request.form.get('max_frames', 60))
        metadata_str = request.form.get('metadata', '{}')
        metadata = json.loads(metadata_str)
        sign = request.form.get('sign', 'false').lower() == 'true'

        # Validate max_frames
        if max_frames < 10 or max_frames > 300:
            return jsonify({'error': 'max_frames must be between 10 and 300'}), 400

        # Generate unique ID
        request_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

        # Save uploaded file
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        input_path = TEMP_DIR / f"{request_id}_input.{ext}"

        file.save(str(input_path))

        # Extract hash
        print(f"Extracting hash from {filename} with max_frames={max_frames}")
        frames = load_video_frames(str(input_path), max_frames=max_frames)

        if not frames:
            input_path.unlink(missing_ok=True)
            return jsonify({'error': 'Failed to load video frames'}), 400

        features = extract_perceptual_features(frames)
        hash_bits = compute_perceptual_hash(features)

        # Convert to string
        hash_str = ''.join(map(str, hash_bits))

        # Convert hash to hex for signing/storage
        hash_hex = hex(int(hash_str, 2))[2:].zfill(64)

        # Generate signature if requested
        signature_doc = None
        if sign:
            try:
                sig_manager = SignatureManager()
                sig_metadata = {
                    'video_filename': filename,
                    'frames_analyzed': len(frames),
                    'api_request_id': request_id
                }
                signature_doc = sig_manager.identity.sign_hash(hash_hex, sig_metadata)
            except Exception as e:
                return jsonify({'error': f'Signature generation failed: {str(e)}'}), 500

        # Store in database
        metadata['original_filename'] = filename
        metadata['num_frames'] = len(frames)

        # Add signature fields if signing
        db_args = {
            'hash_binary': hash_bits,
            'file_path': str(input_path),
            'frame_count': len(frames),
            'metadata': metadata
        }

        if signature_doc:
            db_args['signature'] = signature_doc['proof']['signature']
            db_args['public_key'] = signature_doc['proof']['public_key']
            db_args['key_id'] = signature_doc['proof']['key_id']
            db_args['signed_at'] = signature_doc['proof']['signed_at']
            db_args['signature_version'] = signature_doc['version']

        hash_id = db.store_hash(**db_args)

        # Clean up
        input_path.unlink(missing_ok=True)

        response = {
            'success': True,
            'hash': hash_str,
            'hash_hex': hash_hex,
            'hash_id': hash_id,
            'num_frames': len(frames),
            'metadata': metadata
        }

        if signature_doc:
            response['signature'] = signature_doc

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in extract_hash: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/compare', methods=['POST'])
def compare_hash():
    """
    Compare video hash against database

    Request:
        - video: File upload OR hash: string
        - threshold: int (optional, default 30)

    Response:
        - success: bool
        - matches: list of matching hashes
        - closest_distance: minimum hamming distance
    """
    try:
        threshold = int(request.form.get('threshold', 30))

        # Get hash either from video or direct input
        if 'hash' in request.form:
            hash_str = request.form['hash']
            if len(hash_str) != 256 or not all(c in '01' for c in hash_str):
                return jsonify({'error': 'Invalid hash format. Must be 256-bit binary string'}), 400
            hash_bits = [int(c) for c in hash_str]
        elif 'video' in request.files:
            file = request.files['video']
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400

            max_frames = int(request.form.get('max_frames', 60))
            request_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            input_path = TEMP_DIR / f"{request_id}_input.{ext}"

            file.save(str(input_path))
            frames = load_video_frames(str(input_path), max_frames=max_frames)
            input_path.unlink(missing_ok=True)

            if not frames:
                return jsonify({'error': 'Failed to load video frames'}), 400

            features = extract_perceptual_features(frames)
            hash_bits = compute_perceptual_hash(features)
            hash_str = ''.join(map(str, hash_bits))
        else:
            return jsonify({'error': 'Provide either video file or hash string'}), 400

        # Find matches - convert hash_str back to numpy array for query_similar
        hash_bits_for_query = np.array([int(c) for c in hash_str])
        matches = db.query_similar(hash_bits_for_query, threshold=threshold)

        # Calculate closest distance
        closest_distance = min([m['hamming_distance'] for m in matches]) if matches else 256

        return jsonify({
            'success': True,
            'query_hash': hash_str,
            'matches': matches,
            'num_matches': len(matches),
            'closest_distance': closest_distance,
            'threshold': threshold
        })

    except Exception as e:
        print(f"Error in compare_hash: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        stats = db.get_stats()
        return jsonify({
            'success': True,
            'total_hashes': stats['total_hashes'],
            'by_platform': stats['by_platform'],
            'oldest_entry': stats['oldest_entry'],
            'newest_entry': stats['newest_entry'],
            'database_path': str(DB_PATH)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify_signature():
    """
    Verify a cryptographic signature

    Request:
        - signature: JSON signature document

    Response:
        - valid: bool
        - key_id: string
        - hash_hex: string
        - signed_at: string
        - error: string (if invalid)
    """
    try:
        if not request.json or 'signature' not in request.json:
            return jsonify({'error': 'Signature document required in JSON body'}), 400

        signature_doc = request.json['signature']

        # Verify signature
        is_valid, error = SigilIdentity.verify_signature(signature_doc)

        if is_valid:
            return jsonify({
                'success': True,
                'valid': True,
                'key_id': signature_doc['proof']['key_id'],
                'hash_hex': signature_doc['claim']['hash_hex'],
                'signed_at': signature_doc['proof']['signed_at'],
                'algorithm': signature_doc['proof']['algorithm']
            })
        else:
            return jsonify({
                'success': True,
                'valid': False,
                'error': error
            })

    except Exception as e:
        print(f"Error in verify_signature: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/identity', methods=['GET'])
def get_identity():
    """
    Get current server identity information

    Response:
        - has_identity: bool
        - key_id: string (if exists)
        - public_key: string (PEM format)
    """
    try:
        identity = SigilIdentity()

        if not identity.private_key:
            return jsonify({
                'success': True,
                'has_identity': False,
                'message': 'No server identity configured'
            })

        return jsonify({
            'success': True,
            'has_identity': True,
            'key_id': identity.key_id,
            'public_key': identity.export_public_key(),
            'algorithm': 'Ed25519'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/identity/generate', methods=['POST'])
def generate_identity():
    """
    Generate new server identity (admin only)
    WARNING: This will invalidate all existing signatures

    Response:
        - key_id: string
        - public_key: string
    """
    try:
        # In production, this should require authentication
        identity = SigilIdentity()

        # Check if identity exists
        if identity.private_key and not request.json.get('overwrite', False):
            return jsonify({
                'error': 'Identity already exists. Set overwrite=true to replace'
            }), 400

        # Generate new identity
        identity.generate(overwrite=request.json.get('overwrite', False))

        return jsonify({
            'success': True,
            'key_id': identity.key_id,
            'public_key': identity.export_public_key(),
            'algorithm': 'Ed25519',
            'warning': 'Private key stored unencrypted at ~/.sigil/identity.pem'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("✨ Sigil Perceptual Hash Tracking API")
    print("=" * 60)
    print("Starting Flask server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health               - Health check")
    print("  POST /api/extract          - Extract perceptual hash from video")
    print("  POST /api/compare          - Compare hash against database")
    print("  GET  /api/stats            - Database statistics")
    print("  POST /api/verify           - Verify cryptographic signature")
    print("  GET  /api/identity         - Get server identity info")
    print("  POST /api/identity/generate- Generate new server identity")
    print("=" * 60)
    print()
    print("NOTE: This API uses a FIXED SEED (42) for hash generation.")
    print("WARNING: Fixed seed means hashes are reproducible but not cryptographically secure.")
    print("Anyone with access to this code can compute the same hash for any video.")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
