# Project Restructure Plan - Perceptual Hash Focus

**Date:** December 28, 2025
**Goal:** Reposition Sigil as a perceptual hash tracking system (primary) with radioactive marking as experimental research (isolated)

---

## New Directory Structure

```
sigil/
├── core/                          # PRIMARY: Perceptual hash system
│   ├── perceptual_hash.py            # Main hash extraction (moved from experiments/)
│   ├── video_loader.py               # Video frame loading utilities
│   ├── feature_extractor.py          # Canny/Gabor/Laplacian/RGB features
│   ├── hash_generator.py             # Random projection + binarization
│   ├── hash_database.py              # SQLite database for hash storage
│   └── __init__.py
│
├── cli/                           # Command-line interface
│   ├── extract.py                    # Extract hash from video
│   ├── compare.py                    # Compare two hashes
│   ├── batch.py                      # Batch processing
│   ├── database.py                   # Database management commands
│   └── __init__.py
│
├── api/                           # Flask REST API
│   ├── server.py                     # Updated for perceptual hash endpoints
│   ├── routes/
│   │   ├── hash.py                   # /extract_hash, /compare_hash
│   │   └── database.py               # /store_hash, /query_hash
│   └── requirements.txt
│
├── web-ui/                        # Next.js frontend
│   ├── app/
│   │   ├── page.tsx                  # Main dashboard (perceptual hash focus)
│   │   ├── extract/                  # Hash extraction interface
│   │   ├── database/                 # Hash database viewer
│   │   └── compare/                  # Hash comparison tool
│   └── package.json
│
├── experimental/                  # ISOLATED: Radioactive marking research
│   ├── radioactive/
│   │   ├── poison.py                 # Radioactive marking implementation
│   │   ├── detect.py                 # Signature detection
│   │   ├── cli.py                    # CLI for poisoning (experimental)
│   │   └── README.md                 # ⚠️ Experimental - limitations documented
│   ├── verification/
│   │   ├── verify_poison_FIXED.py    # Empirical validation
│   │   ├── create_dataset.py
│   │   └── README.md
│   └── deprecated_dct_approach/      # Archived research
│
├── tests/                         # Test suite (reorganized)
│   ├── test_perceptual_hash.py       # PRIMARY tests
│   ├── test_feature_extraction.py
│   ├── test_hash_generation.py
│   ├── test_database.py
│   ├── test_api.py
│   └── experimental/
│       └── test_radioactive.py       # Isolated experimental tests
│
├── docs/                          # Documentation
│   ├── Perceptual_Hash_Whitepaper.md # PRIMARY technical doc
│   ├── API_REFERENCE.md              # API documentation
│   ├── CLI_GUIDE.md                  # CLI usage guide
│   ├── DATABASE_SCHEMA.md            # Database schema documentation
│   ├── COMPRESSION_LIMITS.md
│   ├── RESEARCH.md
│   └── experimental/
│       ├── Radioactive_Marking.md    # Experimental research doc
│       └── LAYER1_ALTERNATIVES.md
│
├── notebooks/                     # Jupyter notebooks
│   ├── Sigil_Demo.ipynb          # PRIMARY demo (perceptual hash)
│   └── experimental/
│       └── Radioactive_Research.ipynb
│
├── docker/                        # Docker configuration
│   ├── Dockerfile.api
│   ├── Dockerfile.web
│   ├── docker-compose.yml           # Updated for new structure
│   └── README.md
│
├── setup.sh                       # Updated setup script
├── run_api.sh                     # Updated API runner
├── run_web.sh                     # Updated web UI runner
├── README.md                      # Updated (perceptual hash focus)
├── VERIFICATION_PROOF.md
├── TESTING_SUMMARY.md
├── LICENSE
└── .gitignore
```

---

## Migration Steps

### Phase 1: Create New Core Structure

1. Create `core/` directory
2. Move `experiments/perceptual_hash.py` → `core/perceptual_hash.py`
3. Move `experiments/batch_hash_robustness.py` → `cli/batch.py` (refactored)
4. Create `core/hash_database.py` (new SQLite wrapper)
5. Create `core/feature_extractor.py` (extract from perceptual_hash.py)

### Phase 2: Create CLI Interface

1. Create `cli/` directory
2. Create `cli/extract.py` - Extract hash from single video
3. Create `cli/compare.py` - Compare two hashes
4. Create `cli/batch.py` - Batch processing
5. Create `cli/database.py` - Database management

### Phase 3: Isolate Radioactive Marking

1. Create `experimental/radioactive/` directory
2. Move `Sigil/` → `experimental/radioactive/`
3. Move `verification/` → `experimental/verification/`
4. Move `experiments/deprecated_dct_approach/` → `experimental/deprecated_dct_approach/`
5. Add ⚠️ WARNING README in `experimental/`

### Phase 4: Update API

1. Update `api/server.py` for new imports
2. Create `api/routes/hash.py` (perceptual hash endpoints)
3. Create `api/routes/database.py` (hash database endpoints)
4. Remove radioactive endpoints (or move to experimental API)

### Phase 5: Update Web UI

1. Update `web-ui/app/page.tsx` (focus on perceptual hash)
2. Create hash extraction interface
3. Create hash database viewer
4. Create hash comparison tool
5. Move radioactive UI to experimental section (optional)

### Phase 6: Update Docker

1. Update `docker-compose.yml` for new structure
2. Update Dockerfiles for new paths
3. Update environment variables

### Phase 7: Update Tests

1. Reorganize tests to match new structure
2. Move radioactive tests to `tests/experimental/`
3. Add integration tests for hash database

### Phase 8: Update Documentation

1. Update README.md (already done)
2. Create API_REFERENCE.md
3. Create CLI_GUIDE.md
4. Create DATABASE_SCHEMA.md
5. Move radioactive docs to `docs/experimental/`

---

## API Changes

### New Primary Endpoints

```
POST /extract_hash
- Extract perceptual hash from video
- Input: multipart/form-data (video file)
- Output: { "hash": "256-bit binary string", "hash_hex": "hex", "timestamp": "ISO8601" }

POST /compare_hash
- Compare two hashes (Hamming distance)
- Input: { "hash1": "...", "hash2": "..." }
- Output: { "hamming_distance": 8, "similarity": 96.9, "match": true }

POST /store_hash
- Store hash in database with metadata
- Input: { "hash": "...", "video_id": "...", "platform": "youtube", "metadata": {...} }
- Output: { "id": 123, "stored_at": "..." }

GET /query_hash/:hash
- Query database for similar hashes
- Output: [ { "hash": "...", "distance": 8, "video_id": "...", "platform": "...", ... } ]

GET /database/stats
- Database statistics (total hashes, platforms, date range)
```

### Moved to Experimental API (Optional)

```
POST /experimental/radioactive/poison
POST /experimental/radioactive/detect
```

---

## CLI Changes

### New Primary Commands

```bash
# Extract hash from video
sigil extract video.mp4 --frames 60 --output hash.txt

# Compare two videos
sigil compare video1.mp4 video2.mp4 --frames 60

# Batch process directory
sigil batch videos/ --frames 60 --output hashes.csv

# Store hash in database
sigil store video.mp4 --platform youtube --video-id abc123

# Query database
sigil query hash.txt --threshold 30

# Database statistics
sigil db stats
```

### Moved to Experimental (Optional)

```bash
# Experimental radioactive poisoning
sigil experimental poison image.jpg output.jpg
```

---

## Docker Changes

### New docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "5001:5001"
    environment:
      - HASH_DB_PATH=/data/hashes.db
    volumes:
      - hash_data:/data
      - ./core:/app/core
      - ./api:/app/api
    command: python api/server.py

  web:
    build:
      context: .
      dockerfile: docker/Dockerfile.web
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:5001
    depends_on:
      - api

volumes:
  hash_data:
```

---

## Import Path Changes

### Before

```python
from experiments.perceptual_hash import extract_hash
from poison_core.radioactive_poison import RadioactiveMarker
```

### After

```python
# Primary imports
from core.perceptual_hash import extract_hash
from core.hash_database import HashDatabase

# Experimental imports (clearly marked)
from experimental.radioactive.poison import RadioactiveMarker
```

---

## README Updates

### Old Structure

```markdown
## 🚀 Quick Start

### Option 1: Docker
### Option 2: Local Setup
### Option 3: CLI Only
```

### New Structure

```markdown
## 🚀 Quick Start

### Extract Perceptual Hash from Video
```bash
git clone https://github.com/abendrothj/sigil.git
cd sigil
./setup.sh
source venv/bin/activate

# Extract hash
python -m cli.extract your_video.mp4 --frames 60
```

### Full Stack (Web UI + API + Database)
```bash
docker-compose up
# Visit http://localhost:3000
```
```

---

## Breaking Changes

1. **Import paths** - All imports from `experiments/` and `Sigil/` need updating
2. **CLI commands** - New command structure (`sigil extract` vs `python experiments/perceptual_hash.py`)
3. **API endpoints** - New endpoint structure (`/extract_hash` vs `/poison`)
4. **Docker paths** - Volume mounts updated for new structure

---

## Migration Guide for Users

**If you're using perceptual hash tracking:**
- ✅ Update imports: `from core.perceptual_hash import extract_hash`
- ✅ Use new CLI: `python -m cli.extract video.mp4`
- ✅ No major code changes needed

**If you're using radioactive marking:**
- ⚠️ Move to experimental imports: `from experimental.radioactive.poison import RadioactiveMarker`
- ⚠️ Note: This is now marked as experimental research
- ⚠️ Limitations clearly documented

---

## Timeline

**Estimated completion:** 2-4 hours

1. Phase 1 (30 min) - Create core structure
2. Phase 2 (30 min) - Create CLI
3. Phase 3 (15 min) - Isolate radioactive
4. Phase 4 (30 min) - Update API
5. Phase 5 (45 min) - Update web UI
6. Phase 6 (15 min) - Update Docker
7. Phase 7 (30 min) - Update tests
8. Phase 8 (30 min) - Update docs

---

## Success Criteria

✅ All tests passing with new structure
✅ Docker compose works with new structure
✅ CLI commands functional
✅ API endpoints functional
✅ Web UI loads and works
✅ README reflects new structure
✅ No import errors

---

## Rollback Plan

If restructure fails:
1. Revert to previous commit
2. Cherry-pick documentation updates
3. Keep old structure, just update docs

---

**Ready to proceed? Y/N**
