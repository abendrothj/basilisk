# Sigil Project Structure

**Last Updated:** December 28, 2025

---

## Overview

Sigil is a compression-robust perceptual hash tracking system for forensic video fingerprinting. The project is organized with **production-ready code** in the main directories and **experimental research** isolated in `experimental/`.

---

## Directory Structure

```
sigil/
│
├── core/                          # 🎯 PRIMARY: Perceptual Hash System
│   ├── perceptual_hash.py            # Hash extraction (Canny, Gabor, Laplacian, RGB)
│   ├── batch_robustness.py           # Batch compression testing
│   ├── hash_database.py              # SQLite hash storage & querying
│   └── __init__.py                   # Clean API exports
│
├── cli/                           # 💻 Command-Line Interface
│   ├── extract.py                    # Extract hash from video
│   ├── compare.py                    # Compare two hashes (Hamming distance)
│   └── __init__.py
│
├── api/                           # 🌐 Flask REST API
│   ├── server.py                     # API server (updated for perceptual hash)
│   ├── routes/                       # API route handlers (to be added)
│   └── requirements.txt
│
├── web-ui/                        # 🖥️  Next.js Frontend
│   ├── app/                          # Next.js app router
│   └── package.json
│
├── experiments/                   # 📊 Original Research Files
│   ├── perceptual_hash.py            # Original hash implementation (copied to core/)
│   ├── batch_hash_robustness.py      # Original batch testing
│   ├── make_short_test_video.py      # Test video generation
│   └── short_test.mp4, test_crf*.mp4 # Test videos
│
├── experimental/                  # ⚠️  EXPERIMENTAL RESEARCH ONLY
│   ├── README.md                     # ⚠️ Warnings about limitations
│   ├── radioactive/                  # Radioactive data marking (limited to transfer learning)
│   ├── verification/                 # Verification scripts
│   ├── deprecated_dct_approach/      # Archived DCT poisoning (failed)
│   ├── test_data/                    # Test datasets
│   └── verification_data*/           # Verification datasets
│
├── docs/                          # 📚 Documentation
│   ├── Perceptual_Hash_Whitepaper.md # PRIMARY technical whitepaper
│   ├── COMPRESSION_LIMITS.md         # Compression analysis
│   ├── VERIFICATION_PROOF.md         # Empirical validation
│   ├── LAYER1_ALTERNATIVES.md        # Research on radioactive improvements
│   ├── RESTRUCTURE_PLAN.md           # Restructure documentation
│   ├── LAUNCH_HN.md                  # HN launch post
│   ├── PHASE2_ADVERSARIAL_COLLISION.md
│   ├── APPROACH.md
│   ├── RESEARCH.md
│   └── CREDITS.md
│
├── notebooks/                     # 📓 Jupyter Notebooks
│   └── Sigil_Demo.ipynb           # Interactive perceptual hash demo (Colab-ready)
│
├── tests/                         # ✅ Test Suite
│   ├── test_perceptual_hash.py       # Perceptual hash tests
│   ├── test_radioactive_poison.py    # Radioactive marking tests
│   ├── test_api.py                   # API endpoint tests
│   └── test_cli.py                   # CLI tests
│
├── docker/                        # 🐳 Docker Configuration
│   ├── Dockerfile.api                # API server image
│   └── Dockerfile.web                # Web UI image
│
├── research/                      # 📖 Research Papers & References
│
├── venv/                          # ✨ Python Virtual Environment (local)
│
├── docker-compose.yml             # Docker Compose configuration
├── setup.sh                       # Setup script
├── run_api.sh                     # API runner
├── run_web.sh                     # Web UI runner
├── run_tests.sh                   # Test runner
├── README.md                      # Main project documentation
├── VERIFICATION_PROOF.md          # Empirical validation results
├── TESTING_SUMMARY.md             # Test suite summary
├── RESTRUCTURE_SUMMARY.md         # Restructure documentation
├── PROJECT_STRUCTURE.md           # This file
├── LICENSE                        # MIT License
└── .gitignore
```

---

## Key Directories

### Production-Ready 🎯

- **`core/`** - Primary perceptual hash system (96-97% compression stability)
- **`cli/`** - Professional command-line interface
- **`api/`** - REST API for hash extraction and comparison
- **`web-ui/`** - Web interface for hash management

### Documentation 📚

- **`docs/`** - All technical documentation and research
- **`notebooks/`** - Interactive demos and tutorials
- **`README.md`** - Main project overview

### Experimental ⚠️

- **`experimental/`** - Research code with significant limitations
  - Radioactive marking (transfer learning only)
  - Deprecated DCT approach (proven unsolvable)
  - Verification datasets

### Development 🔧

- **`tests/`** - Comprehensive test suite (55+ tests, 85%+ coverage)
- **`docker/`** - Docker configuration for deployment
- **`experiments/`** - Original research files (preserved for backward compatibility)

---

## File Organization Principles

1. **Production vs Experimental:**
   - Production code: `core/`, `cli/`, `api/`, `web-ui/`
   - Experimental code: `experimental/` (clearly marked)

2. **Documentation:**
   - Primary docs: Root level (README.md, VERIFICATION_PROOF.md)
   - Technical docs: `docs/`
   - API docs: `api/` (to be added)

3. **Tests:**
   - All tests in `tests/`
   - Mirrors directory structure of code

4. **Docker:**
   - All Docker files in `docker/`
   - docker-compose.yml in root

---

## Import Conventions

### Production Code

```python
# Core perceptual hash system
from core import load_video_frames, extract_perceptual_features, compute_perceptual_hash
from core.hash_database import HashDatabase

# CLI usage
# python -m cli.extract video.mp4 --frames 60
# python -m cli.compare video1.mp4 video2.mp4
```

### Experimental Code (Use with Caution)

```python
# Radioactive marking (experimental - transfer learning only)
from experimental.radioactive.radioactive_poison import RadioactiveMarker

# Note: Read experimental/README.md for limitations
```

---

## Data Storage

### Hash Database

- **Location:** `/data/hashes.db` (Docker) or local path
- **Schema:** SQLite database with hash, metadata, timestamps
- **Interface:** `core.hash_database.HashDatabase`

### Test Data

- **Location:** `experimental/test_data/`
- **Contents:** Test videos, verification datasets
- **Not in git:** Large files excluded via .gitignore

---

## Port Assignments

- **API:** 5001 (updated from 5000)
- **Web UI:** 3000
- **Database:** SQLite file (no port)

---

## Environment Variables

### API Server

```bash
FLASK_APP=api/server.py
FLASK_ENV=production
HASH_DB_PATH=/data/hashes.db
PYTHONUNBUFFERED=1
```

### Web UI

```bash
NEXT_PUBLIC_API_URL=http://localhost:5001
NODE_ENV=production
```

---

## Development Workflow

1. **Local Development:**
   ```bash
   ./setup.sh
   source venv/bin/activate
   python -m cli.extract video.mp4 --frames 60
   ```

2. **Docker Development:**
   ```bash
   docker-compose up
   # API: http://localhost:5001
   # Web: http://localhost:3000
   ```

3. **Testing:**
   ```bash
   ./run_tests.sh
   ```

---

## Migration from Old Structure

### Old → New Mappings

```
experiments/perceptual_hash.py       → core/perceptual_hash.py
Sigil/*                        → experimental/radioactive/
verification/*                       → experimental/verification/
experiments/deprecated_dct_approach/ → experimental/deprecated_dct_approach/
Dockerfile.api                       → docker/Dockerfile.api
Dockerfile.web                       → docker/Dockerfile.web
```

### Breaking Changes

- Imports: `experiments.perceptual_hash` → `core.perceptual_hash`
- CLI: `python experiments/perceptual_hash.py` → `python -m cli.extract`
- Port: 5000 → 5001
- Docker paths updated

See [RESTRUCTURE_SUMMARY.md](RESTRUCTURE_SUMMARY.md) for full migration guide.

---

## What Was Removed

**Directories:**
- `Sigil/` (moved to `experimental/radioactive/`)
- `verification/` (moved to `experimental/verification/`)

**Files:**
- None (all preserved, just reorganized)

---

## Next Development Priorities

1. **API Endpoints:** Create `/extract_hash`, `/compare_hash`, `/store_hash`, `/query_hash`
2. **Web UI:** Update for hash extraction and database management
3. **Tests:** Update imports for new structure
4. **Documentation:** Add API reference, CLI guide, database schema docs

---

## Summary

**Sigil is now clearly organized as:**

- **Primary:** Perceptual hash tracking system (production-ready)
- **Experimental:** Radioactive data marking (research-only, clearly isolated)
- **Documentation:** Comprehensive, academic, honest about limitations

**Ready for:** Production use, academic publication, Hacker News launch

---

**Last Updated:** December 28, 2025
