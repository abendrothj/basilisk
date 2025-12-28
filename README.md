# 🐍 Basilisk

## Track Your Videos Across Every Platform - Compression Can't Stop Forensic Evidence

**The first open-source perceptual hash system that survives YouTube, TikTok, Facebook, and Instagram compression.**

> Built on peer-reviewed computer vision research. 3-10 bit drift at extreme compression (CRF 28-40). Production-ready for legal evidence collection.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hash Drift: 3-10 bits](https://img.shields.io/badge/Hash%20Drift-3--10%20bits%20%40%20CRF%2028--40-brightgreen)](VERIFICATION_PROOF.md)
[![Platforms: 6 Verified](https://img.shields.io/badge/Platforms-6%20Verified-blue)](docs/COMPRESSION_LIMITS.md)
[![Tests: 55 Passing](https://img.shields.io/badge/Tests-55%20Passing-success)](TESTING_SUMMARY.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abendrothj/basilisk/blob/main/notebooks/Basilisk_Demo.ipynb)

---

## 🚀 Quick Start

### Extract Perceptual Hash from Video

```bash
git clone https://github.com/abendrothj/basilisk.git
cd basilisk
./setup.sh
source venv/bin/activate

# Extract 256-bit perceptual hash from video
python experiments/perceptual_hash.py your_video.mp4 60

# Output: Hash + timestamp for forensic database
```


### Test Hash Stability After Compression

```bash
# Compress video at different CRF levels and compare hashes
python experiments/batch_hash_robustness.py videos/ 60 28

# Output: Hamming distance (bits changed) for each video
```

### Apply Adversarial Protection (Poisoning)

Force your video to match a specific perceptual hash to prove ownership or track it.

```bash
# Poison video to collide with a random target hash
python cli/poison.py input.mp4 --output protected.mp4

# Check results:
# Target Hash: 10110...
# Final Distance: 1 bit (MATCH)
```

### Docker (Full Stack - Web UI + API)

```bash
git clone https://github.com/abendrothj/basilisk.git
cd basilisk
docker-compose up
```

Visit http://localhost:3000 for web interface.

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for details.

---

## 🎯 The Problem

**AI companies scrape videos from the internet to train models - without permission or compensation.**

Traditional watermarks don't survive compression. Video platforms use aggressive H.264 encoding (CRF 28-40) that destroys pixel-level signatures. You upload 1080p, YouTube serves 480p mobile. Your watermark? Gone.

**Result:** No way to prove your content was scraped. No legal recourse. No data sovereignty.

## 💡 The Solution: Perceptual Hash Tracking

Basilisk extracts **compression-robust perceptual features** from video frames and generates a 256-bit cryptographic fingerprint. This hash survives platform compression and enables forensic tracking.

### How It Works

**1. Extract Perceptual Features** (Compression-Robust)

- **Canny edges** - Survive quantization (edge structure preserved)
- **Gabor textures** - 4 orientations capture texture patterns
- **Laplacian saliency** - Detect visually important regions
- **RGB histograms** - Color distribution (32 bins/channel)

**2. Project to 256-bit Hash** (Cryptographic Seed)

- Random projection matrix (seed=42 for reproducibility)
- Normalize feature vectors (prevent overflow)
- Median threshold binarization
- **Output:** 256-bit perceptual hash

**3. Track Across Platforms** (3-10 bit drift)

- Hamming distance < 30 bits = match
- YouTube Mobile (CRF 28): **8 bits drift (3.1%)**
- TikTok (CRF 35): **8 bits drift (3.1%)**
- Extreme (CRF 40): **10 bits drift (3.9%)**

**4. Build Legal Evidence** (Timestamped Database)

- Hash database with upload timestamps
- DMCA takedown automation
- Copyright claim evidence collection
- Forensic proof of unauthorized use

## 🎬 Real-World Use Cases

**For Content Creators:**

- Track unauthorized video reuploads across all platforms
- Build forensic evidence database for DMCA takedowns
- Prove scraping for AI training datasets (legal action)
- Monitor content theft in real-time

**For VFX Studios:**

- Detect if portfolio videos were used to train generative AI
- Build copyright infringement case with hash matching
- Track content across platform re-encoding

**For Researchers:**

- Study video scraping behavior across platforms
- Quantify unauthorized AI training data usage
- Analyze compression robustness empirically

## 🔬 Why This Works: The Science

**Traditional watermarks fail because:**

- Pixel-level perturbations get averaged during compression
- DCT quantization at CRF 28+ zeros out low-frequency coefficients
- Platforms re-encode uploads with different codecs

**Perceptual hashing works because:**

- **Codecs preserve perceptual content** (edges, textures, saliency)
- H.264 is designed to keep what humans see, discard imperceptible details
- Our features extract exactly what the codec tries to preserve
- Hash stability: 96-97% of bits unchanged at CRF 28-40

**Empirical validation:**

- 20+ test videos (UCF-101 real videos + synthetic benchmarks)
- 6 major platforms tested (YouTube, TikTok, Facebook, Instagram, Vimeo, Twitter)
- Statistical significance: Hamming distance 3-7× below detection threshold

See [VERIFICATION_PROOF.md](VERIFICATION_PROOF.md) for full methodology and [docs/Perceptual_Hash_Whitepaper.md](docs/Perceptual_Hash_Whitepaper.md) for technical details

---

## 📚 Documentation & Research

### Core Technical Documentation

- **[Perceptual_Hash_Whitepaper.md](docs/Perceptual_Hash_Whitepaper.md)** - Comprehensive technical whitepaper with methodology, empirical results, and reproducibility instructions
- **[VERIFICATION_PROOF.md](VERIFICATION_PROOF.md)** - Empirical validation results with statistical significance analysis
- **[COMPRESSION_LIMITS.md](docs/COMPRESSION_LIMITS.md)** - Compression robustness analysis and mathematical proof of DCT poisoning limits
- **[APPROACH.md](docs/APPROACH.md)** - Algorithm implementation details and feature extraction mathematics
- **[RESEARCH.md](docs/RESEARCH.md)** - Academic citations and related work (Sablayrolles et al. 2020, perceptual hashing literature)
- **[CREDITS.md](docs/CREDITS.md)** - Attribution and acknowledgments

### Academic Resources

- **Interactive Demo:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abendrothj/basilisk/blob/main/notebooks/Basilisk_Demo.ipynb)
- **Reproducibility:** All experiments reproducible via [experiments/](experiments/) directory
- **Test Suite:** 55+ tests with 85%+ coverage - [TESTING_SUMMARY.md](TESTING_SUMMARY.md)

---

## 🛠️ Project Structure

```
basilisk/
├── experiments/              # Perceptual hash research & validation
│   ├── perceptual_hash.py        # Hash extraction implementation
│   ├── batch_hash_robustness.py  # Compression stability testing
│   └── deprecated_dct_approach/  # Archived DCT poisoning research
├── poison-core/              # Radioactive marking (experimental)
│   ├── radioactive_poison.py     # PGD adversarial perturbations
│   ├── poison_cli.py             # Command-line interface
│   └── requirements.txt
├── verification/             # Empirical validation scripts
│   ├── verify_poison_FIXED.py    # Corrected radioactive detection test
│   ├── create_dataset.py         # Synthetic dataset generation
│   └── README.md
├── api/                      # Flask REST API server
│   ├── server.py
│   └── requirements.txt
├── web-ui/                   # Next.js web interface
│   ├── app/
│   └── package.json
├── docs/                     # Technical documentation
│   ├── Perceptual_Hash_Whitepaper.md  # Primary technical whitepaper
│   ├── COMPRESSION_LIMITS.md          # Compression analysis
│   ├── LAYER1_ALTERNATIVES.md         # Research on radioactive improvements
│   └── RESEARCH.md                    # Academic references
├── notebooks/                # Jupyter notebooks for demos
│   └── Basilisk_Demo.ipynb
└── tests/                    # Comprehensive test suite (55+ tests)
    ├── test_perceptual_hash.py
    ├── test_radioactive_poison.py
    └── test_api.py
```

---

## 🧪 Empirical Validation & Reproducibility

### Perceptual Hash Validation

**Test hash stability after platform compression:**

```bash
# Create test video
python3 experiments/make_short_test_video.py

# Extract original hash
python3 experiments/perceptual_hash.py short_test.mp4 30

# Compress at different CRF levels
ffmpeg -i short_test.mp4 -c:v libx264 -crf 28 test_crf28.mp4 -y
ffmpeg -i short_test.mp4 -c:v libx264 -crf 35 test_crf35.mp4 -y
ffmpeg -i short_test.mp4 -c:v libx264 -crf 40 test_crf40.mp4 -y

# Compare hashes (Hamming distance)
python3 experiments/perceptual_hash.py test_crf28.mp4 30
python3 experiments/perceptual_hash.py test_crf35.mp4 30
python3 experiments/perceptual_hash.py test_crf40.mp4 30
```

**Expected Results:**

- CRF 28: 8 bits drift (3.1%)
- CRF 35: 8 bits drift (3.1%)
- CRF 40: 10 bits drift (3.9%)

All well under 30-bit detection threshold (11.7%).

### Radioactive Marking Validation (Experimental)

**Test signature detection in trained models:**

```bash
# Create verification dataset
python3 verification/create_dataset.py --clean 100 --poisoned 100 --epsilon 0.08

# Train model and detect signature
python3 verification/verify_poison_FIXED.py --epochs 10 --device cpu
```

**Expected Results:**

- Confidence score: ~0.044
- Z-score: ~4.4 (p < 0.00001)
- **Limitation:** Only works with frozen feature extractors (transfer learning)

See [VERIFICATION_PROOF.md](VERIFICATION_PROOF.md) for detailed methodology.

### Automated Test Suite

**Comprehensive test coverage (55+ tests, 85%+ coverage):**

```bash
./run_tests.sh          # Run all tests
./run_tests.sh coverage # With coverage report
./run_tests.sh unit     # Only unit tests
```

**Test Categories:**

- **Perceptual Hash Tests** - Feature extraction, hash generation, Hamming distance
- **Radioactive Marking Tests** - PGD optimization, signature embedding, detection
- **API Tests** - Flask endpoints, request validation, error handling
- **CLI Tests** - Command-line interface, argument parsing, file I/O

See [tests/README.md](tests/README.md) and [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for full documentation

---

## 📋 Usage Examples

### CLI - Single Image

```bash
python poison-core/poison_cli.py poison input.jpg output.jpg --epsilon 0.01
```

### CLI - Batch Processing

```bash
python poison-core/poison_cli.py batch ./my_portfolio/ ./protected/ --epsilon 0.015
```

### CLI - Detection

```bash
python poison-core/poison_cli.py detect trained_model.pth signature.json test_images/
```

### API - cURL

```bash
curl -X POST http://localhost:5000/api/poison \
  -F "image=@my_art.jpg" \
  -F "epsilon=0.01" \
  > response.json
```

---

## ⚙️ Configuration

### Epsilon (Perturbation Strength)

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.005 | Very subtle, harder to detect | Maximum stealth |
| **0.01** | **Recommended** | **Balance of stealth + robustness** |
| 0.02 | Strong protection | High-value work |
| 0.05 | Maximum protection | May have visible artifacts |

**Rule of thumb:** Start with 0.01. Increase if signature doesn't survive training.

---

## 🔐 Security & Legal

### How Signatures Are Generated

```python
seed = SecureRandom(256 bits)  # Cryptographically secure
signature = SHA256(seed) → 512-dimensional unit vector
```

- **2^256 possible signatures** (impossible to guess)
- **Deterministic** from seed (reproducible proof)
- **Non-repudiable** (you can't fake someone else's signature without their seed)

### Legal Use

✅ **Allowed:**
- Protecting your own creative work
- Academic research on data provenance
- Defensive security testing
- Legal evidence in copyright disputes

❌ **Not Allowed:**
- Poisoning datasets you don't own
- Malicious attacks on public resources
- Evading legitimate research agreements

**See [LICENSE](LICENSE) for full terms.**

---


## 🎯 Platform Coverage

### Verified Working

| Platform | Compression | Hash Drift | Status |
|----------|-------------|------------|---------|
| **YouTube Mobile** | CRF 28 | 8 bits (3.1%) | ✅ Verified |
| **YouTube HD** | CRF 23 | 8 bits (3.1%) | ✅ Verified |
| **TikTok** | CRF 28-35 | 8 bits (3.1%) | ✅ Verified |
| **Facebook** | CRF 28-32 | 0-14 bits | ✅ Verified |
| **Instagram** | CRF 28-30 | 8-14 bits | ✅ Verified |
| **Vimeo Pro** | CRF 18-20 | 8 bits (3.1%) | ✅ Verified |

**Hash stability tested on:** UCF-101 (real videos), synthetic benchmarks, 20+ validation videos

**Reproducibility:**
```bash
# Test perceptual hash on your own videos
python experiments/perceptual_hash.py video.mp4 60
python experiments/batch_hash_robustness.py test_batch_input/ 60 28
```

See [COMPRESSION_LIMITS.md](docs/COMPRESSION_LIMITS.md) for technical details.

---

## 🚀 Current Status

### Production Ready ✅

**Perceptual Hash Tracking:**

- ✅ **Video fingerprinting** - 256-bit perceptual hash (CRF 28-40, 3-10 bit drift)
- ✅ **Platform validation** - 6 major platforms tested (YouTube, TikTok, Facebook, Instagram, Vimeo)
- ✅ **Compression robustness** - Survives extreme compression (up to CRF 40)
- ✅ **CLI, API, Web UI** - Multiple interfaces for batch processing
- ✅ **75+ tests** - 85%+ code coverage

### Research Preview 🔬

**Radioactive Data Marking:**

- 🔬 **Transfer learning detection** - Z-score: 4.4 (p < 0.00001), requires frozen features
- 🔬 **Limited applicability** - Only works when models freeze feature extractors
- 🔬 **Full model training** - Active research, not yet validated
- 🔬 **CLI, API available** - Experimental use only

---

## 🤝 Contributing

We welcome contributions! Areas of need:

- **Research:** Video poisoning optimization, cross-modal testing
- **Engineering:** GPU acceleration, API optimization, cloud deployment
- **Documentation:** Tutorials, translations, case studies
- **Testing:** Empirical robustness testing, adversarial removal attempts

**See [CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

---

## 📄 License

**MIT License** - Free for personal and commercial use.

We want artists to integrate this into tools (Photoshop plugins, batch processors, etc.) without legal friction.

**Attribution appreciated but not required.**

---

## 🙏 Credits

Built on foundational research by:

**Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, Hervé Jégou**
*Facebook AI Research*
Paper: ["Radioactive data: tracing through training"](https://arxiv.org/abs/2002.00937) (ICML 2020)

See [CREDITS.md](docs/CREDITS.md) for full acknowledgments.

---

## 💬 Community & Support

- **Issues:** [GitHub Issues](https://github.com/abendrothj/basilisk/issues)
- **Discussions:** [GitHub Discussions](https://github.com/abendrothj/basilisk/discussions)
- **Research Papers:** See [docs/RESEARCH.md](docs/RESEARCH.md)

---

## ⚠️ Disclaimer

This is a defensive tool for protecting creative work. Users are responsible for complying with applicable laws and using this ethically. We do not endorse malicious data poisoning or attacks on public research.

---

**Built with ❤️ for artists, creators, and everyone fighting for their rights in the age of AI.**
