# 🐍 Project Basilisk

**Protect your creative work from unauthorized AI training using radioactive data marking.**

> Built on peer-reviewed research from Facebook AI Research (ICML 2020)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Phase](https://img.shields.io/badge/Phase-1%20Images-success)](docs/APPROACH.md)

---

## 🚀 Quick Start (5 minutes)

### 1. Clone and Setup

```bash
git clone https://github.com/abendrothj/basilisk.git
cd basilisk
chmod +x setup.sh run_api.sh run_web.sh
./setup.sh
```

### 2. Poison Your First Image (CLI)

```bash
source venv/bin/activate
python poison-core/poison_cli.py poison my_art.jpg protected_art.jpg
```

**Output:** `protected_art.jpg` (poisoned) + `protected_art_signature.json` (proof of ownership)

### 3. Use the Web Interface

**Terminal 1 - Start API:**
```bash
./run_api.sh
```

**Terminal 2 - Start Web UI:**
```bash
./run_web.sh
```

**Visit:** http://localhost:3000

---

## 🎯 What Does This Do?

### The Problem
AI companies scrape your artwork/photos from the internet and train models on them **without permission or compensation**. Traditional watermarks don't work because they get averaged away during training.

### The Solution: Radioactive Marking
1. **Inject** a unique, imperceptible "signature" into your image's features
2. **Publish** the poisoned image instead of the original
3. **Detect** if AI models trained on your work by testing for your signature
4. **Prove** data theft with cryptographic evidence

### Real-World Use Cases
- **Artists**: Protect portfolios from Midjourney/Stable Diffusion training scrapes
- **Photographers**: Prevent unauthorized use in image generation models
- **Studios**: Safeguard proprietary concept art and designs
- **VFX Artists**: Defense against OpenAI Sora video scraping (Phase 2)

---

## 📚 Documentation

- **[RESEARCH.md](docs/RESEARCH.md)** - Academic citations and paper references
- **[APPROACH.md](docs/APPROACH.md)** - Technical deep dive and mathematics
- **[CREDITS.md](docs/CREDITS.md)** - Attribution and acknowledgments

---

## 🛠️ Project Structure

```
basilisk/
├── poison-core/          # Core radioactive marking algorithm
│   ├── radioactive_poison.py
│   ├── poison_cli.py
│   └── requirements.txt
├── api/                  # Flask API server
│   ├── server.py
│   └── requirements.txt
├── web-ui/              # Next.js frontend
│   ├── app/
│   └── package.json
├── verification/        # Testing and detection
│   └── verify_poison.py
├── docs/                # Documentation
│   ├── RESEARCH.md
│   ├── APPROACH.md
│   └── CREDITS.md
└── README.md
```

---

## 🧪 Testing & Verification

### Run Test Suite

Comprehensive test coverage (75+ tests, 85%+ coverage):

```bash
./run_tests.sh          # Run all tests
./run_tests.sh coverage # With coverage report
./run_tests.sh unit     # Only unit tests
```

**Test Categories:**
- **Unit Tests** - Core algorithm (`test_radioactive_poison.py`)
- **API Tests** - Flask endpoints (`test_api.py`)
- **CLI Tests** - Command-line interface (`test_cli.py`)

See [tests/README.md](tests/README.md) for full documentation.

### Verify Poison Works (Integration Test)

Test that the poison actually survives model training:

```bash
source venv/bin/activate
python verification/verify_poison.py
```

This will:
1. Create a mini-dataset (100 clean + 100 poisoned images)
2. Train a small ResNet-18 model
3. Detect your signature in the trained model
4. Output: **Detection confidence score** (should be > 0.1 for poisoned models)

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

## 🚧 Roadmap

### Phase 1: Images ✅ (Weeks 1-6)
- [x] Core radioactive marking implementation
- [x] CLI tool (single + batch)
- [x] Web UI with drag-and-drop
- [x] Verification environment
- [x] Detection algorithm
- [ ] Performance optimization (GPU acceleration)

### Phase 2: Video 🚧 (Weeks 7-12)
- [ ] Optical flow extraction
- [ ] Temporal signature encoding
- [ ] Video compression robustness testing
- [ ] GPU worker infrastructure (Modal.com)
- [ ] "Sora Defense" beta release

### Phase 3: Multi-Modal (Month 4+)
- [ ] Code protection (ACW integration)
- [ ] Audio protection (AudioSeal integration)
- [ ] Text protection (MarkLLM integration)
- [ ] Unified signature management

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

