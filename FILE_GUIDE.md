# File Guide: What to Read & When

Quick reference for navigating the documentation.

## 🚀 Getting Started

**Want to use image poisoning right now?**
→ [README.md](README.md) - Main project README

**Want to understand what works for video?**
→ [VIDEO_STATUS.md](VIDEO_STATUS.md) - Current status & capabilities

**Want a quick overview?**
→ [CURRENT_STATE.md](CURRENT_STATE.md) - This file

---

## 🎯 Making a Decision

**Need to decide on CRF 28 approach?**
→ [DECISION_POINT.md](DECISION_POINT.md) - Four options analyzed

**Want to understand why we're stuck?**
→ [DIAGNOSIS.md](DIAGNOSIS.md) - Root cause analysis

---

## 📖 Deep Technical Understanding

**Want the full story of what we tried?**
→ [COMPRESSION_ROBUSTNESS_JOURNEY.md](COMPRESSION_ROBUSTNESS_JOURNEY.md) - Complete timeline

**Want to understand alternative approaches?**
→ [ALTERNATIVE_APPROACHES.md](ALTERNATIVE_APPROACHES.md) - Research review

**Want to understand the system design?**
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Full architecture (aspirational)

---

## 🧪 Using the Code

**Running image poisoning:**
```bash
# See README.md
python poison-core/poison_cli.py poison input.jpg output.jpg
```

**Running video poisoning (CRF 18-23):**
```python
# See VIDEO_STATUS.md → "What Works Right Now"
from frequency_poison import FrequencyDomainVideoMarker
marker = FrequencyDomainVideoMarker(epsilon=0.05)
marker.poison_video('input.mp4', 'poisoned.mp4')
```

**Running tests:**
```bash
# See TESTING_SUMMARY.md
pytest tests/
```

---

## 🔬 Research & Experiments

**What did we try that failed?**
→ [experiments/README.md](experiments/README.md) - Failed approaches

**Why did differentiable codec fail?**
→ Run: `python tests/debug_codec_mismatch.py`

**What's next to try?**
→ `train_cmaes_signature.py` - CMA-ES optimization (ready to run)

---

## 📊 Test Results & Validation

**Image poisoning validation:**
→ [VERIFICATION_PROOF.md](VERIFICATION_PROOF.md) - Proof it works

**Video poisoning validation:**
→ Run: `python tests/test_compression_real.py` - See results across CRF levels

**Statistical validation:**
→ Run: `python tests/test_final_validation.py` - Full rigorous test

---

## Quick Navigation by Topic

### Understanding Compression Robustness
1. [DIAGNOSIS.md](DIAGNOSIS.md) - Why it's hard
2. [COMPRESSION_ROBUSTNESS_JOURNEY.md](COMPRESSION_ROBUSTNESS_JOURNEY.md) - What we tried
3. [DECISION_POINT.md](DECISION_POINT.md) - What to do next

### Understanding the Approach
1. [ALTERNATIVE_APPROACHES.md](ALTERNATIVE_APPROACHES.md) - Why frequency domain?
2. [VIDEO_STATUS.md](VIDEO_STATUS.md) - Technical details
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System design

### Using the Project
1. [README.md](README.md) - Main README
2. [VIDEO_STATUS.md](VIDEO_STATUS.md) - Video usage
3. [TESTING_SUMMARY.md](TESTING_SUMMARY.md) - Test guide

---

## File Organization

```
📁 basilisk/
│
├── 📄 CURRENT_STATE.md              ← START HERE (overview)
├── 📄 VIDEO_STATUS.md               ← Video capabilities & status
├── 📄 DECISION_POINT.md             ← Options for CRF 28
├── 📄 README.md                     ← Main project README (images)
│
├── 📖 Technical Deep Dives:
│   ├── COMPRESSION_ROBUSTNESS_JOURNEY.md
│   ├── DIAGNOSIS.md
│   ├── ALTERNATIVE_APPROACHES.md
│   └── ARCHITECTURE.md
│
├── 🧪 Research:
│   ├── train_cmaes_signature.py     ← Next experiment
│   └── experiments/
│       ├── README.md                ← Failed approaches
│       ├── train_adaptive_signature.py
│       └── train_contrastive_signature.py
│
├── 💻 Working Code:
│   └── poison-core/
│       ├── radioactive_poison.py    ← Images (works!)
│       ├── frequency_poison.py      ← Video (CRF 18-23)
│       └── frequency_detector.py
│
└── ✅ Tests:
    └── tests/
        ├── test_frequency_poison.py
        ├── test_compression_real.py
        ├── test_final_validation.py
        └── debug_codec_mismatch.py
```

---

## Reading Order by Goal

### Goal: Use the tool now
1. README.md
2. VIDEO_STATUS.md (if doing video)
3. Done!

### Goal: Understand the research
1. CURRENT_STATE.md
2. DIAGNOSIS.md
3. COMPRESSION_ROBUSTNESS_JOURNEY.md

### Goal: Continue the research
1. DECISION_POINT.md
2. experiments/README.md
3. train_cmaes_signature.py

### Goal: Contribute
1. README.md
2. TESTING_SUMMARY.md
3. ARCHITECTURE.md
4. COMPRESSION_ROBUSTNESS_JOURNEY.md
