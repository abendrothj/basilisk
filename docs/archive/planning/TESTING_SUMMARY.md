# 🧪 Project Sigil - Test Suite Summary

## Overview

Complete test coverage for the radioactive data poisoning platform with 75+ tests achieving 85%+ code coverage.

## Test Statistics

| Category | File | Tests | Focus Area |
|----------|------|-------|------------|
| **Core Algorithm** | `test_radioactive_poison.py` | 25+ | RadioactiveMarker, RadioactiveDetector, signature generation |
| **REST API** | `test_api.py` | 30+ | Flask endpoints, image upload, validation, CORS |
| **CLI Tool** | `test_cli.py` | 20+ | Command parsing, batch processing, error handling |
| **Total** | 3 test files | **75+** | **85%+ coverage** |

## Quick Start

```bash
# Install and run all tests
./run_tests.sh

# Run with coverage report
./run_tests.sh coverage

# Run specific category
./run_tests.sh unit    # Core algorithm only
./run_tests.sh api     # API endpoints only
./run_tests.sh cli     # CLI commands only
```

## Test Coverage Highlights

### ✅ Core Algorithm (`test_radioactive_poison.py`)

**What's Tested:**
- Signature generation (deterministic & random)
- Cryptographic seed handling
- Image poisoning pip3eline
- Feature extraction
- Signature persistence (save/load JSON)
- Detection algorithm
- Threshold validation
- Error handling for invalid inputs

**Key Tests:**
- `test_signature_deterministic()` - Same seed produces same signature
- `test_poison_image()` - End-to-end poisoning
- `test_detection_on_clean_model()` - False positive prevention
- `test_full_poison_workflow()` - Integration test

**Coverage:** 90%+

### ✅ API Endpoints (`test_api.py`)

**What's Tested:**
- `/health` - Service health check
- `/api/poison` - Single image poisoning
- `/api/batch` - Batch processing (up to 100 images)
- Input validation (file types, epsilon range)
- Error responses (400, 500)
- Response format (JSON structure)
- Base64 image encoding
- CORS headers

**Key Tests:**
- `test_poison_success()` - Complete API workflow
- `test_poison_invalid_epsilon()` - Validation
- `test_batch_poison_success()` - Multi-image processing
- `test_large_image()` - Performance limits
- `test_cors_headers_present()` - CORS configuration

**Coverage:** 85%+

### ✅ CLI Commands (`test_cli.py`)

**What's Tested:**
- `poison` command (single image)
- `batch` command (directory processing)
- `detect` command structure
- `info` command
- Argument parsing (`--epsilon`, `--signature`, `--device`)
- Output formatting (success messages, progress bars)
- Error messages
- Edge cases (unicode filenames, long paths)

**Key Tests:**
- `test_poison_basic()` - Single image via CLI
- `test_batch_basic()` - Directory batch processing
- `test_poison_with_existing_signature()` - Signature reuse
- `test_batch_mixed_files()` - Filter non-images
- `test_unicode_filenames()` - Character encoding

**Coverage:** 80%+

## Test Architecture

### Fixtures (Reusable Test Components)

```python
@pytest.fixture
def test_image(tmp_path):
    """Creates temporary 224x224 JPEG"""
    img = Image.new('RGB', (224, 224), color='red')
    path = tmp_path / "test.jpg"
    img.save(path)
    return str(path)

@pytest.fixture
def marker():
    """Initialized RadioactiveMarker"""
    return RadioactiveMarker(epsilon=0.01, device='cpu')

@pytest.fixture
def client():
    """Flask test client for API testing"""
    app.config['TESTING'] = True
    return app.test_client()

@pytest.fixture
def runner():
    """Click CLI test runner"""
    return CliRunner()
```

### Test Categories

```python
# Unit test example
def test_signature_generation(marker):
    signature = marker.generate_signature(seed=12345)
    assert len(signature) == 512
    assert np.linalg.norm(signature) ≈ 1.0  # Unit vector

# API test example
def test_poison_endpoint(client, test_image):
    response = client.post('/api/poison', data={
        'image': (test_image, 'test.jpg'),
        'epsilon': '0.01'
    })
    assert response.status_code == 200
    assert json.loads(response.data)['success'] == True

# CLI test example
def test_poison_command(runner, test_image, tmp_path):
    result = runner.invoke(cli, [
        'poison', test_image, str(tmp_path / 'out.jpg')
    ])
    assert result.exit_code == 0
    assert 'Poisoned image saved' in result.output
```

## Running Tests

### Standard Modes

```bash
./run_tests.sh              # All tests, standard output
./run_tests.sh verbose      # Full tracebacks
./run_tests.sh quick        # Minimal output
./run_tests.sh coverage     # With coverage report
```

### Selective Testing

```bash
./run_tests.sh unit         # Core algorithm only
./run_tests.sh api          # API endpoints only
./run_tests.sh cli          # CLI commands only
./run_tests.sh failed       # Re-run failures only
```

### Direct pytest Usage

```bash
# Activate environment
source venv/bin/activate

# Run specific test file
pytest tests/test_radioactive_poison.py -v

# Run specific test class
pytest tests/test_api.py::TestPoisonEndpoint -v

# Run specific test function
pytest tests/test_cli.py::test_poison_basic -v

# Run tests matching keyword
pytest tests/ -k "signature" -v

# Run with coverage
pytest tests/ --cov=Sigil --cov=api --cov-report=html
```

## Edge Cases Covered

### Input Validation
- ✅ Invalid file paths
- ✅ Corrupted image files
- ✅ Wrong file types (.txt, .pdf)
- ✅ Empty filenames
- ✅ Unicode characters in filenames
- ✅ Very long file paths
- ✅ Out-of-range epsilon values

### Size/Performance
- ✅ Very small images (10x10)
- ✅ Very large images (4K)
- ✅ Batch processing (100 images)
- ✅ Empty directories
- ✅ Mixed file types in batch

### Algorithm Edge Cases
- ✅ Zero epsilon (no perturbation)
- ✅ Negative epsilon (reverse direction)
- ✅ Same signature across multiple images
- ✅ Detection with different thresholds

## Continuous Testing

### Pre-commit Hook (Recommended)

```bash
# .git/hooks/pre-commit
#!/bin/bash
./run_tests.sh quick
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

### CI/CD Integration (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: ./setup.sh
      - run: ./run_tests.sh coverage
      - uses: codecov/codecov-action@v3
```

## Coverage Report

```bash
# Generate HTML coverage report
./run_tests.sh coverage

# Open in browser
open htmlcov/index.html
```

**Example Output:**
```
Name                              Stmts   Miss  Cover
-----------------------------------------------------
Sigil/radioactive_poison.py   245     24    90%
api/server.py                       189     28    85%
Sigil/poison_cli.py           156     31    80%
-----------------------------------------------------
TOTAL                               590     83    86%
```

## Test Maintenance

### Adding New Tests

1. **Write test first** (TDD)
2. **Use descriptive names**: `test_poison_image_with_invalid_epsilon()`
3. **Follow Arrange-Act-Assert pattern**
4. **Use existing fixtures** to avoid duplication
5. **Test both success and failure cases**

### Test Checklist for New Features

- [ ] Unit tests for core logic
- [ ] API tests if endpoint added
- [ ] CLI tests if command added
- [ ] Integration test for full workflow
- [ ] Edge case tests
- [ ] Error handling tests
- [ ] Coverage stays > 80%
- [ ] All tests pass locally

## Troubleshooting

### Common Issues

**"Module not found"**
```bash
source venv/bin/activate
pip3 install -r tests/requirements.txt
```

**"CUDA not available"**
```bash
# Tests default to CPU, but verify:
grep "device='cuda'" tests/*.py  # Should return nothing
```

**"Permission denied"**
```bash
chmod +x run_tests.sh
```

**Tests are slow**
```bash
# Run only fast tests
./run_tests.sh quick

# Skip slow tests
pytest tests/ -m "not slow"
```

## Documentation

- **Full Test Docs**: [tests/README.md](tests/README.md)
- **Pytest Config**: [pytest.ini](pytest.ini)
- **Test Requirements**: [tests/requirements.txt](tests/requirements.txt)

## Test Philosophy

> "Tests are the specification. Code is the implementation."

**Our Principles:**
1. **Fast by default** - Tests run in < 30 seconds
2. **Independent** - Tests don't depend on each other
3. **Deterministic** - Same input = same output
4. **Readable** - Tests serve as documentation
5. **Comprehensive** - 85%+ coverage target

## Next Steps

- [ ] Add integration tests for video poisoning (Phase 2)
- [ ] Add performance benchmarks
- [ ] Set up GitHub Actions CI
- [ ] Add mutation testing (pytest-mutpy)
- [ ] Add property-based testing (Hypothesis)

---

**Last Updated:** 2025  
**Maintained By:** Project Sigil Team  
**Run Tests:** `./run_tests.sh`
