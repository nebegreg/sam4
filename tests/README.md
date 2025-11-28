# SAM3 Roto Ultimate - Test Suite

Comprehensive test suite for SAM3 Roto Ultimate with unit and integration tests.

---

## 📋 Test Structure

```
tests/
├── __init__.py                   # Test package initialization
├── conftest.py                   # Pytest configuration and fixtures
├── test_memory_manager.py        # Memory Manager tests (20+ tests)
├── test_feature_cache.py         # Feature Cache tests (15+ tests)
├── test_batch_processor.py       # Batch Processor tests (10+ tests)
├── test_integration.py           # Integration tests (10+ tests)
└── README.md                     # This file
```

---

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Or use the test runner script
./run_tests.sh
```

### Test Runner Options

```bash
# Run only unit tests
./run_tests.sh --unit

# Run only integration tests
./run_tests.sh --integration

# Run fast tests (exclude slow tests)
./run_tests.sh --fast

# Run with verbose output
./run_tests.sh --verbose

# Run with coverage report
./run_tests.sh --coverage

# Run tests in parallel
./run_tests.sh --parallel
```

---

## 📊 Test Categories

### Unit Tests (`-m unit`)

Tests for individual components in isolation:

- **Memory Manager** (`test_memory_manager.py`)
  - 20+ tests covering all MemoryManager functionality
  - Stats tracking, cleanup, pressure detection
  - Optimal batch size calculation
  - Singleton pattern

- **Feature Cache** (`test_feature_cache.py`)
  - 15+ tests for LRU cache operations
  - Basic set/get operations
  - LRU eviction (entries and memory)
  - @cached decorator
  - Statistics and management

- **Batch Processor** (`test_batch_processor.py`)
  - 10+ tests for batch processing
  - Auto batch sizing
  - Progress tracking
  - Empty/single item edge cases
  - Context managers

### Integration Tests (`-m integration`)

Tests for component interactions:

- **Memory + Cache Integration**
  - Memory tracking during caching
  - Cleanup coordination

- **Batch + Memory Integration**
  - Memory cleanup between batches
  - Memory tracking during processing

- **Full Workflow Tests**
  - Complete optimization pipeline
  - Video processing simulation
  - End-to-end workflows

---

## 🏷️ Test Markers

Tests are organized with pytest markers:

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.slow          # Tests that take >5s
@pytest.mark.gpu           # Tests requiring GPU
@pytest.mark.model         # Tests requiring model loading
```

### Running Specific Markers

```bash
# Run only unit tests
pytest tests/ -m "unit"

# Run only integration tests
pytest tests/ -m "integration"

# Run fast tests (exclude slow)
pytest tests/ -m "not slow"

# Run GPU tests
pytest tests/ -m "gpu"

# Combine markers
pytest tests/ -m "unit and not slow"
```

---

## 🧪 Test Fixtures

Common fixtures available in `conftest.py`:

### Data Fixtures

```python
def test_example(mock_image, mock_video_frames, mock_masks):
    """Example using fixtures"""
    # mock_image: PIL Image (512x512 RGB)
    # mock_video_frames: List of 10 numpy arrays (256x256x3)
    # mock_masks: Dict of 3 masks (512x512)
    pass
```

### Directory Fixtures

```python
def test_example(test_data_dir, temp_dir):
    """Example using directory fixtures"""
    # test_data_dir: Path to tests/data/
    # temp_dir: Temporary directory for outputs
    pass
```

### Component Fixtures

```python
def test_example(memory_manager, feature_cache, batch_processor):
    """Example using component fixtures"""
    # Fresh instances of components
    pass
```

---

## 📈 Coverage

Generate coverage reports:

```bash
# HTML report
pytest tests/ --cov=sam3roto --cov-report=html

# View report
firefox htmlcov/index.html

# Terminal report
pytest tests/ --cov=sam3roto --cov-report=term

# Combined
./run_tests.sh --coverage
```

---

## ⚡ Parallel Testing

Run tests in parallel for faster execution:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run in parallel (auto-detect CPUs)
pytest tests/ -n auto

# Or use test runner
./run_tests.sh --parallel
```

---

## 🐛 Debugging Tests

### Verbose Output

```bash
# Show more details
pytest tests/ -v

# Show even more details
pytest tests/ -vv

# Show print statements
pytest tests/ -s
```

### Run Specific Tests

```bash
# Run specific file
pytest tests/test_memory_manager.py

# Run specific class
pytest tests/test_memory_manager.py::TestMemoryManager

# Run specific test
pytest tests/test_memory_manager.py::TestMemoryManager::test_initialization

# Run tests matching pattern
pytest tests/ -k "memory"
```

### Drop into Debugger on Failure

```bash
# Drop into pdb on failure
pytest tests/ --pdb

# Drop into pdb on first failure
pytest tests/ -x --pdb
```

---

## 📝 Writing New Tests

### Unit Test Template

```python
"""
Unit tests for MyComponent
"""

import pytest
from sam3roto.utils import MyComponent


class TestMyComponent:
    """Test suite for MyComponent"""

    def test_initialization(self):
        """Test component initialization"""
        component = MyComponent()
        assert component is not None

    def test_basic_functionality(self):
        """Test basic functionality"""
        component = MyComponent()
        result = component.do_something()
        assert result is not None


@pytest.fixture
def my_component():
    """Fixture providing MyComponent instance"""
    return MyComponent()


def test_with_fixture(my_component):
    """Test using fixture"""
    assert my_component is not None
```

### Integration Test Template

```python
"""
Integration tests for MyFeature
"""

import pytest


@pytest.mark.integration
class TestMyFeatureIntegration:
    """Integration tests for MyFeature"""

    def test_component_interaction(self):
        """Test interaction between components"""
        from sam3roto.utils import ComponentA, ComponentB

        a = ComponentA()
        b = ComponentB()

        # Test interaction
        result = a.interact_with(b)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
def test_complex_workflow():
    """Test complex workflow"""
    # Test implementation
    pass
```

---

## 🔍 Best Practices

### 1. Test Isolation

- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Clear Test Names

```python
# Good
def test_memory_manager_cleans_up_after_threshold():
    pass

# Bad
def test_mm_cleanup():
    pass
```

### 3. Arrange-Act-Assert Pattern

```python
def test_feature_cache_eviction():
    # Arrange
    cache = FeatureCache(max_entries=3)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Act
    cache.set("key4", "value4")

    # Assert
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key4") == "value4"
```

### 4. Use Markers

```python
@pytest.mark.slow
def test_expensive_operation():
    """This test takes a long time"""
    pass

@pytest.mark.gpu
def test_cuda_operation():
    """This test requires GPU"""
    pass
```

### 5. Mock External Dependencies

```python
@pytest.fixture
def mock_model():
    """Mock SAM3 model for testing"""
    class MockModel:
        def predict(self, x):
            return "mock_result"
    return MockModel()
```

---

## 📊 Test Statistics

Current test coverage:

| Component | Tests | Coverage |
|-----------|-------|----------|
| Memory Manager | 20+ | High |
| Feature Cache | 15+ | High |
| Batch Processor | 10+ | High |
| Integration | 10+ | Medium |
| **Total** | **55+** | **High** |

---

## 🎯 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ --cov=sam3roto --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 🆘 Troubleshooting

### Tests Failing

```bash
# Run with verbose output
pytest tests/ -vv

# Run single failing test
pytest tests/test_memory_manager.py::test_specific_test -vv

# Show print statements
pytest tests/ -s
```

### Import Errors

```bash
# Ensure sam3roto is in PYTHONPATH
export PYTHONPATH=/home/user/sam4:$PYTHONPATH

# Or install in editable mode
pip install -e .
```

### Fixture Not Found

Check that:
1. Fixture is defined in `conftest.py` or test file
2. Fixture name matches parameter name
3. Scope is correct (function/class/module/session)

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Markers](https://docs.pytest.org/en/latest/how-to/mark.html)
- [Pytest Fixtures](https://docs.pytest.org/en/latest/how-to/fixtures.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: 2025-11-28
**Total Tests**: 55+
**Coverage**: High
