# 🎯 SAM3 Roto - Example Scripts

Complete examples demonstrating the optimization features of SAM3 Roto Ultimate.

## 📋 Available Examples

### 1. Memory Optimization (`memory_optimization_example.py`)

Demonstrates the **MemoryManager** system for GPU/CPU monitoring and OOM prevention.

**Features**:
- ✅ Basic memory monitoring
- ✅ Model loading with memory check
- ✅ Manual cleanup
- ✅ Automatic cleanup with thresholds
- ✅ Memory usage history

**Run**:
```bash
python examples/memory_optimization_example.py
```

**What you'll learn**:
- How to monitor CPU/GPU memory in real-time
- How to check if enough memory before loading models
- How to perform manual/automatic cleanup
- How to prevent OOM crashes

---

### 2. Feature Cache (`caching_example.py`)

Demonstrates the **LRU Cache** system for 10-20x speedup on repeated operations.

**Features**:
- ✅ Basic cache operations (get/set)
- ✅ Get-or-compute pattern
- ✅ Decorator pattern (`@cached()`)
- ✅ Image processing cache
- ✅ Cache management

**Run**:
```bash
python examples/caching_example.py
```

**What you'll learn**:
- How to cache expensive computations
- How to use the `@cached()` decorator
- How to cache image processing results
- How to manage cache size and statistics

---

### 3. Batch Processing (`batch_processing_example.py`)

Demonstrates the **BatchProcessor** system for 2x speedup with auto-sizing.

**Features**:
- ✅ Basic batch processing
- ✅ Automatic batch size calculation
- ✅ Image batch processing
- ✅ Video frame processing
- ✅ Progress tracking
- ✅ Batch vs Sequential comparison

**Run**:
```bash
python examples/batch_processing_example.py
```

**What you'll learn**:
- How to process items in optimized batches
- How to auto-calculate optimal batch size
- How to process video frames efficiently
- How to track progress
- Speedup comparison (batch vs sequential)

---

## 🚀 Quick Start

### Run All Examples

```bash
# Memory optimization
python examples/memory_optimization_example.py

# Feature caching
python examples/caching_example.py

# Batch processing
python examples/batch_processing_example.py
```

### Run Individual Examples

Each script is standalone and can be run directly:

```bash
cd /home/user/sam4
python examples/<example_name>.py
```

---

## 📊 Expected Output

### Memory Optimization

```
🚀 SAM3 Roto - Memory Management Examples
================================================================================
Example 1: Basic Memory Monitoring
================================================================================

📊 Current Memory Status:
Memory Statistics:
  CPU: 12.5/32.0 GB (39.1%)
  Process: RSS=4.2 GB, VMS=18.7 GB
  GPU: 6.8/24.0 GB (28.3%), Cached=2.1 GB

✅ Memory usage is healthy
```

### Feature Cache

```
🚀 SAM3 Roto - Feature Cache Examples
================================================================================
Example 2: Get or Compute Pattern
================================================================================

🔄 First call (cache miss):
   💤 Computing (this takes 2 seconds)...
   ✅ Result: {'result': 'computed_value', 'timestamp': 1732723456.789}
   ⏱️  Time: 2.003s

🔄 Second call (cache hit):
   ✅ Result: {'result': 'computed_value', 'timestamp': 1732723456.789}
   ⏱️  Time: 0.001s
   🚀 Speedup: 2003.0x faster!
```

### Batch Processing

```
🚀 SAM3 Roto - Batch Processing Examples
================================================================================
Example 1: Basic Batch Processing
================================================================================

🔄 Processing 100 items...
[BatchProcessor] Processing 100 items with batch_size=16
[BatchProcessor] Batch 1/7 (16 items)
[BatchProcessor] Batch 2/7 (16 items)
...
[BatchProcessor] Completed! Processed 100 items

✅ Processed 100 results
```

---

## 🔧 Configuration

### GPU vs CPU

All examples auto-detect GPU availability. To force CPU mode:

```python
# In the script, change:
processor = BatchProcessor(device="cpu")  # Force CPU
```

### Adjust Settings

Each example has configurable parameters at the top:

```python
# Memory Manager
mm.gc_threshold_percent = 80.0  # Cleanup threshold

# Feature Cache
cache = FeatureCache(
    max_memory_mb=2048.0,  # 2GB cache
    max_entries=100,
)

# Batch Processor
processor = BatchProcessor(
    max_batch_size=32,  # Adjust based on GPU memory
)
```

---

## 📚 Additional Resources

### Documentation

- [`OPTIMIZATIONS_GUIDE.md`](../OPTIMIZATIONS_GUIDE.md) - Complete optimization guide
- [`AUDIT_REPORT.md`](../AUDIT_REPORT.md) - Technical audit and fixes
- [`ROADMAP.md`](../ROADMAP.md) - Development roadmap
- [`FINAL_RECAP.md`](../FINAL_RECAP.md) - Complete session summary

### Source Code

- [`sam3roto/utils/memory_manager.py`](../sam3roto/utils/memory_manager.py) - Memory management
- [`sam3roto/utils/feature_cache.py`](../sam3roto/utils/feature_cache.py) - Feature caching
- [`sam3roto/utils/optimizations.py`](../sam3roto/utils/optimizations.py) - Batch processing

---

## 🐛 Troubleshooting

### ImportError: No module named 'sam3roto'

Make sure you run from the project root:

```bash
cd /home/user/sam4
python examples/<script>.py
```

Or install the package:

```bash
pip install -e .
```

### CUDA Out of Memory

Reduce batch size or cache size:

```python
processor.max_batch_size = 8  # Smaller batches
cache.max_memory_mb = 1024.0  # 1GB instead of 2GB
```

### No GPU Available

The examples will automatically fall back to CPU mode. Performance will be slower but still functional.

---

## 💡 Tips

1. **Start with memory_optimization_example.py** to understand your system's memory constraints
2. **Use caching_example.py** to see how caching speeds up repeated operations
3. **Try batch_processing_example.py** to understand optimal batch sizing for your GPU

---

**Last Updated**: 2025-11-28
**Version**: 1.0
**Author**: Claude Code Ultimate Mode
