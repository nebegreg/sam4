#!/usr/bin/env python3
"""
Feature Cache Example
Demonstrates LRU caching for 10-20x speedup on repeated operations
"""

import sys
from pathlib import Path
import time
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3roto.utils import (
    get_feature_cache,
    cached,
)
from PIL import Image
import numpy as np


def example_basic_cache_usage():
    """Example 1: Basic cache operations"""
    print("=" * 80)
    print("Example 1: Basic Cache Operations")
    print("=" * 80)

    cache = get_feature_cache()

    # Store value
    print("\n📝 Storing value in cache...")
    cache.set("my_key", {"data": "expensive_computation_result", "value": 42})

    # Retrieve value
    print("🔍 Retrieving value from cache...")
    value = cache.get("my_key")
    print(f"   Retrieved: {value}")

    # Try non-existent key
    print("\n🔍 Trying non-existent key...")
    value = cache.get("non_existent")
    print(f"   Retrieved: {value}")

    # Print statistics
    print("\n📊 Cache Statistics:")
    cache.print_stats()


def example_get_or_compute():
    """Example 2: get_or_compute pattern"""
    print("\n" + "=" * 80)
    print("Example 2: Get or Compute Pattern")
    print("=" * 80)

    cache = get_feature_cache()

    def expensive_computation():
        """Simulate expensive computation"""
        print("   💤 Computing (this takes 2 seconds)...")
        time.sleep(2)
        return {"result": "computed_value", "timestamp": time.time()}

    # First call - will compute
    print("\n🔄 First call (cache miss):")
    start = time.time()
    result1 = cache.get_or_compute("computation_1", expensive_computation)
    elapsed1 = time.time() - start
    print(f"   ✅ Result: {result1}")
    print(f"   ⏱️  Time: {elapsed1:.3f}s")

    # Second call - from cache
    print("\n🔄 Second call (cache hit):")
    start = time.time()
    result2 = cache.get_or_compute("computation_1", expensive_computation)
    elapsed2 = time.time() - start
    print(f"   ✅ Result: {result2}")
    print(f"   ⏱️  Time: {elapsed2:.3f}s")
    print(f"   🚀 Speedup: {elapsed1/elapsed2:.1f}x faster!")


def example_decorator_pattern():
    """Example 3: Using @cached decorator"""
    print("\n" + "=" * 80)
    print("Example 3: Decorator Pattern")
    print("=" * 80)

    @cached()
    def fibonacci(n):
        """Calculate fibonacci number (slow recursive)"""
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    @cached()
    def expensive_multiplication(x, y):
        """Simulate expensive operation"""
        time.sleep(0.5)  # Simulate computation time
        return x * y

    # Test fibonacci
    print("\n🔄 Testing cached fibonacci:")

    print("   First call fib(10):")
    start = time.time()
    result1 = fibonacci(10)
    elapsed1 = time.time() - start
    print(f"   Result: {result1}, Time: {elapsed1:.3f}s")

    print("\n   Second call fib(10):")
    start = time.time()
    result2 = fibonacci(10)
    elapsed2 = time.time() - start
    print(f"   Result: {result2}, Time: {elapsed2:.3f}s")
    print(f"   🚀 Speedup: {elapsed1/elapsed2:.0f}x faster!")

    # Test multiplication
    print("\n🔄 Testing cached multiplication:")

    print("   First call multiply(5, 10):")
    start = time.time()
    result1 = expensive_multiplication(5, 10)
    elapsed1 = time.time() - start
    print(f"   Result: {result1}, Time: {elapsed1:.3f}s")

    print("\n   Second call multiply(5, 10):")
    start = time.time()
    result2 = expensive_multiplication(5, 10)
    elapsed2 = time.time() - start
    print(f"   Result: {result2}, Time: {elapsed2:.3f}s")
    print(f"   🚀 Speedup: {elapsed1/elapsed2:.0f}x faster!")


def example_image_processing_cache():
    """Example 4: Caching image processing results"""
    print("\n" + "=" * 80)
    print("Example 4: Image Processing Cache")
    print("=" * 80)

    cache = get_feature_cache()

    def process_image(image_path: Path):
        """Simulate expensive image processing"""
        print(f"   💤 Processing {image_path.name}...")
        time.sleep(1)  # Simulate processing time

        # Simulate feature extraction
        features = np.random.randn(512)  # 512-d feature vector
        return features

    def process_image_with_cache(image_path: Path):
        """Process image with caching"""
        # Generate cache key from file path
        key = hashlib.md5(str(image_path).encode()).hexdigest()

        # Check cache first
        result = cache.get(key)
        if result is not None:
            print(f"   ✅ Cache HIT for {image_path.name}")
            return result

        # Compute if not cached
        print(f"   ❌ Cache MISS for {image_path.name}")
        result = process_image(image_path)

        # Store in cache
        cache.set(key, result)

        return result

    # Simulate processing same images multiple times
    test_images = [
        Path("image1.jpg"),
        Path("image2.jpg"),
        Path("image1.jpg"),  # Repeat
        Path("image3.jpg"),
        Path("image2.jpg"),  # Repeat
        Path("image1.jpg"),  # Repeat again
    ]

    print("\n🔄 Processing images:")
    start_total = time.time()

    for i, img_path in enumerate(test_images, 1):
        print(f"\n   [{i}/{len(test_images)}] {img_path.name}:")
        features = process_image_with_cache(img_path)
        print(f"      Features shape: {features.shape}")

    elapsed_total = time.time() - start_total

    print(f"\n📊 Total processing time: {elapsed_total:.2f}s")
    print(f"   Without cache would be: {len(test_images) * 1.0:.2f}s")
    print(f"   🚀 Speedup: {(len(test_images) * 1.0) / elapsed_total:.1f}x")

    # Show cache statistics
    print("\n📊 Final Cache Statistics:")
    cache.print_stats()


def example_cache_management():
    """Example 5: Cache management and clearing"""
    print("\n" + "=" * 80)
    print("Example 5: Cache Management")
    print("=" * 80)

    cache = get_feature_cache()

    # Add some entries
    print("\n📝 Adding entries to cache...")
    for i in range(10):
        cache.set(f"key_{i}", {"data": np.random.randn(100), "id": i})

    print("\n📊 Cache Status:")
    stats = cache.get_stats()
    print(f"   Entries: {stats['size']}/{stats['max_entries']}")
    print(f"   Memory: {stats['memory_mb']:.2f}/{stats['max_memory_mb']:.2f} MB")
    print(f"   Hit Rate: {stats['hit_rate']*100:.1f}%")

    # Clear cache
    print("\n🧹 Clearing cache...")
    cache.clear()

    print("\n📊 Cache Status After Clear:")
    stats = cache.get_stats()
    print(f"   Entries: {stats['size']}/{stats['max_entries']}")
    print(f"   Memory: {stats['memory_mb']:.2f}/{stats['max_memory_mb']:.2f} MB")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("🚀 SAM3 Roto - Feature Cache Examples")
    print("=" * 80)

    try:
        # Example 1: Basic usage
        example_basic_cache_usage()

        # Example 2: Get or compute
        example_get_or_compute()

        # Example 3: Decorator pattern
        example_decorator_pattern()

        # Example 4: Image processing
        example_image_processing_cache()

        # Example 5: Cache management
        example_cache_management()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
