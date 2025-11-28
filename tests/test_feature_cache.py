"""
Unit tests for Feature Cache
"""

import pytest
import time
import numpy as np
from sam3roto.utils.feature_cache import (
    FeatureCache,
    CacheEntry,
    get_feature_cache,
    cached,
)


class TestFeatureCache:
    """Test suite for FeatureCache"""

    def test_initialization(self):
        """Test FeatureCache initialization"""
        cache = FeatureCache(max_memory_mb=1024.0, max_entries=100)

        assert cache.max_memory_bytes == 1024 * 1024 * 1024
        assert cache.max_entries == 100
        assert cache._current_size_bytes == 0
        assert len(cache._cache) == 0

    def test_set_and_get(self):
        """Test basic set/get operations"""
        cache = FeatureCache()

        # Set a value
        cache.set("test_key", {"data": "test_value"})

        # Get the value
        value = cache.get("test_key")

        assert value is not None
        assert value["data"] == "test_value"

    def test_get_nonexistent_key(self):
        """Test getting non-existent key returns None"""
        cache = FeatureCache()

        value = cache.get("nonexistent_key")

        assert value is None

    def test_lru_eviction(self):
        """Test LRU eviction when max entries exceeded"""
        cache = FeatureCache(max_entries=3)

        # Add 3 items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Add 4th item - should evict key1 (oldest)
        cache.set("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_reordering(self):
        """Test that accessing a key moves it to end (most recent)"""
        cache = FeatureCache(max_entries=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 (makes it most recent)
        cache.get("key1")

        # Add key4 - should evict key2 (oldest), not key1
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_memory_based_eviction(self):
        """Test eviction based on memory limit"""
        # 1MB cache
        cache = FeatureCache(max_memory_mb=1.0, max_entries=1000)

        # Add large numpy arrays until eviction
        for i in range(10):
            # Each array is ~400KB
            large_array = np.random.randn(100, 1000).astype(np.float32)
            cache.set(f"array_{i}", large_array)

        # Should have evicted some old entries
        stats = cache.get_stats()
        assert stats["memory_mb"] <= 1.0

    def test_get_or_compute(self):
        """Test get_or_compute pattern"""
        cache = FeatureCache()

        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return {"result": "computed"}

        # First call should compute
        result1 = cache.get_or_compute("test_key", compute_fn)
        assert result1["result"] == "computed"
        assert call_count == 1

        # Second call should use cache
        result2 = cache.get_or_compute("test_key", compute_fn)
        assert result2["result"] == "computed"
        assert call_count == 1  # Not incremented

    def test_clear(self):
        """Test clearing cache"""
        cache = FeatureCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache._cache) == 2

        cache.clear()

        assert len(cache._cache) == 0
        assert cache._current_size_bytes == 0

    def test_get_stats(self):
        """Test getting cache statistics"""
        cache = FeatureCache(max_memory_mb=1024.0, max_entries=100)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["max_entries"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert "memory_mb" in stats
        assert "max_memory_mb" in stats

    def test_contains(self):
        """Test __contains__ method"""
        cache = FeatureCache()

        cache.set("key1", "value1")

        assert "key1" in cache
        assert "nonexistent" not in cache

    def test_singleton_pattern(self):
        """Test that get_feature_cache returns singleton"""
        cache1 = get_feature_cache()
        cache2 = get_feature_cache()

        assert cache1 is cache2


class TestCachedDecorator:
    """Test suite for @cached decorator"""

    def test_basic_caching(self):
        """Test basic decorator caching"""
        call_count = 0

        @cached()
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args (cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different args (not cached)
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_multiple_args_caching(self):
        """Test caching with multiple arguments"""
        call_count = 0

        @cached()
        def add(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # Various combinations
        assert add(1, 2) == 3
        assert call_count == 1

        assert add(1, 2) == 3  # Cached
        assert call_count == 1

        assert add(2, 3) == 5  # Not cached
        assert call_count == 2

    def test_kwargs_caching(self):
        """Test caching with keyword arguments"""
        call_count = 0

        @cached()
        def greet(name, greeting="Hello"):
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        assert greet("Alice") == "Hello, Alice!"
        assert call_count == 1

        assert greet("Alice") == "Hello, Alice!"  # Cached
        assert call_count == 1

        assert greet("Alice", greeting="Hi") == "Hi, Alice!"  # Different kwargs
        assert call_count == 2

    def test_custom_cache_instance(self):
        """Test decorator with custom cache instance"""
        custom_cache = FeatureCache(max_entries=2)

        @cached(cache=custom_cache)
        def func(x):
            return x * 2

        func(1)
        func(2)
        func(3)  # Should evict func(1)

        stats = custom_cache.get_stats()
        assert stats["size"] == 2  # Only 2 entries max


class TestCacheEntry:
    """Test suite for CacheEntry dataclass"""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=100,
            timestamp=time.time()
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 100
        assert entry.timestamp > 0


@pytest.fixture
def feature_cache():
    """Fixture providing a fresh FeatureCache instance"""
    cache = FeatureCache(max_memory_mb=100.0, max_entries=50)
    yield cache
    cache.clear()


def test_feature_cache_fixture(feature_cache):
    """Test using FeatureCache fixture"""
    assert feature_cache is not None
    feature_cache.set("test", "value")
    assert feature_cache.get("test") == "value"
