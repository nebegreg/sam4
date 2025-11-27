"""
Intelligent Feature Cache with LRU Eviction
Caches expensive computations like backbone features, embeddings, etc.
"""

from __future__ import annotations
import hashlib
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Callable, Dict
import numpy as np
import torch


@dataclass
class CacheEntry:
    """Single cache entry"""
    key: str
    value: Any
    size_bytes: int
    access_count: int
    last_access: float
    created: float

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


class FeatureCache:
    """
    LRU cache for expensive computations

    Features:
    - LRU eviction policy
    - Memory-based size limits
    - Thread-safe
    - Optional disk persistence
    - Statistics tracking
    """

    def __init__(
        self,
        max_memory_mb: float = 2048.0,
        max_entries: int = 100,
        enable_disk_cache: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            max_memory_mb: Maximum cache size in MB
            max_entries: Maximum number of cached entries
            enable_disk_cache: Enable disk-based caching
            cache_dir: Directory for disk cache
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.enable_disk_cache = enable_disk_cache

        # Cache storage (OrderedDict for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size_bytes = 0
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Disk cache
        if enable_disk_cache:
            if cache_dir is None:
                cache_dir = Path.cwd() / ".sam3roto_cache" / "features"
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _compute_key(self, *args, **kwargs) -> str:
        """Compute cache key from arguments"""
        # Create stable hash from arguments
        key_data = pickle.dumps((args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        key_hash = hashlib.sha256(key_data).hexdigest()[:16]
        return key_hash

    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes"""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(
                self._estimate_size(k) + self._estimate_size(v)
                for k, v in obj.items()
            )
        else:
            # Fallback: use pickle size
            try:
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            except:
                return 1024  # Assume 1KB if can't estimate

    def _evict_lru(self, required_bytes: int = 0):
        """Evict least recently used entries to make space"""
        with self._lock:
            while self._cache and (
                self._current_size_bytes + required_bytes > self.max_memory_bytes
                or len(self._cache) >= self.max_entries
            ):
                # Remove oldest entry (first in OrderedDict)
                key, entry = self._cache.popitem(last=False)
                self._current_size_bytes -= entry.size_bytes
                self._evictions += 1

                print(
                    f"[FeatureCache] Evicted: {key[:8]}... "
                    f"(size={entry.size_bytes/1024/1024:.2f} MB, "
                    f"hits={entry.access_count})"
                )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                entry = self._cache.pop(key)
                entry.update_access()
                self._cache[key] = entry

                self._hits += 1
                return entry.value

            self._misses += 1

            # Try disk cache if enabled
            if self.enable_disk_cache and self.cache_dir:
                disk_path = self.cache_dir / f"{key}.pkl"
                if disk_path.exists():
                    try:
                        with open(disk_path, 'rb') as f:
                            value = pickle.load(f)
                        print(f"[FeatureCache] Loaded from disk: {key[:8]}...")
                        # Add to memory cache
                        self.set(key, value)
                        return value
                    except Exception as e:
                        print(f"[FeatureCache] Error loading from disk: {e}")

            return None

    def set(self, key: str, value: Any, force: bool = False):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            force: Force caching even if size exceeds limits
        """
        size_bytes = self._estimate_size(value)

        # Don't cache if too large (unless forced)
        if not force and size_bytes > self.max_memory_bytes * 0.5:
            print(
                f"[FeatureCache] Value too large to cache: "
                f"{size_bytes/1024/1024:.2f} MB > "
                f"{self.max_memory_bytes/1024/1024*0.5:.2f} MB"
            )
            return

        with self._lock:
            # Remove if already exists (to update)
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_size_bytes -= old_entry.size_bytes

            # Make space
            self._evict_lru(required_bytes=size_bytes)

            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                access_count=1,
                last_access=time.time(),
                created=time.time(),
            )

            self._cache[key] = entry
            self._current_size_bytes += size_bytes

            # Save to disk if enabled
            if self.enable_disk_cache and self.cache_dir:
                disk_path = self.cache_dir / f"{key}.pkl"
                try:
                    with open(disk_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"[FeatureCache] Error saving to disk: {e}")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Get from cache or compute if not present

        Args:
            key: Cache key (or None to compute from args/kwargs)
            compute_fn: Function to compute value if not cached
            *args, **kwargs: Arguments for compute_fn

        Returns:
            Cached or computed value
        """
        # Compute key if not provided
        if key is None:
            key = self._compute_key(*args, **kwargs)

        # Try to get from cache
        value = self.get(key)

        if value is not None:
            return value

        # Compute
        print(f"[FeatureCache] Computing: {key[:8]}...")
        value = compute_fn(*args, **kwargs)

        # Cache
        self.set(key, value)

        return value

    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            print("[FeatureCache] Cache cleared")

            # Clear disk cache if enabled
            if self.enable_disk_cache and self.cache_dir:
                for file in self.cache_dir.glob("*.pkl"):
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f"[FeatureCache] Error deleting {file}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_memory_bytes / (1024 * 1024),
                'utilization': self._current_size_bytes / self.max_memory_bytes,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
            }

    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("Feature Cache Statistics:")
        print(f"  Entries: {stats['entries']}/{self.max_entries}")
        print(f"  Size: {stats['size_mb']:.2f}/{stats['max_size_mb']:.2f} MB ({stats['utilization']*100:.1f}%)")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}, Evictions: {stats['evictions']}")
        print(f"  Hit Rate: {stats['hit_rate']*100:.1f}%")
        print("="*60 + "\n")


# Global cache instance
_global_feature_cache: Optional[FeatureCache] = None


def get_feature_cache() -> FeatureCache:
    """Get global feature cache instance"""
    global _global_feature_cache
    if _global_feature_cache is None:
        _global_feature_cache = FeatureCache(
            max_memory_mb=2048.0,  # 2GB default
            max_entries=100,
            enable_disk_cache=False,  # Disabled by default (can be slow)
        )
    return _global_feature_cache


def cached(key_fn: Optional[Callable] = None):
    """
    Decorator to cache function results

    Args:
        key_fn: Optional function to compute cache key from arguments

    Example:
        @cached()
        def expensive_computation(x, y):
            return x * y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_feature_cache()

            # Compute key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = cache._compute_key(func.__name__, *args, **kwargs)

            # Get or compute
            return cache.get_or_compute(key, func, *args, **kwargs)

        return wrapper
    return decorator
