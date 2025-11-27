"""
SAM3 Roto Utilities
Optimizations, caching, memory management
"""

from .memory_manager import (
    MemoryManager,
    MemoryStats,
    get_memory_manager,
    print_memory_stats,
    cleanup_memory,
)

from .feature_cache import (
    FeatureCache,
    CacheEntry,
    get_feature_cache,
    cached,
)

from .optimizations import (
    BatchProcessor,
    Prefetcher,
    AsyncProcessor,
    torch_inference_mode,
    timed_operation,
    optimize_tensor_for_inference,
    batch_images_to_tensor,
    ProgressTracker,
)

__all__ = [
    # Memory Management
    "MemoryManager",
    "MemoryStats",
    "get_memory_manager",
    "print_memory_stats",
    "cleanup_memory",
    # Feature Cache
    "FeatureCache",
    "CacheEntry",
    "get_feature_cache",
    "cached",
    # Optimizations
    "BatchProcessor",
    "Prefetcher",
    "AsyncProcessor",
    "torch_inference_mode",
    "timed_operation",
    "optimize_tensor_for_inference",
    "batch_images_to_tensor",
    "ProgressTracker",
]
