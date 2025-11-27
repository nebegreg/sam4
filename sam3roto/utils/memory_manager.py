"""
Memory Management System for SAM3 Roto
Handles GPU/CPU memory allocation, monitoring, and optimization
"""

from __future__ import annotations
import gc
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil
import torch
import numpy as np


@dataclass
class MemoryStats:
    """Memory statistics snapshot"""
    timestamp: float

    # CPU Memory
    cpu_total_gb: float
    cpu_used_gb: float
    cpu_available_gb: float
    cpu_percent: float

    # GPU Memory (if available)
    gpu_total_gb: Optional[float] = None
    gpu_allocated_gb: Optional[float] = None
    gpu_cached_gb: Optional[float] = None
    gpu_free_gb: Optional[float] = None
    gpu_percent: Optional[float] = None

    # Process Memory
    process_rss_gb: float = 0.0
    process_vms_gb: float = 0.0

    def __str__(self) -> str:
        lines = [
            "Memory Statistics:",
            f"  CPU: {self.cpu_used_gb:.2f}/{self.cpu_total_gb:.2f} GB ({self.cpu_percent:.1f}%)",
            f"  Process: RSS={self.process_rss_gb:.2f} GB, VMS={self.process_vms_gb:.2f} GB",
        ]

        if self.gpu_total_gb is not None:
            lines.append(
                f"  GPU: {self.gpu_allocated_gb:.2f}/{self.gpu_total_gb:.2f} GB "
                f"({self.gpu_percent:.1f}%), Cached={self.gpu_cached_gb:.2f} GB"
            )

        return "\n".join(lines)


class MemoryManager:
    """
    Central memory management system

    Features:
    - Monitor CPU/GPU memory usage
    - Automatic garbage collection
    - Memory pressure detection
    - GPU cache management
    - Memory leak detection
    """

    def __init__(self, auto_gc: bool = True, gc_threshold_percent: float = 80.0):
        """
        Args:
            auto_gc: Enable automatic garbage collection
            gc_threshold_percent: Trigger GC when memory usage exceeds this %
        """
        self.auto_gc = auto_gc
        self.gc_threshold_percent = gc_threshold_percent

        self.has_gpu = torch.cuda.is_available()
        self._lock = threading.Lock()
        self._stats_history: list[MemoryStats] = []
        self._max_history = 100

        # Memory warnings
        self._last_warning_time = 0.0
        self._warning_cooldown = 10.0  # seconds

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # CPU Memory
        cpu_mem = psutil.virtual_memory()

        # Process Memory
        process = psutil.Process()
        process_mem = process.memory_info()

        stats = MemoryStats(
            timestamp=time.time(),
            cpu_total_gb=cpu_mem.total / (1024**3),
            cpu_used_gb=cpu_mem.used / (1024**3),
            cpu_available_gb=cpu_mem.available / (1024**3),
            cpu_percent=cpu_mem.percent,
            process_rss_gb=process_mem.rss / (1024**3),
            process_vms_gb=process_mem.vms / (1024**3),
        )

        # GPU Memory (if available)
        if self.has_gpu:
            try:
                gpu_mem = torch.cuda.memory_stats()
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated()
                gpu_cached = torch.cuda.memory_reserved()

                stats.gpu_total_gb = gpu_total / (1024**3)
                stats.gpu_allocated_gb = gpu_allocated / (1024**3)
                stats.gpu_cached_gb = gpu_cached / (1024**3)
                stats.gpu_free_gb = (gpu_total - gpu_allocated) / (1024**3)
                stats.gpu_percent = (gpu_allocated / gpu_total) * 100
            except Exception as e:
                print(f"[MemoryManager] Warning: Could not get GPU stats: {e}")

        # Store in history
        with self._lock:
            self._stats_history.append(stats)
            if len(self._stats_history) > self._max_history:
                self._stats_history.pop(0)

        return stats

    def check_memory_pressure(self, stats: Optional[MemoryStats] = None) -> bool:
        """
        Check if system is under memory pressure

        Returns:
            True if memory usage is high and action should be taken
        """
        if stats is None:
            stats = self.get_stats()

        # Check CPU memory
        if stats.cpu_percent > self.gc_threshold_percent:
            return True

        # Check GPU memory (if available)
        if stats.gpu_percent is not None and stats.gpu_percent > self.gc_threshold_percent:
            return True

        return False

    def cleanup(self, aggressive: bool = False) -> MemoryStats:
        """
        Perform memory cleanup

        Args:
            aggressive: If True, perform more aggressive cleanup

        Returns:
            Memory stats after cleanup
        """
        stats_before = self.get_stats()

        print(f"[MemoryManager] Starting cleanup (aggressive={aggressive})...")
        print(f"[MemoryManager] Before:\n{stats_before}")

        # Python garbage collection
        collected = gc.collect()
        print(f"[MemoryManager] Collected {collected} objects")

        # GPU cleanup (if available)
        if self.has_gpu:
            # Clear GPU cache
            torch.cuda.empty_cache()
            print("[MemoryManager] GPU cache cleared")

            if aggressive:
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                print("[MemoryManager] GPU synchronized and stats reset")

        # Get stats after cleanup
        stats_after = self.get_stats()

        # Calculate freed memory
        cpu_freed = stats_before.cpu_used_gb - stats_after.cpu_used_gb
        if stats_before.gpu_allocated_gb is not None:
            gpu_freed = stats_before.gpu_allocated_gb - stats_after.gpu_allocated_gb
            print(f"[MemoryManager] Freed: CPU={cpu_freed:.2f} GB, GPU={gpu_freed:.2f} GB")
        else:
            print(f"[MemoryManager] Freed: CPU={cpu_freed:.2f} GB")

        print(f"[MemoryManager] After:\n{stats_after}")

        return stats_after

    def auto_cleanup_if_needed(self) -> bool:
        """
        Automatically cleanup if under memory pressure

        Returns:
            True if cleanup was performed
        """
        if not self.auto_gc:
            return False

        stats = self.get_stats()

        if self.check_memory_pressure(stats):
            # Check cooldown
            current_time = time.time()
            if current_time - self._last_warning_time < self._warning_cooldown:
                return False

            self._last_warning_time = current_time

            print(f"[MemoryManager] ⚠️  Memory pressure detected!")
            self.cleanup(aggressive=False)
            return True

        return False

    def estimate_available_for_model(self) -> Dict[str, float]:
        """
        Estimate available memory for model loading

        Returns:
            Dict with 'cpu_gb' and 'gpu_gb' estimates
        """
        stats = self.get_stats()

        # Reserve 20% for system/overhead
        safety_margin = 0.8

        result = {
            'cpu_gb': stats.cpu_available_gb * safety_margin
        }

        if stats.gpu_free_gb is not None:
            result['gpu_gb'] = stats.gpu_free_gb * safety_margin

        return result

    def can_load_model(self, estimated_size_gb: float, device: str = "cuda") -> bool:
        """
        Check if there's enough memory to load a model

        Args:
            estimated_size_gb: Estimated model size in GB
            device: 'cuda' or 'cpu'

        Returns:
            True if model can be loaded
        """
        available = self.estimate_available_for_model()

        if device == "cuda":
            if 'gpu_gb' not in available:
                print("[MemoryManager] GPU not available")
                return False
            return available['gpu_gb'] >= estimated_size_gb
        else:
            return available['cpu_gb'] >= estimated_size_gb

    def get_optimal_batch_size(
        self,
        single_item_memory_gb: float,
        max_batch_size: int = 32,
        device: str = "cuda"
    ) -> int:
        """
        Calculate optimal batch size based on available memory

        Args:
            single_item_memory_gb: Memory required for single item
            max_batch_size: Maximum allowed batch size
            device: 'cuda' or 'cpu'

        Returns:
            Optimal batch size
        """
        available = self.estimate_available_for_model()

        if device == "cuda" and 'gpu_gb' in available:
            available_gb = available['gpu_gb']
        else:
            available_gb = available['cpu_gb']

        # Calculate batch size (leave some margin)
        if single_item_memory_gb <= 0:
            return max_batch_size

        calculated_batch = int(available_gb / single_item_memory_gb)
        optimal = min(calculated_batch, max_batch_size)

        return max(1, optimal)  # At least 1

    def print_summary(self):
        """Print memory summary"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print(stats)

        # Show trend if history available
        with self._lock:
            if len(self._stats_history) >= 2:
                first = self._stats_history[0]
                last = self._stats_history[-1]

                cpu_trend = last.cpu_used_gb - first.cpu_used_gb
                trend_str = "↑" if cpu_trend > 0 else "↓"
                print(f"\n  Trend: CPU {trend_str} {abs(cpu_trend):.2f} GB over {len(self._stats_history)} samples")

                if last.gpu_allocated_gb is not None and first.gpu_allocated_gb is not None:
                    gpu_trend = last.gpu_allocated_gb - first.gpu_allocated_gb
                    trend_str = "↑" if gpu_trend > 0 else "↓"
                    print(f"         GPU {trend_str} {abs(gpu_trend):.2f} GB")

        print("="*60 + "\n")


# Global singleton instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def print_memory_stats():
    """Convenience function to print memory stats"""
    mm = get_memory_manager()
    mm.print_summary()


def cleanup_memory(aggressive: bool = False):
    """Convenience function to cleanup memory"""
    mm = get_memory_manager()
    mm.cleanup(aggressive=aggressive)
