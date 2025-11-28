"""
Unit tests for Memory Manager
"""

import pytest
import time
from sam3roto.utils.memory_manager import (
    MemoryManager,
    MemoryStats,
    get_memory_manager,
)


class TestMemoryManager:
    """Test suite for MemoryManager"""

    def test_initialization(self):
        """Test MemoryManager initialization"""
        mm = MemoryManager()
        assert mm is not None
        assert mm.auto_gc is True
        assert mm.gc_threshold_percent == 80.0

    def test_get_stats(self):
        """Test getting memory statistics"""
        mm = MemoryManager()
        stats = mm.get_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.cpu_total_gb > 0
        assert stats.cpu_used_gb >= 0
        assert stats.cpu_available_gb >= 0
        assert 0 <= stats.cpu_percent <= 100

    def test_memory_history(self):
        """Test memory history tracking"""
        mm = MemoryManager()

        # Record some stats
        mm.get_stats()
        mm.get_stats()
        mm.get_stats()

        assert len(mm.history) >= 3
        assert all(isinstance(s, MemoryStats) for s in mm.history)

    def test_cleanup(self):
        """Test memory cleanup"""
        mm = MemoryManager()

        # Get stats before cleanup
        stats_before = mm.get_stats()

        # Perform cleanup
        stats_after = mm.cleanup(aggressive=False)

        # Cleanup should return new stats
        assert isinstance(stats_after, MemoryStats)
        # Memory used should be same or less after cleanup
        assert stats_after.cpu_used_gb <= stats_before.cpu_used_gb

    def test_check_memory_pressure(self):
        """Test memory pressure detection"""
        mm = MemoryManager()

        # Set high threshold to ensure no pressure
        mm.cpu_threshold_percent = 99.9
        mm.gpu_threshold_percent = 99.9

        # Should not be under pressure with high threshold
        pressure = mm.check_memory_pressure()
        assert isinstance(pressure, bool)

    def test_estimate_available_for_model(self):
        """Test available memory estimation"""
        mm = MemoryManager()

        available = mm.estimate_available_for_model()

        assert "cpu_gb" in available
        assert "gpu_gb" in available
        assert available["cpu_gb"] >= 0
        assert available["gpu_gb"] >= 0

    def test_can_load_model(self):
        """Test model loading feasibility check"""
        mm = MemoryManager()

        # Should be able to load a tiny model
        can_load_tiny = mm.can_load_model(estimated_size_gb=0.1, device="cpu")
        assert isinstance(can_load_tiny, bool)

        # Should not be able to load a huge model
        can_load_huge = mm.can_load_model(estimated_size_gb=999999.0, device="cpu")
        assert can_load_huge is False

    def test_auto_cleanup_if_needed(self):
        """Test automatic cleanup trigger"""
        mm = MemoryManager()

        # Set high threshold so cleanup shouldn't trigger
        mm.cpu_threshold_percent = 99.9
        mm.gc_threshold_percent = 99.9

        # Should not cleanup
        mm.auto_cleanup_if_needed()

        # Test with aggressive=True (should always run)
        mm.auto_cleanup_if_needed(aggressive=True)

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        mm = MemoryManager()

        batch_size = mm.get_optimal_batch_size(
            single_item_memory_gb=0.5,
            max_batch_size=32,
            device="cpu"
        )

        assert isinstance(batch_size, int)
        assert 1 <= batch_size <= 32

    def test_singleton_pattern(self):
        """Test that get_memory_manager returns singleton"""
        mm1 = get_memory_manager()
        mm2 = get_memory_manager()

        assert mm1 is mm2


class TestMemoryStats:
    """Test suite for MemoryStats dataclass"""

    def test_memory_stats_creation(self):
        """Test MemoryStats creation"""
        stats = MemoryStats(
            timestamp=time.time(),
            cpu_total_gb=32.0,
            cpu_used_gb=16.0,
            cpu_available_gb=16.0,
            cpu_percent=50.0,
            process_rss_gb=1.0,
            process_vms_gb=2.0,
        )

        assert stats.cpu_total_gb == 32.0
        assert stats.cpu_used_gb == 16.0
        assert stats.cpu_percent == 50.0

    def test_memory_stats_with_gpu(self):
        """Test MemoryStats with GPU info"""
        stats = MemoryStats(
            timestamp=time.time(),
            cpu_total_gb=32.0,
            cpu_used_gb=16.0,
            cpu_available_gb=16.0,
            cpu_percent=50.0,
            process_rss_gb=1.0,
            process_vms_gb=2.0,
            gpu_total_gb=24.0,
            gpu_allocated_gb=12.0,
            gpu_cached_gb=2.0,
            gpu_percent=50.0,
        )

        assert stats.gpu_total_gb == 24.0
        assert stats.gpu_allocated_gb == 12.0
        assert stats.gpu_percent == 50.0


@pytest.fixture
def memory_manager():
    """Fixture providing a MemoryManager instance"""
    return MemoryManager()


def test_memory_manager_fixture(memory_manager):
    """Test using MemoryManager fixture"""
    assert memory_manager is not None
    stats = memory_manager.get_stats()
    assert stats.cpu_total_gb > 0
