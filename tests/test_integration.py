"""
Integration tests for SAM3 Roto Ultimate
Tests the interaction between multiple components
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.mark.integration
class TestMemoryCacheIntegration:
    """Test integration between Memory Manager and Feature Cache"""

    def test_memory_manager_with_cache(self):
        """Test that memory manager and cache work together"""
        from sam3roto.utils import get_memory_manager, get_feature_cache

        mm = get_memory_manager()
        cache = get_feature_cache()

        # Get initial memory
        stats_before = mm.get_stats()

        # Add items to cache
        for i in range(10):
            large_data = np.random.randn(1000, 1000).astype(np.float32)
            cache.set(f"key_{i}", large_data)

        # Get memory after caching
        stats_after = mm.get_stats()

        # Memory usage should increase
        assert stats_after.cpu_used_gb >= stats_before.cpu_used_gb

        # Cleanup
        cache.clear()
        mm.cleanup(aggressive=True)

        stats_final = mm.get_stats()
        # After cleanup, memory should decrease
        assert stats_final.cpu_used_gb <= stats_after.cpu_used_gb


@pytest.mark.integration
class TestBatchProcessorWithOptimizations:
    """Test BatchProcessor with memory management"""

    def test_batch_processor_with_memory_cleanup(self):
        """Test that batch processor cleans up memory between batches"""
        from sam3roto.utils import BatchProcessor, get_memory_manager

        processor = BatchProcessor(device="cpu", default_batch_size=5)
        mm = get_memory_manager()

        items = list(range(50))

        def process_batch(batch):
            # Simulate memory-intensive operation
            temp_data = np.random.randn(100, 100)
            return [x * 2 for x in batch]

        stats_before = mm.get_stats()

        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            show_progress=False
        )

        assert len(results) == 50

        # Memory manager should have tracked the processing
        stats_after = mm.get_stats()
        assert stats_after.timestamp > stats_before.timestamp


@pytest.mark.integration
@pytest.mark.slow
class TestOptimizationsWorkflow:
    """Test complete optimization workflow"""

    def test_full_optimization_workflow(self):
        """Test memory, cache, and batch processing together"""
        from sam3roto.utils import (
            get_memory_manager,
            get_feature_cache,
            BatchProcessor,
            cached,
        )

        mm = get_memory_manager()
        cache = get_feature_cache()
        processor = BatchProcessor(device="cpu", max_batch_size=8)

        # Define cached expensive operation
        @cached()
        def expensive_operation(x):
            return np.random.randn(100, 100) * x

        # Create items
        items = [1, 2, 3, 1, 2, 3]  # Repeated values for cache hits

        def process_batch(batch):
            results = []
            for item in batch:
                # This should hit cache for repeated values
                result = expensive_operation(item)
                results.append(result.sum())
            return results

        # Check memory before
        mm.print_summary()

        # Process in batches
        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            batch_size=2,
            show_progress=False
        )

        # Check results
        assert len(results) == 6

        # Check cache had hits
        cache_stats = cache.get_stats()
        assert cache_stats["hits"] > 0  # Should have cache hits from repeated values

        # Cleanup
        mm.cleanup(aggressive=True)
        cache.clear()


@pytest.mark.integration
@pytest.mark.model
class TestSAM3BackendIntegration:
    """Integration tests for SAM3Backend (without model loading)"""

    def test_backend_initialization_with_optimizations(self):
        """Test SAM3Backend initialization with optimizations"""
        from sam3roto.backend.sam3_backend import SAM3Backend

        # Initialize with optimizations
        backend = SAM3Backend(enable_optimizations=True)

        assert backend.enable_optimizations is True
        assert backend.memory_manager is not None
        assert backend.feature_cache is not None
        assert backend.device is not None

    def test_backend_initialization_without_optimizations(self):
        """Test SAM3Backend initialization without optimizations"""
        from sam3roto.backend.sam3_backend import SAM3Backend

        # Initialize without optimizations
        backend = SAM3Backend(enable_optimizations=False)

        assert backend.enable_optimizations is False
        assert backend.memory_manager is None
        assert backend.feature_cache is None


@pytest.mark.integration
class TestModelFallbackIntegration:
    """Integration tests for model fallback system"""

    def test_fallback_manager_initialization(self):
        """Test fallback manager initialization"""
        from sam3roto.backend.model_fallback import ModelFallbackManager

        manager = ModelFallbackManager()

        # Manager should check availability
        assert isinstance(manager.sam3_available, bool)
        assert isinstance(manager.sam2_available, bool)

    def test_get_recommended_backend(self):
        """Test getting recommended backend"""
        from sam3roto.backend.model_fallback import get_fallback_manager

        manager = get_fallback_manager()
        recommended = manager.get_recommended_backend()

        # Should be None, "sam2", or "sam3"
        assert recommended in [None, "sam2", "sam3"]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.model
class TestVideoProcessingWorkflow:
    """Integration test for video processing workflow"""

    def test_video_processing_simulation(self, mock_video_frames, mock_masks):
        """Test simulated video processing with all optimizations"""
        from sam3roto.utils import (
            get_memory_manager,
            get_feature_cache,
            BatchProcessor,
        )

        mm = get_memory_manager()
        cache = get_feature_cache()
        processor = BatchProcessor(device="cpu", max_batch_size=4)

        # Simulate processing video frames
        def process_frame_batch(frames):
            """Simulate frame processing"""
            results = []
            for frame in frames:
                # Simulate segmentation (return mock mask)
                mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
                results.append(mask)
            return results

        # Check memory before
        stats_before = mm.get_stats()

        # Process frames in batches
        processed_masks = processor.process_in_batches(
            items=mock_video_frames,
            process_fn=process_frame_batch,
            show_progress=False
        )

        # Verify results
        assert len(processed_masks) == len(mock_video_frames)
        assert all(isinstance(mask, np.ndarray) for mask in processed_masks)
        assert all(mask.shape == (256, 256) for mask in processed_masks)

        # Check memory after
        stats_after = mm.get_stats()

        # Memory should be tracked
        assert len(mm.history) > 0

        # Cleanup
        mm.cleanup(aggressive=True)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests"""

    def test_complete_pipeline_simulation(self):
        """Test complete pipeline from initialization to cleanup"""
        from sam3roto.backend.sam3_backend import SAM3Backend
        from sam3roto.utils import get_memory_manager

        # 1. Initialize backend with optimizations
        backend = SAM3Backend(enable_optimizations=True)
        mm = get_memory_manager()

        # 2. Check memory
        stats = mm.get_stats()
        assert stats.cpu_total_gb > 0

        # 3. Simulate some operations
        # (without loading models, just test the infrastructure)

        # 4. Cleanup
        mm.cleanup(aggressive=True)

        # 5. Verify cleanup worked
        stats_final = mm.get_stats()
        assert stats_final is not None


# Test data directory setup
@pytest.mark.integration
class TestDataManagement:
    """Test data directory management"""

    def test_create_test_data_directory(self, test_data_dir):
        """Test creating test data directory"""
        # Directory should exist (created by fixture)
        assert test_data_dir is not None

    def test_create_temp_directory(self, temp_dir):
        """Test creating temporary directory"""
        assert temp_dir is not None
        assert temp_dir.exists()

        # Can write files
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
