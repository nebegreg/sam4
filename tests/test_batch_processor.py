"""
Unit tests for Batch Processor
"""

import pytest
import time
from sam3roto.utils.optimizations import (
    BatchProcessor,
    ProgressTracker,
    torch_inference_mode,
    timed_operation,
)


class TestBatchProcessor:
    """Test suite for BatchProcessor"""

    def test_initialization(self):
        """Test BatchProcessor initialization"""
        processor = BatchProcessor(
            device="cpu",
            auto_batch_size=True,
            default_batch_size=8,
            max_batch_size=32
        )

        assert processor.device == "cpu"
        assert processor.auto_batch_size is True
        assert processor.default_batch_size == 8
        assert processor.max_batch_size == 32

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        processor = BatchProcessor(device="cpu", max_batch_size=32)

        batch_size = processor.get_optimal_batch_size(single_item_memory_gb=0.5)

        assert isinstance(batch_size, int)
        assert 1 <= batch_size <= 32

    def test_process_in_batches_basic(self):
        """Test basic batch processing"""
        processor = BatchProcessor(device="cpu", default_batch_size=5)

        items = list(range(20))

        def process_batch(batch):
            return [x * 2 for x in batch]

        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            batch_size=5,
            show_progress=False
        )

        assert len(results) == 20
        assert results[0] == 0
        assert results[10] == 20
        assert results[19] == 38

    def test_process_in_batches_auto_size(self):
        """Test batch processing with auto batch size"""
        processor = BatchProcessor(
            device="cpu",
            auto_batch_size=True,
            max_batch_size=16
        )

        items = list(range(50))

        def process_batch(batch):
            return [x * 2 for x in batch]

        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            show_progress=False
        )

        assert len(results) == 50
        assert all(results[i] == i * 2 for i in range(50))

    def test_process_in_batches_empty_list(self):
        """Test processing empty list"""
        processor = BatchProcessor(device="cpu")

        items = []

        def process_batch(batch):
            return [x * 2 for x in batch]

        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            show_progress=False
        )

        assert results == []

    def test_process_in_batches_single_item(self):
        """Test processing single item"""
        processor = BatchProcessor(device="cpu")

        items = [42]

        def process_batch(batch):
            return [x * 2 for x in batch]

        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            show_progress=False
        )

        assert results == [84]

    def test_process_in_batches_with_progress(self):
        """Test batch processing with progress tracking"""
        processor = BatchProcessor(device="cpu", default_batch_size=5)

        items = list(range(15))

        def process_batch(batch):
            return [x * 2 for x in batch]

        # Should not raise any errors with progress tracking
        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            batch_size=5,
            show_progress=True  # This will print progress
        )

        assert len(results) == 15


class TestProgressTracker:
    """Test suite for ProgressTracker"""

    def test_initialization(self):
        """Test ProgressTracker initialization"""
        tracker = ProgressTracker(total=100, name="Testing")

        assert tracker.total == 100
        assert tracker.name == "Testing"
        assert tracker.current == 0

    def test_update(self):
        """Test updating progress"""
        tracker = ProgressTracker(total=100, name="Testing")

        tracker.update(10)
        assert tracker.current == 10

        tracker.update(20)
        assert tracker.current == 30

    def test_finish(self):
        """Test finishing progress"""
        tracker = ProgressTracker(total=100, name="Testing")

        tracker.update(50)
        tracker.finish()

        # After finish, should be at 100%
        assert tracker.current == tracker.total

    def test_progress_percentage(self):
        """Test progress percentage calculation"""
        tracker = ProgressTracker(total=100, name="Testing")

        tracker.update(25)
        assert tracker.current == 25

        tracker.update(25)
        assert tracker.current == 50


class TestContextManagers:
    """Test suite for context manager utilities"""

    def test_torch_inference_mode(self):
        """Test torch_inference_mode context manager"""
        # Should work even without torch available
        with torch_inference_mode():
            # Some code here
            pass

    def test_timed_operation(self):
        """Test timed_operation context manager"""
        with timed_operation("Test Operation"):
            time.sleep(0.1)

        # Should print elapsed time (visual verification)

    def test_timed_operation_nesting(self):
        """Test nested timed operations"""
        with timed_operation("Outer"):
            time.sleep(0.05)
            with timed_operation("Inner"):
                time.sleep(0.05)


@pytest.fixture
def batch_processor():
    """Fixture providing a BatchProcessor instance"""
    return BatchProcessor(device="cpu", default_batch_size=4, max_batch_size=16)


def test_batch_processor_fixture(batch_processor):
    """Test using BatchProcessor fixture"""
    assert batch_processor is not None

    items = [1, 2, 3, 4]

    def process_fn(batch):
        return [x * 2 for x in batch]

    results = batch_processor.process_in_batches(
        items=items,
        process_fn=process_fn,
        show_progress=False
    )

    assert results == [2, 4, 6, 8]


@pytest.fixture
def progress_tracker():
    """Fixture providing a ProgressTracker instance"""
    return ProgressTracker(total=100, name="Test")


def test_progress_tracker_fixture(progress_tracker):
    """Test using ProgressTracker fixture"""
    assert progress_tracker is not None
    progress_tracker.update(50)
    assert progress_tracker.current == 50
