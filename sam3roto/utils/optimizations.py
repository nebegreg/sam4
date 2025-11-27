"""
Performance Optimizations
Batch processing, prefetching, async operations, etc.
"""

from __future__ import annotations
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from typing import List, Iterator, Callable, Any, Optional, TypeVar
from contextlib import contextmanager
import time
import torch
import numpy as np
from PIL import Image

from .memory_manager import get_memory_manager

T = TypeVar('T')


class BatchProcessor:
    """
    Efficient batch processing for images/frames

    Features:
    - Automatic batch size optimization
    - GPU/CPU aware batching
    - Progress tracking
    """

    def __init__(
        self,
        device: str = "cuda",
        auto_batch_size: bool = True,
        default_batch_size: int = 8,
        max_batch_size: int = 32,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            auto_batch_size: Automatically determine optimal batch size
            default_batch_size: Default batch size if auto disabled
            max_batch_size: Maximum allowed batch size
        """
        self.device = device
        self.auto_batch_size = auto_batch_size
        self.default_batch_size = default_batch_size
        self.max_batch_size = max_batch_size

        self.memory_manager = get_memory_manager()

    def get_optimal_batch_size(self, single_item_memory_gb: float = 0.5) -> int:
        """
        Calculate optimal batch size

        Args:
            single_item_memory_gb: Estimated memory per item

        Returns:
            Optimal batch size
        """
        if not self.auto_batch_size:
            return self.default_batch_size

        return self.memory_manager.get_optimal_batch_size(
            single_item_memory_gb=single_item_memory_gb,
            max_batch_size=self.max_batch_size,
            device=self.device,
        )

    def process_in_batches(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], List[Any]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Any]:
        """
        Process items in batches

        Args:
            items: List of items to process
            process_fn: Function that processes a batch
            batch_size: Batch size (auto-computed if None)
            show_progress: Show progress prints

        Returns:
            List of results
        """
        if batch_size is None:
            batch_size = self.get_optimal_batch_size()

        if show_progress:
            print(f"[BatchProcessor] Processing {len(items)} items with batch_size={batch_size}")

        results = []
        num_batches = (len(items) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(items))

            batch = items[start_idx:end_idx]

            if show_progress:
                print(f"[BatchProcessor] Batch {batch_idx+1}/{num_batches} ({len(batch)} items)")

            # Process batch
            batch_results = process_fn(batch)
            results.extend(batch_results)

            # Auto cleanup if memory pressure
            self.memory_manager.auto_cleanup_if_needed()

        if show_progress:
            print(f"[BatchProcessor] Completed! Processed {len(results)} items")

        return results


class Prefetcher:
    """
    Prefetch data in background thread

    Useful for loading/preprocessing frames while model is processing
    """

    def __init__(self, max_queue_size: int = 10):
        """
        Args:
            max_queue_size: Maximum items in prefetch queue
        """
        self.max_queue_size = max_queue_size
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, data_iterator: Iterator[T], transform_fn: Optional[Callable[[T], Any]] = None):
        """
        Start prefetching

        Args:
            data_iterator: Iterator yielding data
            transform_fn: Optional transform to apply before queuing
        """
        def prefetch_worker():
            for item in data_iterator:
                if self._stop_event.is_set():
                    break

                # Apply transform if provided
                if transform_fn is not None:
                    item = transform_fn(item)

                # Put in queue (blocks if full)
                self._queue.put(item)

            # Sentinel to indicate end
            self._queue.put(None)

        self._thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._thread.start()

    def __iter__(self):
        """Iterate over prefetched items"""
        while True:
            item = self._queue.get()
            if item is None:  # Sentinel
                break
            yield item

    def stop(self):
        """Stop prefetching"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class AsyncProcessor:
    """
    Process items asynchronously in thread pool

    Useful for I/O bound operations like loading/saving
    """

    def __init__(self, max_workers: int = 4):
        """
        Args:
            max_workers: Maximum concurrent workers
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: List[Future] = []

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit task for async execution"""
        future = self.executor.submit(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def wait_all(self, timeout: Optional[float] = None) -> List[Any]:
        """
        Wait for all submitted tasks to complete

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            List of results
        """
        results = []
        for future in self.futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                print(f"[AsyncProcessor] Task failed: {e}")
                results.append(None)

        self.futures.clear()
        return results

    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


@contextmanager
def torch_inference_mode(device: str = "cuda"):
    """
    Context manager for optimized inference

    Features:
    - torch.inference_mode()
    - Automatic GPU cache cleanup
    - Memory monitoring
    """
    mem_manager = get_memory_manager()

    # Print memory before
    print("[InferenceMode] Starting inference...")
    stats_before = mem_manager.get_stats()

    try:
        with torch.inference_mode():
            yield
    finally:
        # Cleanup after
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        stats_after = mem_manager.get_stats()

        # Print memory after
        if stats_before.gpu_allocated_gb is not None:
            gpu_used = stats_after.gpu_allocated_gb - stats_before.gpu_allocated_gb
            print(f"[InferenceMode] GPU memory change: {gpu_used:+.2f} GB")


@contextmanager
def timed_operation(name: str):
    """
    Context manager to time operations

    Example:
        with timed_operation("Model Loading"):
            model.load()
    """
    print(f"[Timer] {name} - Starting...")
    start = time.time()

    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[Timer] {name} - Completed in {elapsed:.2f}s")


def optimize_tensor_for_inference(tensor: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Optimize tensor for inference

    Args:
        tensor: Input tensor
        device: Target device

    Returns:
        Optimized tensor
    """
    # Move to device
    tensor = tensor.to(device)

    # Use appropriate dtype
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            tensor = tensor.to(dtype=torch.bfloat16)
        else:
            tensor = tensor.to(dtype=torch.float16)
    else:
        tensor = tensor.to(dtype=torch.float32)

    # Contiguous memory layout
    tensor = tensor.contiguous()

    return tensor


def batch_images_to_tensor(
    images: List[Image.Image],
    target_size: Optional[tuple[int, int]] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Convert batch of PIL Images to optimized torch tensor

    Args:
        images: List of PIL Images
        target_size: Optional (height, width) to resize
        device: Target device

    Returns:
        Tensor of shape (B, C, H, W)
    """
    # Convert to numpy arrays
    arrays = []
    for img in images:
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

        # RGB
        arr = np.array(img.convert("RGB"))
        arrays.append(arr)

    # Stack and convert to tensor
    # (B, H, W, C) -> (B, C, H, W)
    batch = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)

    # Normalize to [0, 1]
    tensor = tensor.float() / 255.0

    # Optimize
    tensor = optimize_tensor_for_inference(tensor, device=device)

    return tensor


class ProgressTracker:
    """Simple progress tracker"""

    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        percent = (self.current / self.total) * 100 if self.total > 0 else 0

        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            print(
                f"[{self.name}] {self.current}/{self.total} ({percent:.1f}%) "
                f"- {rate:.2f} it/s, ETA: {eta:.1f}s"
            )
        else:
            print(f"[{self.name}] {self.current}/{self.total} ({percent:.1f}%)")

    def finish(self):
        """Mark as complete"""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        print(f"[{self.name}] ✅ Completed {self.total} items in {elapsed:.2f}s ({rate:.2f} it/s)")
