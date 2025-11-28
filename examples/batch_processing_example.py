#!/usr/bin/env python3
"""
Batch Processing Example
Demonstrates efficient batch processing with auto-sizing for 2x speedup
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3roto.utils import (
    BatchProcessor,
    ProgressTracker,
    torch_inference_mode,
    timed_operation,
)
from PIL import Image
import numpy as np


def example_basic_batch_processing():
    """Example 1: Basic batch processing"""
    print("=" * 80)
    print("Example 1: Basic Batch Processing")
    print("=" * 80)

    processor = BatchProcessor(
        device="cuda",
        auto_batch_size=True,
        default_batch_size=8,
        max_batch_size=32,
    )

    # Create dummy items
    items = list(range(100))

    def process_batch(batch):
        """Process a batch of items"""
        time.sleep(0.1)  # Simulate processing
        return [item * 2 for item in batch]

    print(f"\n🔄 Processing {len(items)} items...")

    # Process in batches
    results = processor.process_in_batches(
        items=items,
        process_fn=process_batch,
        batch_size=None,  # Auto-computed
        show_progress=True,
    )

    print(f"\n✅ Processed {len(results)} results")
    print(f"   Sample results: {results[:5]}")


def example_auto_batch_sizing():
    """Example 2: Automatic batch size calculation"""
    print("\n" + "=" * 80)
    print("Example 2: Automatic Batch Sizing")
    print("=" * 80)

    processor = BatchProcessor(
        device="cuda",
        auto_batch_size=True,
        max_batch_size=32,
    )

    # Estimate memory per item (e.g., 1080p image ~0.5GB after encoding)
    single_item_memory_gb = 0.5

    print(f"\n📊 Calculating optimal batch size...")
    print(f"   Estimated memory per item: {single_item_memory_gb} GB")

    optimal_size = processor.get_optimal_batch_size(
        single_item_memory_gb=single_item_memory_gb
    )

    print(f"\n✅ Optimal batch size: {optimal_size}")
    print(f"   This will use available GPU memory efficiently")


def example_image_batch_processing():
    """Example 3: Processing image batches"""
    print("\n" + "=" * 80)
    print("Example 3: Image Batch Processing")
    print("=" * 80)

    processor = BatchProcessor(
        device="cuda",
        auto_batch_size=True,
        max_batch_size=16,
    )

    # Simulate image list
    num_images = 50
    print(f"\n🖼️  Simulating {num_images} images...")

    # Create dummy image data
    images = []
    for i in range(num_images):
        # Create random image (512x512 RGB)
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)

    def process_image_batch(batch):
        """Process batch of images"""
        results = []
        for img in batch:
            # Simulate processing (e.g., segmentation)
            time.sleep(0.05)  # Simulate model inference

            # Return dummy mask
            mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
            results.append(mask)

        return results

    print(f"\n🔄 Processing images in batches...")

    with timed_operation("Image Batch Processing"):
        results = processor.process_in_batches(
            items=images,
            process_fn=process_image_batch,
            show_progress=True,
        )

    print(f"\n✅ Processed {len(results)} masks")
    print(f"   Mask shape: {results[0].shape}")


def example_video_frame_processing():
    """Example 4: Video frame processing simulation"""
    print("\n" + "=" * 80)
    print("Example 4: Video Frame Processing")
    print("=" * 80)

    processor = BatchProcessor(
        device="cuda",
        auto_batch_size=True,
        max_batch_size=16,
    )

    # Simulate video frames
    num_frames = 120  # 4 seconds at 30fps
    fps = 30

    print(f"\n🎬 Simulating video with {num_frames} frames ({num_frames/fps:.1f}s @ {fps}fps)")

    # Create dummy frames
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frames.append(frame)

    def process_frame_batch(batch):
        """Process batch of frames"""
        results = []

        # Use inference mode context
        with torch_inference_mode():
            for frame in batch:
                # Simulate segmentation
                time.sleep(0.02)  # 20ms per frame

                # Return dummy mask
                mask = np.random.randint(0, 2, (1080, 1920), dtype=np.uint8) * 255
                results.append(mask)

        return results

    print(f"\n🔄 Processing video frames...")

    start = time.time()
    results = processor.process_in_batches(
        items=frames,
        process_fn=process_frame_batch,
        show_progress=True,
    )
    elapsed = time.time() - start

    processing_fps = num_frames / elapsed if elapsed > 0 else 0

    print(f"\n✅ Processed {len(results)} frames")
    print(f"   Processing time: {elapsed:.2f}s")
    print(f"   Processing FPS: {processing_fps:.1f}")
    print(f"   Real-time factor: {processing_fps/fps:.2f}x")


def example_progress_tracking():
    """Example 5: Manual progress tracking"""
    print("\n" + "=" * 80)
    print("Example 5: Progress Tracking")
    print("=" * 80)

    total_items = 100
    tracker = ProgressTracker(total=total_items, name="Processing")

    print(f"\n🔄 Processing {total_items} items with progress tracking...")

    for i in range(total_items):
        # Simulate work
        time.sleep(0.02)

        # Update progress every 10 items
        if (i + 1) % 10 == 0:
            tracker.update(10)

    tracker.finish()


def example_comparison_batch_vs_sequential():
    """Example 6: Compare batch vs sequential processing"""
    print("\n" + "=" * 80)
    print("Example 6: Batch vs Sequential Comparison")
    print("=" * 80)

    num_items = 48
    items = list(range(num_items))

    def process_item(item):
        """Process single item"""
        time.sleep(0.05)
        return item * 2

    def process_batch(batch):
        """Process batch"""
        # Batch processing has overhead reduction
        time.sleep(0.05 * len(batch) * 0.7)  # 30% faster due to batching
        return [item * 2 for item in batch]

    # Sequential processing
    print("\n🐌 Sequential Processing:")
    start = time.time()
    results_seq = [process_item(item) for item in items]
    elapsed_seq = time.time() - start
    print(f"   Time: {elapsed_seq:.2f}s")

    # Batch processing
    print("\n🚀 Batch Processing:")
    processor = BatchProcessor(device="cpu", default_batch_size=8)
    start = time.time()
    results_batch = processor.process_in_batches(
        items=items,
        process_fn=process_batch,
        batch_size=8,
        show_progress=False,
    )
    elapsed_batch = time.time() - start
    print(f"   Time: {elapsed_batch:.2f}s")

    # Comparison
    speedup = elapsed_seq / elapsed_batch if elapsed_batch > 0 else 0
    print(f"\n📊 Comparison:")
    print(f"   Sequential: {elapsed_seq:.2f}s")
    print(f"   Batch: {elapsed_batch:.2f}s")
    print(f"   🚀 Speedup: {speedup:.2f}x")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("🚀 SAM3 Roto - Batch Processing Examples")
    print("=" * 80)

    try:
        # Example 1: Basic batch processing
        example_basic_batch_processing()

        # Example 2: Auto batch sizing
        example_auto_batch_sizing()

        # Example 3: Image processing
        example_image_batch_processing()

        # Example 4: Video processing (can be slow)
        # Uncomment to test:
        # example_video_frame_processing()

        # Example 5: Progress tracking
        example_progress_tracking()

        # Example 6: Batch vs sequential
        example_comparison_batch_vs_sequential()

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
