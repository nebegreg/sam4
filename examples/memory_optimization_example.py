#!/usr/bin/env python3
"""
Memory Management Example
Demonstrates how to use the MemoryManager for optimal GPU/CPU usage
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3roto.utils import (
    get_memory_manager,
    cleanup_memory,
    print_memory_stats,
)
from sam3roto.backend.sam3_backend import SAM3Backend
from PIL import Image
import torch


def example_basic_monitoring():
    """Example 1: Basic memory monitoring"""
    print("=" * 80)
    print("Example 1: Basic Memory Monitoring")
    print("=" * 80)

    mm = get_memory_manager()

    # Get current stats
    print("\n📊 Current Memory Status:")
    mm.print_summary()

    # Check if under pressure
    if mm.check_memory_pressure():
        print("\n⚠️  Warning: System is under memory pressure!")
    else:
        print("\n✅ Memory usage is healthy")


def example_model_loading_with_check():
    """Example 2: Check memory before loading model"""
    print("\n" + "=" * 80)
    print("Example 2: Memory Check Before Model Loading")
    print("=" * 80)

    mm = get_memory_manager()

    # Estimate if we can load SAM3-Large (approximately 4GB)
    estimated_size = 4.0  # GB

    available = mm.estimate_available_for_model()
    print(f"\n📊 Available Memory:")
    print(f"   CPU: {available['cpu_gb']:.2f} GB")
    print(f"   GPU: {available['gpu_gb']:.2f} GB")

    can_load = mm.can_load_model(estimated_size_gb=estimated_size, device="cuda")

    if can_load:
        print(f"\n✅ Sufficient memory to load {estimated_size}GB model")

        # Load model with monitoring
        backend = SAM3Backend(enable_optimizations=True)

        print("\n🔄 Loading SAM3 model...")
        try:
            backend.load("facebook/sam3-hiera-large")
            print("✅ Model loaded successfully!")

            # Print stats after loading
            print("\n📊 Memory After Loading:")
            mm.print_summary()

        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"\n❌ Insufficient memory to load {estimated_size}GB model")
        print("   Consider:")
        print("   - Closing other applications")
        print("   - Using a smaller model")
        print("   - Running cleanup:")
        mm.cleanup(aggressive=True)


def example_manual_cleanup():
    """Example 3: Manual memory cleanup"""
    print("\n" + "=" * 80)
    print("Example 3: Manual Memory Cleanup")
    print("=" * 80)

    mm = get_memory_manager()

    print("\n📊 Before Cleanup:")
    stats_before = mm.get_stats()
    print(f"   CPU: {stats_before.cpu_used_gb:.2f}/{stats_before.cpu_total_gb:.2f} GB ({stats_before.cpu_percent:.1f}%)")
    if stats_before.gpu_allocated_gb is not None:
        print(f"   GPU: {stats_before.gpu_allocated_gb:.2f}/{stats_before.gpu_total_gb:.2f} GB")

    # Perform cleanup
    print("\n🧹 Running cleanup...")
    mm.cleanup(aggressive=True)

    print("\n📊 After Cleanup:")
    stats_after = mm.get_stats()
    print(f"   CPU: {stats_after.cpu_used_gb:.2f}/{stats_after.cpu_total_gb:.2f} GB ({stats_after.cpu_percent:.1f}%)")
    if stats_after.gpu_allocated_gb is not None:
        print(f"   GPU: {stats_after.gpu_allocated_gb:.2f}/{stats_after.gpu_total_gb:.2f} GB")

        # Calculate freed memory
        gpu_freed = stats_before.gpu_allocated_gb - stats_after.gpu_allocated_gb
        if gpu_freed > 0:
            print(f"\n✅ Freed {gpu_freed:.2f} GB GPU memory")


def example_auto_cleanup():
    """Example 4: Automatic cleanup with threshold"""
    print("\n" + "=" * 80)
    print("Example 4: Automatic Cleanup")
    print("=" * 80)

    mm = get_memory_manager()

    # Configure auto cleanup
    mm.auto_gc = True
    mm.gc_threshold_percent = 80.0

    print(f"\n⚙️  Auto cleanup configured:")
    print(f"   Enabled: {mm.auto_gc}")
    print(f"   Threshold: {mm.gc_threshold_percent}%")

    # This would be called automatically during processing
    print("\n🔄 Checking if cleanup needed...")
    mm.auto_cleanup_if_needed()
    print("✅ Auto cleanup check complete")


def example_memory_history():
    """Example 5: View memory usage history"""
    print("\n" + "=" * 80)
    print("Example 5: Memory Usage History")
    print("=" * 80)

    mm = get_memory_manager()

    # Simulate some memory operations
    print("\n🔄 Simulating memory operations...")

    # Operation 1
    mm.get_stats()
    print("   Operation 1 complete")

    # Operation 2
    mm.get_stats()
    print("   Operation 2 complete")

    # Operation 3
    mm.get_stats()
    print("   Operation 3 complete")

    # View history
    print(f"\n📊 Memory History ({len(mm.history)} snapshots):")
    for i, stats in enumerate(mm.history[-5:], 1):  # Last 5
        print(f"   Snapshot {i}: CPU {stats.cpu_percent:.1f}%", end="")
        if stats.gpu_percent is not None:
            print(f", GPU {stats.gpu_percent:.1f}%")
        else:
            print()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("🚀 SAM3 Roto - Memory Management Examples")
    print("=" * 80)

    try:
        # Example 1: Basic monitoring
        example_basic_monitoring()

        # Example 2: Model loading check (optional - requires model download)
        # Uncomment to test with actual model:
        # example_model_loading_with_check()

        # Example 3: Manual cleanup
        example_manual_cleanup()

        # Example 4: Auto cleanup
        example_auto_cleanup()

        # Example 5: Memory history
        example_memory_history()

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
