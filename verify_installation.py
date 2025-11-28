#!/usr/bin/env python3
"""
SAM3 Roto - Installation Verification Script
Tests all components and optimizations to ensure everything is working correctly
"""

import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


def test_imports():
    """Test 1: Check all required imports"""
    print_header("Test 1: Import Verification")

    required_modules = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("PySide6", "PySide6"),
        ("tqdm", "tqdm"),
        ("imageio", "imageio"),
        ("psutil", "psutil"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
    ]

    optional_modules = [
        ("decord", "Decord (video decoding)"),
        ("pycocotools", "pycocotools (COCO API)"),
        ("transformers", "Transformers (HuggingFace)"),
    ]

    success_count = 0
    fail_count = 0

    # Test required modules
    print("Required Dependencies:")
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print_success(f"{display_name}")
            success_count += 1
        except ImportError as e:
            print_error(f"{display_name} - {e}")
            fail_count += 1

    # Test optional modules
    print("\nOptional Dependencies:")
    for module_name, display_name in optional_modules:
        try:
            __import__(module_name)
            print_success(f"{display_name}")
        except ImportError:
            print_warning(f"{display_name} - Not installed (optional)")

    return fail_count == 0


def test_sam3roto_imports():
    """Test 2: Check SAM3 Roto modules"""
    print_header("Test 2: SAM3 Roto Module Verification")

    success_count = 0
    fail_count = 0

    # Test utils imports
    print("Utils Module:")
    try:
        from sam3roto.utils import (
            get_memory_manager,
            get_feature_cache,
            BatchProcessor,
            torch_inference_mode,
            timed_operation,
        )
        print_success("Memory Manager")
        print_success("Feature Cache")
        print_success("Batch Processor")
        print_success("Optimization Utils")
        success_count += 4
    except ImportError as e:
        print_error(f"Utils import failed: {e}")
        fail_count += 1

    # Test backend imports
    print("\nBackend Module:")
    try:
        from sam3roto.backend.sam3_backend import SAM3Backend
        print_success("SAM3Backend")
        success_count += 1
    except ImportError as e:
        print_error(f"Backend import failed: {e}")
        fail_count += 1

    return fail_count == 0


def test_memory_manager():
    """Test 3: Memory Manager functionality"""
    print_header("Test 3: Memory Manager")

    try:
        from sam3roto.utils import get_memory_manager

        mm = get_memory_manager()

        # Test get_stats
        print("Testing get_stats()...")
        stats = mm.get_stats()
        print_success(f"CPU: {stats.cpu_used_gb:.2f}/{stats.cpu_total_gb:.2f} GB ({stats.cpu_percent:.1f}%)")

        if stats.gpu_total_gb is not None:
            print_success(f"GPU: {stats.gpu_allocated_gb:.2f}/{stats.gpu_total_gb:.2f} GB")
        else:
            print_warning("GPU not available (CPU mode)")

        # Test memory pressure check
        print("\nTesting check_memory_pressure()...")
        under_pressure = mm.check_memory_pressure()
        if under_pressure:
            print_warning("System is under memory pressure")
        else:
            print_success("Memory pressure is normal")

        # Test cleanup
        print("\nTesting cleanup()...")
        mm.cleanup(aggressive=False)
        print_success("Cleanup completed successfully")

        # Test available estimation
        print("\nTesting estimate_available_for_model()...")
        available = mm.estimate_available_for_model()
        print_success(f"Available: CPU={available['cpu_gb']:.2f} GB, GPU={available['gpu_gb']:.2f} GB")

        return True

    except Exception as e:
        print_error(f"Memory Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_cache():
    """Test 4: Feature Cache functionality"""
    print_header("Test 4: Feature Cache")

    try:
        from sam3roto.utils import get_feature_cache, cached
        import time

        cache = get_feature_cache()

        # Test basic set/get
        print("Testing set/get...")
        cache.set("test_key", {"data": "test_value"})
        value = cache.get("test_key")
        if value and value.get("data") == "test_value":
            print_success("Set/Get working correctly")
        else:
            print_error("Set/Get failed")
            return False

        # Test get_or_compute
        print("\nTesting get_or_compute...")
        def compute_fn():
            time.sleep(0.1)
            return {"computed": True}

        start = time.time()
        result1 = cache.get_or_compute("compute_test", compute_fn)
        time1 = time.time() - start

        start = time.time()
        result2 = cache.get_or_compute("compute_test", compute_fn)
        time2 = time.time() - start

        if time2 < time1 * 0.5:  # Second call should be much faster
            print_success(f"Cache speedup: {time1/time2:.1f}x")
        else:
            print_warning("Cache may not be working optimally")

        # Test decorator
        print("\nTesting @cached decorator...")
        @cached()
        def test_func(x):
            time.sleep(0.1)
            return x * 2

        start = time.time()
        result1 = test_func(5)
        time1 = time.time() - start

        start = time.time()
        result2 = test_func(5)
        time2 = time.time() - start

        if result1 == result2 == 10 and time2 < time1 * 0.5:
            print_success("Decorator working correctly")
        else:
            print_error("Decorator failed")
            return False

        # Test statistics
        print("\nTesting statistics...")
        stats = cache.get_stats()
        print_success(f"Hit rate: {stats['hit_rate']*100:.1f}%")

        return True

    except Exception as e:
        print_error(f"Feature Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processor():
    """Test 5: Batch Processor functionality"""
    print_header("Test 5: Batch Processor")

    try:
        from sam3roto.utils import BatchProcessor
        import time

        processor = BatchProcessor(
            device="cuda",
            auto_batch_size=True,
            max_batch_size=32,
        )

        # Test optimal batch size
        print("Testing get_optimal_batch_size()...")
        batch_size = processor.get_optimal_batch_size(single_item_memory_gb=0.5)
        print_success(f"Optimal batch size: {batch_size}")

        # Test batch processing
        print("\nTesting process_in_batches()...")
        items = list(range(50))

        def process_batch(batch):
            time.sleep(0.01)
            return [x * 2 for x in batch]

        results = processor.process_in_batches(
            items=items,
            process_fn=process_batch,
            show_progress=False,
        )

        if len(results) == len(items) and results[0] == 0 and results[-1] == 98:
            print_success(f"Batch processing completed: {len(results)} items")
        else:
            print_error("Batch processing failed")
            return False

        return True

    except Exception as e:
        print_error(f"Batch Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sam3_backend():
    """Test 6: SAM3 Backend (without model loading)"""
    print_header("Test 6: SAM3 Backend Initialization")

    try:
        from sam3roto.backend.sam3_backend import SAM3Backend

        # Test initialization
        print("Testing SAM3Backend initialization...")
        backend = SAM3Backend(enable_optimizations=True)
        print_success("Backend initialized with optimizations enabled")

        print_info(f"Device: {backend.device}")
        print_info(f"Dtype: {backend.dtype}")
        print_info(f"Optimizations: {backend.enable_optimizations}")

        # Note: We don't test model loading here as it requires downloading large models
        print_warning("Model loading test skipped (requires model download)")
        print_info("To test model loading, run: python test_sam3_loading.py")

        return True

    except Exception as e:
        print_error(f"SAM3 Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_cuda():
    """Test 7: PyTorch CUDA availability"""
    print_header("Test 7: PyTorch & CUDA")

    try:
        import torch

        print("PyTorch Configuration:")
        print_info(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print_success("CUDA is available")
            print_info(f"CUDA version: {torch.version.cuda}")
            print_info(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print_info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

            # Test bfloat16 support
            if torch.cuda.is_bf16_supported():
                print_success("BF16 supported (optimal performance)")
            else:
                print_warning("BF16 not supported (will use FP16)")

        else:
            print_warning("CUDA not available (CPU mode)")
            print_info("GPU acceleration disabled, performance will be slower")

        return True

    except Exception as e:
        print_error(f"PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test 8: Verify file structure"""
    print_header("Test 8: File Structure")

    required_files = [
        "sam3roto/__init__.py",
        "sam3roto/backend/__init__.py",
        "sam3roto/backend/sam3_backend.py",
        "sam3roto/utils/__init__.py",
        "sam3roto/utils/memory_manager.py",
        "sam3roto/utils/feature_cache.py",
        "sam3roto/utils/optimizations.py",
        "requirements.txt",
        "OPTIMIZATIONS_GUIDE.md",
        "AUDIT_REPORT.md",
        "ROADMAP.md",
        "FINAL_RECAP.md",
    ]

    optional_files = [
        "examples/memory_optimization_example.py",
        "examples/caching_example.py",
        "examples/batch_processing_example.py",
        "examples/README.md",
    ]

    success_count = 0
    fail_count = 0

    print("Required Files:")
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(file_path)
            success_count += 1
        else:
            print_error(f"{file_path} - Not found")
            fail_count += 1

    print("\nOptional Files:")
    for file_path in optional_files:
        if Path(file_path).exists():
            print_success(file_path)
        else:
            print_warning(f"{file_path} - Not found")

    return fail_count == 0


def main():
    """Run all verification tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}🚀 SAM3 Roto Ultimate - Installation Verification{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")

    tests = [
        ("Import Verification", test_imports),
        ("SAM3 Roto Modules", test_sam3roto_imports),
        ("Memory Manager", test_memory_manager),
        ("Feature Cache", test_feature_cache),
        ("Batch Processor", test_batch_processor),
        ("SAM3 Backend", test_sam3_backend),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("File Structure", test_file_structure),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print_header("Verification Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Installation is complete and working.{Colors.RESET}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.RESET}")
        print("  1. Try the example scripts in examples/")
        print("  2. Read OPTIMIZATIONS_GUIDE.md for usage instructions")
        print("  3. Check ROADMAP.md for future enhancements")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed. Please check the errors above.{Colors.RESET}")
        print(f"\n{Colors.BOLD}Troubleshooting:{Colors.RESET}")
        print("  1. Run: bash install_venv_complete.sh")
        print("  2. Activate venv: source venv_sam3roto/bin/activate")
        print("  3. Check requirements.txt is installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
