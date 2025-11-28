"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Provide temporary directory for test outputs"""
    return tmp_path_factory.mktemp("test_outputs")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Clear memory manager singleton
    import sam3roto.utils.memory_manager as mm_module
    if hasattr(mm_module, '_MEMORY_MANAGER'):
        delattr(mm_module, '_MEMORY_MANAGER')

    # Clear feature cache singleton
    import sam3roto.utils.feature_cache as fc_module
    if hasattr(fc_module, '_FEATURE_CACHE'):
        delattr(fc_module, '_FEATURE_CACHE')

    yield

    # Cleanup after test
    pass


@pytest.fixture
def mock_image():
    """Provide a mock PIL Image for testing"""
    try:
        from PIL import Image
        import numpy as np

        # Create 512x512 RGB image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    except ImportError:
        pytest.skip("PIL not available")


@pytest.fixture
def mock_video_frames():
    """Provide mock video frames for testing"""
    try:
        import numpy as np

        # Create 10 frames of 256x256 RGB
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            frames.append(frame)

        return frames
    except ImportError:
        pytest.skip("NumPy not available")


@pytest.fixture
def mock_masks():
    """Provide mock segmentation masks for testing"""
    try:
        import numpy as np

        # Create 3 masks of 512x512
        masks = {}
        for i in range(3):
            mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
            masks[i] = mask

        return masks
    except ImportError:
        pytest.skip("NumPy not available")


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "model: mark test as requiring model loading"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Auto-mark tests based on their names
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.nodeid.lower() or "video" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark model tests
        if "model" in item.nodeid.lower() or "sam" in item.nodeid.lower():
            item.add_marker(pytest.mark.model)
