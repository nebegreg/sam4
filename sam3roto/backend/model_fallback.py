"""
SAM2 Fallback Module
Provides automatic fallback from SAM3 to SAM2 for compatibility
"""

from __future__ import annotations
from typing import Optional, Literal
import warnings

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ModelFallbackManager:
    """Manages automatic fallback between SAM3 and SAM2 models"""

    def __init__(self):
        self.current_backend: Optional[Literal["sam3", "sam2"]] = None
        self.sam3_available = False
        self.sam2_available = False

        # Check availability
        self._check_sam3_availability()
        self._check_sam2_availability()

    def _check_sam3_availability(self) -> bool:
        """Check if SAM3 is available"""
        try:
            # Try transformers API
            from transformers import Sam3Model, Sam3Processor
            self.sam3_available = True
            return True
        except ImportError:
            pass

        try:
            # Try GitHub repo API
            from sam3.model_builder import build_sam3_image_model
            self.sam3_available = True
            return True
        except ImportError:
            pass

        self.sam3_available = False
        return False

    def _check_sam2_availability(self) -> bool:
        """Check if SAM2 is available"""
        try:
            from sam2.build_sam import build_sam2
            self.sam2_available = True
            return True
        except ImportError:
            self.sam2_available = False
            return False

    def get_recommended_backend(self) -> Literal["sam3", "sam2", None]:
        """Get recommended backend based on availability

        Priority:
        1. SAM3 (newest, best performance)
        2. SAM2 (stable fallback)
        3. None (no models available)
        """
        if self.sam3_available:
            return "sam3"
        elif self.sam2_available:
            warnings.warn(
                "SAM3 not available, falling back to SAM2. "
                "For best performance, install SAM3: "
                "pip install git+https://github.com/facebookresearch/sam3.git",
                UserWarning
            )
            return "sam2"
        else:
            return None

    def load_model(
        self,
        model_type: Optional[Literal["sam3", "sam2"]] = None,
        model_id: str = "facebook/sam3-hiera-large",
        device: str = "cuda",
    ):
        """Load model with automatic fallback

        Args:
            model_type: Preferred model type ("sam3", "sam2", or None for auto)
            model_id: Model identifier or checkpoint path
            device: Device to load model on ("cuda" or "cpu")

        Returns:
            Tuple of (model, processor, backend_name)
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required but not installed")

        # Auto-select if not specified
        if model_type is None:
            model_type = self.get_recommended_backend()
            if model_type is None:
                raise RuntimeError(
                    "No SAM models available. Please install SAM3 or SAM2:\n"
                    "  SAM3: pip install git+https://github.com/facebookresearch/sam3.git\n"
                    "  SAM2: pip install git+https://github.com/facebookresearch/sam2.git"
                )

        # Load requested model
        if model_type == "sam3":
            return self._load_sam3(model_id, device)
        elif model_type == "sam2":
            return self._load_sam2(model_id, device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _load_sam3(self, model_id: str, device: str):
        """Load SAM3 model"""
        if not self.sam3_available:
            raise RuntimeError("SAM3 is not available")

        # Try transformers first
        try:
            from transformers import Sam3Model, Sam3Processor

            print(f"[SAM3] Loading via transformers: {model_id}")
            model = Sam3Model.from_pretrained(model_id).to(device)
            processor = Sam3Processor.from_pretrained(model_id)

            self.current_backend = "sam3"
            return model, processor, "sam3-transformers"

        except ImportError:
            pass

        # Try GitHub repo
        try:
            from sam3.model_builder import build_sam3_image_model

            print(f"[SAM3] Loading via GitHub repo: {model_id}")

            # Determine if loading from HuggingFace or local checkpoint
            load_from_hf = not model_id.endswith((".pth", ".pt"))

            model = build_sam3_image_model(
                checkpoint_path=model_id if not load_from_hf else None,
                load_from_HF=load_from_hf,
                device=device,
                eval_mode=True
            )

            # Create processor
            from sam3.processor import Sam3Processor as Sam3ProcessorGH
            processor = Sam3ProcessorGH(model, device=device)

            self.current_backend = "sam3"
            return model, processor, "sam3-github"

        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3: {e}")

    def _load_sam2(self, model_id: str, device: str):
        """Load SAM2 model as fallback"""
        if not self.sam2_available:
            raise RuntimeError("SAM2 is not available")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            print(f"[SAM2] Loading fallback model: {model_id}")

            # Map SAM3 model IDs to SAM2 equivalents
            sam2_mapping = {
                "facebook/sam3-hiera-large": "sam2_hiera_large.pt",
                "facebook/sam3-hiera-base": "sam2_hiera_base_plus.pt",
                "facebook/sam3-hiera-small": "sam2_hiera_small.pt",
            }

            # Use mapping if available
            checkpoint = sam2_mapping.get(model_id, model_id)

            # Build SAM2
            sam2_model = build_sam2(
                config_file=checkpoint.replace(".pt", ".yaml"),
                ckpt_path=checkpoint,
                device=device,
            )

            # Create predictor
            predictor = SAM2ImagePredictor(sam2_model)

            self.current_backend = "sam2"
            return sam2_model, predictor, "sam2"

        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2: {e}")


def get_fallback_manager() -> ModelFallbackManager:
    """Get singleton fallback manager"""
    global _FALLBACK_MANAGER

    if "_FALLBACK_MANAGER" not in globals():
        _FALLBACK_MANAGER = ModelFallbackManager()

    return _FALLBACK_MANAGER


# Convenience function
def load_best_available_model(
    model_id: str = "facebook/sam3-hiera-large",
    device: str = "cuda",
    prefer_sam3: bool = True,
):
    """Load the best available SAM model

    Args:
        model_id: Model identifier
        device: Device to load on
        prefer_sam3: If True, prefer SAM3 over SAM2

    Returns:
        Tuple of (model, processor, backend_name)

    Example:
        >>> model, processor, backend = load_best_available_model()
        >>> print(f"Loaded backend: {backend}")
        Loaded backend: sam3-transformers
    """
    manager = get_fallback_manager()

    if prefer_sam3:
        model_type = None  # Auto-select (will prefer SAM3)
    else:
        model_type = "sam2" if manager.sam2_available else None

    return manager.load_model(model_type, model_id, device)
