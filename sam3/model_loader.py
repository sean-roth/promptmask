"""
SAM 3 Model Loader
Handles loading and initialization of the Meta SAM 3 model from Hugging Face.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class SAM3ModelLoader:
    """Manages loading and caching of the SAM 3 model."""

    def __init__(self, model_id: str = "facebook/sam3", device: Optional[str] = None):
        """
        Initialize the model loader.

        Args:
            model_id: Hugging Face model identifier
            device: Device to load model on ('mps', 'cuda', 'cpu'). Auto-detects if None.
        """
        self.model_id = model_id
        self.device = device or self._auto_detect_device()
        self.model = None
        self.processor = None
        self._is_loaded = False

        logger.info(f"Initializing SAM3ModelLoader with device: {self.device}")

    def _auto_detect_device(self) -> str:
        """
        Auto-detect the best available device.

        Returns:
            Device string ('mps', 'cuda', or 'cpu')
        """
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("CUDA available - using NVIDIA GPU")
            return "cuda"
        else:
            logger.warning("No GPU acceleration available - falling back to CPU (will be slow)")
            return "cpu"

    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the SAM 3 model and processor into memory.

        Args:
            force_reload: If True, reload even if already loaded

        Returns:
            True if successful, False otherwise
        """
        if self._is_loaded and not force_reload:
            logger.info("Model already loaded, skipping")
            return True

        try:
            logger.info(f"Loading SAM 3 model from {self.model_id}...")
            logger.info("This may take a few minutes on first run (downloading ~3.4GB model)")

            # Load processor (handles text prompts and image preprocessing)
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Load model
            logger.info(f"Loading model to {self.device}...")
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32
            )

            # Move to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            self._is_loaded = True
            logger.info("Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_loaded = False
            return False

    def unload_model(self):
        """Unload model from memory to free up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear GPU cache
        if self.device in ["cuda", "mps"]:
            torch.cuda.empty_cache() if self.device == "cuda" else None

        self._is_loaded = False
        logger.info("Model unloaded from memory")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded

    def get_model(self):
        """Get the loaded model instance."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_processor(self):
        """Get the loaded processor instance."""
        if not self._is_loaded:
            raise RuntimeError("Processor not loaded. Call load_model() first.")
        return self.processor

    def get_device(self) -> str:
        """Get the device the model is loaded on."""
        return self.device
