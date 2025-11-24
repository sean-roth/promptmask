"""
SAM 3 Model Loader
Downloads and loads SAM 3 model locally using Hugging Face Transformers.
"""

import logging
import torch
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger(__name__)


class SAM3ModelLoader:
    """Handles loading and caching of SAM 3 model."""

    def __init__(self):
        """Initialize model loader."""
        self.model = None
        self.processor = None
        self.device = self._get_device()
        logger.info(f"Initialized SAM3ModelLoader with device: {self.device}")

    def _get_device(self) -> torch.device:
        """
        Auto-detect best available device.

        Returns:
            torch.device: Best available device (CUDA > MPS > CPU)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (no GPU available)")

        return device

    def load_model(self):
        """
        Download and load SAM 3 model locally.

        First run downloads ~3GB to ~/.cache/huggingface/hub/
        Subsequent runs use cached model.

        Returns:
            tuple: (model, processor)
        """
        logger.info("Loading SAM 3 model from Hugging Face...")

        try:
            # Download and load model (cached after first run)
            self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self.processor = Sam3Processor.from_pretrained("facebook/sam3")

            logger.info("SAM 3 model loaded successfully!")
            logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")

            return self.model, self.processor

        except Exception as e:
            logger.error(f"Failed to load SAM 3 model: {e}")
            logger.error("Make sure you're authenticated with Hugging Face:")
            logger.error("  Run: huggingface-cli login")
            raise

    def get_model(self):
        """Get loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_processor(self):
        """Get loaded processor."""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_model() first.")
        return self.processor

    def get_device(self):
        """Get device being used."""
        return self.device

    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
