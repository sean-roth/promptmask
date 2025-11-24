"""
SAM 3 Model Loader

Handles HuggingFace Transformers-based SAM 3 model loading with proper
device detection (CUDA/MPS/CPU) and local caching.
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAM3ModelLoader:
    """
    Loads and manages SAM 3 model from HuggingFace.

    Features:
    - Automatic device detection (CUDA > MPS > CPU)
    - Local model caching
    - .env file authentication
    - Proper error handling for authentication
    """

    MODEL_ID = "facebook/sam-3-large"

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        self.device = self._detect_device()
        self.model = None
        self.processor = None
        logger.info(f"Initialized SAM3ModelLoader with device: {self.device}")

    def _detect_device(self) -> str:
        """
        Automatically detect the best available device.

        Returns:
            str: Device name ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_model(self, model_id: Optional[str] = None) -> Tuple[AutoModelForZeroShotObjectDetection, AutoProcessor]:
        """
        Load SAM 3 model and processor from HuggingFace.

        Args:
            model_id: Optional model identifier. Defaults to facebook/sam-3-large

        Returns:
            Tuple of (model, processor)

        Raises:
            ValueError: If HuggingFace authentication fails
            RuntimeError: If model loading fails
        """
        if model_id is None:
            model_id = self.MODEL_ID

        try:
            logger.info(f"Loading model: {model_id}")
            logger.info("This may take a few minutes on first run...")

            # Load processor with token
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=self.hf_token
            )
            logger.info("Processor loaded successfully")

            # Load model with token
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=self.hf_token
            )

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.processor

        except OSError as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                raise ValueError(
                    "Authentication failed. Please check your .env file:\n"
                    "1. Make sure .env exists in the project root\n"
                    "2. It should contain: HUGGINGFACE_TOKEN=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\n"
                    "3. See MIKE_START_HERE.md for detailed instructions"
                ) from e
            raise RuntimeError(f"Failed to load model: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading model: {e}") from e

    def get_model(self) -> AutoModelForZeroShotObjectDetection:
        """Get the loaded model instance."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_processor(self) -> AutoProcessor:
        """Get the loaded processor instance."""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Call load_model() first.")
        return self.processor

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded from memory")
