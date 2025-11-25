"""
SAM 3 Model Loader

Handles HuggingFace Transformers-based SAM 3 model loading with proper
device detection (CUDA/MPS/CPU) and local caching.

NOTE: SAM 3 was released Nov 19, 2025 and requires:
- transformers installed from main branch
- Access request approved at https://huggingface.co/facebook/sam3
"""

import os
import torch
from typing import Optional, Tuple
import logging

# Try to load .env for local development (optional for HF Spaces)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# SAM 3 specific imports - requires transformers from main branch
try:
    from transformers import Sam3Model, Sam3Processor
except ImportError:
    raise ImportError(
        "SAM 3 not found in transformers. Please install from main branch:\n"
        "pip install git+https://github.com/huggingface/transformers.git"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAM3ModelLoader:
    """
    Loads and manages SAM 3 model from HuggingFace.

    Features:
    - Automatic device detection (CUDA > MPS > CPU)
    - Local model caching
    - Environment variable authentication (works with .env or HF Spaces secrets)
    - Proper error handling for authentication
    """

    # Correct model ID for SAM 3
    MODEL_ID = "facebook/sam3"

    def __init__(self):
        # Get token from environment (works for both .env and HF Spaces secrets)
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        self.device = self._detect_device()
        self.model = None
        self.processor = None
        logger.info(f"Initialized SAM3ModelLoader with device: {self.device}")
        
        # Log token status (not the actual token!)
        if self.hf_token:
            logger.info(f"HuggingFace token found (starts with: {self.hf_token[:10]}...)")
        else:
            logger.warning("No HUGGINGFACE_TOKEN found in environment")

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

    def load_model(self, model_id: Optional[str] = None) -> Tuple[Sam3Model, Sam3Processor]:
        """
        Load SAM 3 model and processor from HuggingFace.

        Args:
            model_id: Optional model identifier. Defaults to facebook/sam3

        Returns:
            Tuple of (model, processor)

        Raises:
            ValueError: If HuggingFace authentication fails or access not granted
            RuntimeError: If model loading fails
        """
        if model_id is None:
            model_id = self.MODEL_ID

        # Check for token before attempting to load
        if not self.hf_token:
            raise ValueError(
                "No HUGGINGFACE_TOKEN found.\n\n"
                "For HuggingFace Spaces:\n"
                "  Add HUGGINGFACE_TOKEN in Settings → Repository Secrets\n\n"
                "For local use:\n"
                "  Create a .env file with: HUGGINGFACE_TOKEN=your_token_here"
            )

        try:
            logger.info(f"Loading model: {model_id}")
            logger.info("This may take a few minutes on first run (model is ~2GB)...")

            # Load processor with token
            self.processor = Sam3Processor.from_pretrained(
                model_id,
                token=self.hf_token
            )
            logger.info("Processor loaded successfully")

            # Load model with token
            # Use bfloat16 for efficiency on supported hardware
            dtype = torch.bfloat16 if self.device in ['cuda', 'mps'] else torch.float32
            
            self.model = Sam3Model.from_pretrained(
                model_id,
                token=self.hf_token,
                torch_dtype=dtype
            )

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.processor

        except OSError as e:
            error_str = str(e).lower()
            if "401" in str(e) or "authentication" in error_str or "unauthorized" in error_str:
                raise ValueError(
                    "Authentication failed. Please check:\n"
                    "1. You have requested access at: https://huggingface.co/facebook/sam3\n"
                    "2. Your access request has been approved\n"
                    "3. Your token is set correctly (HUGGINGFACE_TOKEN)\n"
                    "4. Your token has 'read' permissions\n\n"
                    "Get a token from: https://huggingface.co/settings/tokens"
                ) from e
            if "403" in str(e) or "access" in error_str or "gated" in error_str:
                raise ValueError(
                    "Access denied. SAM 3 is a gated model.\n\n"
                    "Please:\n"
                    "1. Go to https://huggingface.co/facebook/sam3\n"
                    "2. Click 'Access repository' and accept terms\n"
                    "3. Wait for approval (usually instant)\n"
                    "4. Try again"
                ) from e
            raise RuntimeError(f"Failed to load model: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading model: {e}") from e

    def get_model(self) -> Sam3Model:
        """Get the loaded model instance."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_processor(self) -> Sam3Processor:
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
