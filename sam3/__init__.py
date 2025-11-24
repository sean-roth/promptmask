"""
SAM 3 integration for PromptMask.
Handles local model loading and inference using Hugging Face Transformers.
"""

from .model_loader import SAM3ModelLoader
from .inference import SAM3Inference

__all__ = ["SAM3ModelLoader", "SAM3Inference"]
