"""
SAM 3 Model Integration Module

This module provides HuggingFace Transformers-based SAM 3 model loading
and inference capabilities for text-prompted instance segmentation.
"""

from .model_loader import SAM3ModelLoader
from .inference import SAM3Inference

__all__ = ['SAM3ModelLoader', 'SAM3Inference']
