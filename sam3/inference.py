"""
SAM 3 Inference Engine

Provides high-level inference API for text-prompted instance segmentation
using the correct HuggingFace Transformers post-processing pipeline.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Callable
import logging

from .model_loader import SAM3ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAM3Inference:
    """
    High-level inference engine for SAM 3 segmentation.

    Uses the correct Transformers API with post_process_instance_segmentation
    for proper mask extraction and scoring.
    """

    def __init__(self, model_loader: SAM3ModelLoader):
        """
        Initialize inference engine.

        Args:
            model_loader: Pre-initialized SAM3ModelLoader instance
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.processor = model_loader.get_processor()
        self.device = model_loader.device

    def segment_image(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[np.ndarray, float]:
        """
        Segment a single image using text prompt.

        Args:
            image: PIL Image to segment
            text_prompt: Text description of object to segment
            confidence_threshold: Minimum confidence score (0.0-1.0)

        Returns:
            Tuple of (binary_mask, confidence_score)
            - binary_mask: numpy array of shape (H, W) with 0/255 values
            - confidence_score: float confidence of best match
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process using correct Transformers API
            results = self.processor.post_process_instance_segmentation(
                outputs,
                target_sizes=[image.size[::-1]]  # (height, width)
            )[0]

            # Extract best mask above threshold
            if len(results['masks']) == 0:
                logger.warning("No masks found")
                return np.zeros(image.size[::-1], dtype=np.uint8), 0.0

            # Get scores and find best match
            scores = results['scores'].cpu().numpy()
            masks = results['masks'].cpu().numpy()

            # Filter by confidence
            valid_indices = scores >= confidence_threshold
            if not valid_indices.any():
                logger.warning(f"No masks above threshold {confidence_threshold}")
                return np.zeros(image.size[::-1], dtype=np.uint8), 0.0

            # Get best mask
            best_idx = scores.argmax()
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx]

            # Convert to binary mask (0/255)
            binary_mask = (best_mask * 255).astype(np.uint8)

            logger.info(f"Segmented with confidence: {best_score:.3f}")
            return binary_mask, best_score

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise RuntimeError(f"Failed to segment image: {e}") from e

    def segment_frame_sequence(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        confidence_threshold: float = 0.7,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Segment a sequence of video frames.

        Uses the correct SAM 3 API from Branch 2 for each frame.

        Args:
            frames: List of PIL Images (video frames)
            text_prompt: Text description of object to segment
            confidence_threshold: Minimum confidence score
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Tuple of (masks, scores)
            - masks: List of binary numpy arrays
            - scores: List of confidence scores
        """
        masks = []
        scores = []

        total_frames = len(frames)
        logger.info(f"Segmenting {total_frames} frames...")

        for i, frame in enumerate(frames):
            try:
                # Use correct Branch 2 inference
                mask, score = self.segment_image(
                    frame,
                    text_prompt,
                    confidence_threshold
                )
                masks.append(mask)
                scores.append(score)

                # Report progress
                if progress_callback:
                    progress_callback(i + 1, total_frames)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{total_frames} frames")

            except Exception as e:
                logger.error(f"Failed to process frame {i}: {e}")
                # Add empty mask on failure
                if len(masks) > 0:
                    masks.append(np.zeros_like(masks[0]))
                else:
                    masks.append(np.zeros((frames[0].size[1], frames[0].size[0]), dtype=np.uint8))
                scores.append(0.0)

        logger.info(f"Completed segmentation of {total_frames} frames")
        return masks, scores

    def segment_batch(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
        confidence_threshold: float = 0.7
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Segment multiple images with different prompts.

        Args:
            images: List of PIL Images
            text_prompts: List of text prompts (one per image)
            confidence_threshold: Minimum confidence score

        Returns:
            List of (mask, score) tuples
        """
        if len(images) != len(text_prompts):
            raise ValueError("Number of images must match number of prompts")

        results = []
        for image, prompt in zip(images, text_prompts):
            mask, score = self.segment_image(image, prompt, confidence_threshold)
            results.append((mask, score))

        return results
