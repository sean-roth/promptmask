"""
SAM 3 Inference
Handles image segmentation using text prompts with SAM 3 model.
"""

import logging
import numpy as np
import torch
from PIL import Image
from typing import Tuple

logger = logging.getLogger(__name__)


class SAM3Inference:
    """Handles SAM 3 inference for image segmentation."""

    def __init__(self, model_loader):
        """
        Initialize inference engine.

        Args:
            model_loader: SAM3ModelLoader instance with loaded model
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.processor = model_loader.get_processor()
        self.device = model_loader.get_device()
        logger.info("Initialized SAM3Inference engine")

    def segment_image(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an image using a text prompt - runs locally on GPU/CPU.

        Args:
            image: PIL Image to segment
            text_prompt: Natural language description of what to segment
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            Tuple of (mask as numpy array, confidence score)
            mask: Binary mask where 1 = object, 0 = background
            confidence: Confidence score for the best mask
        """
        try:
            # Prepare inputs for local inference
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)

            # Run local inference (no API call)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results using official API
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            # Extract best mask if any found
            if len(results['masks']) > 0:
                best_mask = results['masks'][0].cpu().numpy().astype(np.uint8)
                best_score = float(results['scores'][0].item())

                logger.debug(
                    f"Segmented image with prompt '{text_prompt}' - "
                    f"found {len(results['masks'])} objects, best confidence: {best_score:.2f}"
                )

                return best_mask, best_score
            else:
                logger.warning(
                    f"No objects found for prompt '{text_prompt}' "
                    f"above confidence threshold {confidence_threshold}"
                )
                # Return empty mask
                h, w = image.size[1], image.size[0]
                return np.zeros((h, w), dtype=np.uint8), 0.0

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            # Return empty mask on error
            h, w = image.size[1], image.size[0]
            return np.zeros((h, w), dtype=np.uint8), 0.0

    def segment_batch(
        self,
        images: list,
        text_prompt: str,
        confidence_threshold: float = 0.7,
        progress_callback=None
    ) -> list:
        """
        Segment multiple images with the same text prompt.

        Args:
            images: List of PIL Images to segment
            text_prompt: Natural language description of what to segment
            confidence_threshold: Minimum confidence score (0-1)
            progress_callback: Optional callback function(current, total)

        Returns:
            List of tuples [(mask, confidence), ...] for each image
        """
        results = []
        total = len(images)

        for i, image in enumerate(images):
            mask, confidence = self.segment_image(image, text_prompt, confidence_threshold)
            results.append((mask, confidence))

            if progress_callback:
                progress_callback(i + 1, total)

        return results
