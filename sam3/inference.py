"""
SAM 3 Inference Engine
Handles inference on images and videos using the loaded SAM 3 model.
"""

import logging
from typing import Optional, Tuple, List
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SAM3Inference:
    """Performs inference with SAM 3 model on images."""

    def __init__(self, model_loader):
        """
        Initialize inference engine.

        Args:
            model_loader: Instance of SAM3ModelLoader with loaded model
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.processor = model_loader.get_processor()
        self.device = model_loader.get_device()

        logger.info("SAM3Inference initialized")

    def segment_image(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an image using a text prompt.

        Args:
            image: PIL Image to segment
            text_prompt: Natural language description of what to segment
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            Tuple of (mask as numpy array, confidence score)
            mask: Binary mask where 1 = object, 0 = background
            confidence: Average confidence score for the segmentation
        """
        try:
            # Preprocess image and text
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract masks and scores
            masks = outputs.pred_masks[0].cpu().numpy()  # Shape: [num_masks, H, W]
            scores = outputs.iou_scores[0].cpu().numpy()  # Shape: [num_masks]

            # Select best mask above threshold
            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx]

            if best_score < confidence_threshold:
                logger.warning(
                    f"Best mask confidence ({best_score:.2f}) below threshold ({confidence_threshold})"
                )

            # Convert to binary mask
            binary_mask = (best_mask > 0.5).astype(np.uint8)

            logger.debug(
                f"Segmented image with prompt '{text_prompt}' - confidence: {best_score:.2f}"
            )

            return binary_mask, best_score

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            # Return empty mask on error
            h, w = image.size[1], image.size[0]
            return np.zeros((h, w), dtype=np.uint8), 0.0

    def segment_frame_sequence(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        confidence_threshold: float = 0.7,
        use_temporal_tracking: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Segment a sequence of video frames with temporal consistency.

        Args:
            frames: List of PIL Images (video frames)
            text_prompt: Natural language description of what to segment
            confidence_threshold: Minimum confidence score (0-1)
            use_temporal_tracking: Use previous frame's mask to improve tracking
            progress_callback: Optional callback function(current_frame, total_frames)

        Returns:
            Tuple of (list of masks, list of confidence scores)
        """
        masks = []
        scores = []
        prev_mask = None

        total_frames = len(frames)
        logger.info(f"Processing {total_frames} frames with prompt: '{text_prompt}'")

        for i, frame in enumerate(frames):
            try:
                # TODO: Implement temporal tracking using prev_mask
                # For now, process each frame independently
                mask, score = self.segment_image(frame, text_prompt, confidence_threshold)

                masks.append(mask)
                scores.append(score)
                prev_mask = mask

                # Call progress callback
                if progress_callback:
                    progress_callback(i + 1, total_frames)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{total_frames} frames")

            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                # Use previous mask or empty mask on error
                if prev_mask is not None:
                    masks.append(prev_mask)
                    scores.append(0.0)
                else:
                    h, w = frame.size[1], frame.size[0]
                    masks.append(np.zeros((h, w), dtype=np.uint8))
                    scores.append(0.0)

        avg_score = np.mean(scores)
        logger.info(f"Completed processing {total_frames} frames - avg confidence: {avg_score:.2f}")

        return masks, scores

    def get_mask_as_image(self, mask: np.ndarray) -> Image.Image:
        """
        Convert a binary mask to a PIL Image.

        Args:
            mask: Binary mask as numpy array

        Returns:
            PIL Image (black and white)
        """
        # Scale to 0-255 range
        mask_img = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask_img, mode='L')

    def overlay_mask_on_image(
        self,
        image: Image.Image,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Overlay a mask on an image with transparency.

        Args:
            image: Original PIL Image
            mask: Binary mask as numpy array
            color: RGB color for the mask overlay (default: green)
            alpha: Transparency (0=transparent, 1=opaque)

        Returns:
            PIL Image with mask overlay
        """
        # Convert image to numpy
        img_array = np.array(image.convert('RGB'))

        # Create colored mask
        colored_mask = np.zeros_like(img_array)
        for i in range(3):
            colored_mask[:, :, i] = mask * color[i]

        # Blend with original image
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended = img_array * (1 - mask_3d * alpha) + colored_mask * alpha
        blended = blended.astype(np.uint8)

        return Image.fromarray(blended)
