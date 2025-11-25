"""
SAM 3 Inference Engine

Provides high-level inference API for text-prompted instance segmentation
using SAM 3's Promptable Concept Segmentation (PCS) capabilities.

SAM 3 can segment all instances of an open-vocabulary concept specified
by a short text phrase (e.g., "person", "yellow school bus").
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

    Uses SAM 3's text-prompted Promptable Concept Segmentation (PCS)
    to find and segment all matching instances.
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
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> Tuple[np.ndarray, float]:
        """
        Segment a single image using text prompt.

        Args:
            image: PIL Image to segment
            text_prompt: Text description of object to segment (e.g., "person")
            confidence_threshold: Minimum confidence score for detection (0.0-1.0)
            mask_threshold: Threshold for mask binarization (0.0-1.0)

        Returns:
            Tuple of (binary_mask, confidence_score)
            - binary_mask: numpy array of shape (H, W) with 0/255 values
            - confidence_score: float confidence of best match
        """
        try:
            # Get original image size
            original_size = image.size  # (width, height)
            logger.info(f"[DEBUG] Image size: {original_size}, mode: {image.mode}")
            
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"[DEBUG] Converted to RGB")
            
            # Prepare inputs using SAM 3 processor
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)

            # Debug: Log input keys and shapes
            logger.info(f"[DEBUG] Input keys: {inputs.keys()}")
            for key, val in inputs.items():
                if torch.is_tensor(val):
                    logger.info(f"[DEBUG] inputs['{key}']: shape={val.shape}, dtype={val.dtype}")
                else:
                    logger.info(f"[DEBUG] inputs['{key}']: {type(val)}")

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Debug: Log output attributes
            logger.info(f"[DEBUG] Output type: {type(outputs)}")
            if hasattr(outputs, 'pred_masks'):
                logger.info(f"[DEBUG] pred_masks shape: {outputs.pred_masks.shape}")
                logger.info(f"[DEBUG] pred_masks min/max: {outputs.pred_masks.min():.4f} / {outputs.pred_masks.max():.4f}")
            if hasattr(outputs, 'pred_boxes'):
                logger.info(f"[DEBUG] pred_boxes shape: {outputs.pred_boxes.shape}")
            if hasattr(outputs, 'pred_logits'):
                logger.info(f"[DEBUG] pred_logits shape: {outputs.pred_logits.shape}")
                logger.info(f"[DEBUG] pred_logits min/max: {outputs.pred_logits.min():.4f} / {outputs.pred_logits.max():.4f}")
            if hasattr(outputs, 'scores'):
                logger.info(f"[DEBUG] scores: {outputs.scores}")

            # Post-process using SAM 3's instance segmentation
            # Use original_sizes from processor if available (per official docs)
            if hasattr(inputs, 'original_sizes') and inputs.original_sizes is not None:
                target_sizes = inputs.original_sizes.tolist()
                logger.info(f"[DEBUG] Using processor original_sizes: {target_sizes}")
            else:
                target_sizes = [[original_size[1], original_size[0]]]  # (height, width)
                logger.info(f"[DEBUG] Using manual target_sizes: {target_sizes}")
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes
            )[0]

            # Debug: Log results
            logger.info(f"[DEBUG] Results keys: {results.keys() if isinstance(results, dict) else 'not a dict'}")
            if isinstance(results, dict):
                for key, val in results.items():
                    if torch.is_tensor(val):
                        logger.info(f"[DEBUG] results['{key}']: shape={val.shape}")
                    elif isinstance(val, (list, np.ndarray)):
                        logger.info(f"[DEBUG] results['{key}']: len={len(val)}")
                    else:
                        logger.info(f"[DEBUG] results['{key}']: {val}")

            # Extract masks and scores
            if len(results.get('masks', [])) == 0:
                logger.warning(f"No masks found for prompt: '{text_prompt}'")
                return np.zeros((original_size[1], original_size[0]), dtype=np.uint8), 0.0

            masks = results['masks']
            scores = results['scores']
            
            logger.info(f"[DEBUG] Found {len(masks)} masks, scores: {scores}")
            
            # Convert to numpy if tensor
            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()

            # Combine all masks above threshold into one
            # (SAM 3 can return multiple instances)
            combined_mask = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
            valid_scores = []
            
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score >= confidence_threshold:
                    combined_mask = np.maximum(combined_mask, mask.astype(np.float32))
                    valid_scores.append(float(score))

            if not valid_scores:
                logger.warning(f"No masks above threshold {confidence_threshold}")
                return np.zeros((original_size[1], original_size[0]), dtype=np.uint8), 0.0

            # Calculate average confidence
            avg_score = np.mean(valid_scores)

            # Convert to binary mask (0/255)
            binary_mask = (combined_mask > mask_threshold).astype(np.uint8) * 255

            logger.info(f"Segmented '{text_prompt}' with {len(valid_scores)} instances, avg confidence: {avg_score:.3f}")
            return binary_mask, float(avg_score)

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to segment image: {e}") from e

    def segment_frame_sequence(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        confidence_threshold: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Segment a sequence of video frames.

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
        logger.info(f"Segmenting {total_frames} frames with prompt: '{text_prompt}'")

        for i, frame in enumerate(frames):
            try:
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
        avg_score = np.mean([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0.0
        logger.info(f"Average confidence across all frames: {avg_score:.3f}")
        
        return masks, scores

    def segment_batch(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
        confidence_threshold: float = 0.5
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
