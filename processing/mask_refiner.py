"""
Mask Refinement
Cleans up masks with edge feathering, hole filling, and smoothing.
"""

import logging
from typing import List
import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class MaskRefiner:
    """Refines segmentation masks for better quality."""

    def __init__(self):
        """Initialize mask refiner."""
        logger.info("MaskRefiner initialized")

    def refine_mask(
        self,
        mask: np.ndarray,
        feather_radius: int = 10,
        fill_holes: bool = True,
        smooth_edges: bool = True
    ) -> np.ndarray:
        """
        Refine a single mask with various post-processing steps.

        Args:
            mask: Binary mask as numpy array
            feather_radius: Radius for edge feathering (0 = no feathering)
            fill_holes: Fill small holes in the mask
            smooth_edges: Smooth jagged edges

        Returns:
            Refined mask
        """
        refined = mask.copy()

        # Fill small holes
        if fill_holes:
            refined = self._fill_holes(refined)

        # Smooth edges
        if smooth_edges:
            refined = self._smooth_edges(refined)

        # Apply edge feathering
        if feather_radius > 0:
            refined = self._feather_edges(refined, feather_radius)

        return refined

    def refine_masks_batch(
        self,
        masks: List[np.ndarray],
        feather_radius: int = 10,
        fill_holes: bool = True,
        smooth_edges: bool = True,
        progress_callback=None
    ) -> List[np.ndarray]:
        """
        Refine a batch of masks.

        Args:
            masks: List of binary masks
            feather_radius: Radius for edge feathering
            fill_holes: Fill small holes in the masks
            smooth_edges: Smooth jagged edges
            progress_callback: Optional callback(current, total)

        Returns:
            List of refined masks
        """
        logger.info(f"Refining {len(masks)} masks")

        refined_masks = []
        for i, mask in enumerate(masks):
            refined = self.refine_mask(mask, feather_radius, fill_holes, smooth_edges)
            refined_masks.append(refined)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(masks))

        logger.info("Mask refinement complete")
        return refined_masks

    def _fill_holes(self, mask: np.ndarray, max_hole_size: int = 1000) -> np.ndarray:
        """
        Fill small holes in a binary mask.

        Args:
            mask: Binary mask
            max_hole_size: Maximum hole size in pixels to fill

        Returns:
            Mask with holes filled
        """
        # Use binary fill holes
        filled = ndimage.binary_fill_holes(mask).astype(np.uint8)

        # Only keep filled regions smaller than max_hole_size
        # This prevents filling large intentional gaps
        holes = filled - mask
        labeled_holes, num_holes = ndimage.label(holes)

        for i in range(1, num_holes + 1):
            hole_size = np.sum(labeled_holes == i)
            if hole_size > max_hole_size:
                # Remove this hole (don't fill it)
                filled[labeled_holes == i] = 0

        return filled

    def _smooth_edges(self, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Smooth jagged edges using morphological operations.

        Args:
            mask: Binary mask
            iterations: Number of smoothing iterations

        Returns:
            Mask with smoothed edges
        """
        # Use morphological closing then opening
        # This removes small protrusions and fills small gaps
        smoothed = mask.copy()

        # Closing: removes small holes and smooths edges
        smoothed = ndimage.binary_closing(smoothed, iterations=iterations)

        # Opening: removes small protrusions
        smoothed = ndimage.binary_opening(smoothed, iterations=iterations)

        return smoothed.astype(np.uint8)

    def _feather_edges(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply edge feathering to create soft edges.

        Args:
            mask: Binary mask
            radius: Feather radius in pixels

        Returns:
            Mask with feathered edges (may have values 0-255)
        """
        if radius <= 0:
            return mask

        # Convert to PIL Image for gaussian blur
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Apply gaussian blur
        feathered = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Convert back to numpy
        feathered_array = np.array(feathered)

        # Normalize to 0-1 range, then threshold
        # This creates a soft edge while maintaining the overall shape
        feathered_normalized = feathered_array.astype(np.float32) / 255.0

        # Apply sigmoid to create smooth transition
        feathered_smooth = self._sigmoid(feathered_normalized, steepness=10.0, midpoint=0.5)

        return (feathered_smooth * 255).astype(np.uint8)

    def _sigmoid(self, x: np.ndarray, steepness: float = 10.0, midpoint: float = 0.5) -> np.ndarray:
        """
        Apply sigmoid function for smooth transition.

        Args:
            x: Input array (0-1 range)
            steepness: Steepness of the sigmoid curve
            midpoint: Midpoint of the transition

        Returns:
            Sigmoid-transformed array
        """
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))

    def remove_small_regions(
        self,
        mask: np.ndarray,
        min_size: int = 100
    ) -> np.ndarray:
        """
        Remove small disconnected regions from mask.

        Args:
            mask: Binary mask
            min_size: Minimum region size to keep (in pixels)

        Returns:
            Mask with small regions removed
        """
        # Label connected components
        labeled, num_features = ndimage.label(mask)

        # Remove small regions
        cleaned = mask.copy()
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled == i)
            if region_size < min_size:
                cleaned[labeled == i] = 0

        return cleaned

    def apply_morphological_operations(
        self,
        mask: np.ndarray,
        operation: str = "close",
        kernel_size: int = 5,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological operations to mask.

        Args:
            mask: Binary mask
            operation: Operation type ('dilate', 'erode', 'open', 'close')
            kernel_size: Size of the structuring element
            iterations: Number of times to apply the operation

        Returns:
            Processed mask
        """
        # Create structuring element
        structure = np.ones((kernel_size, kernel_size))

        if operation == "dilate":
            result = ndimage.binary_dilation(mask, structure=structure, iterations=iterations)
        elif operation == "erode":
            result = ndimage.binary_erosion(mask, structure=structure, iterations=iterations)
        elif operation == "open":
            result = ndimage.binary_opening(mask, structure=structure, iterations=iterations)
        elif operation == "close":
            result = ndimage.binary_closing(mask, structure=structure, iterations=iterations)
        else:
            logger.warning(f"Unknown operation '{operation}', returning original mask")
            return mask

        return result.astype(np.uint8)
