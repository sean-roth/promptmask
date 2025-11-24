"""
Mask Refinement Module

Provides edge enhancement, feathering, and quality improvements for
segmentation masks.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from PIL import Image, ImageFilter
import cv2
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskRefiner:
    """
    Refines segmentation masks for better visual quality.

    Features:
    - Edge feathering (soft transitions)
    - Morphological operations (open/close)
    - Hole filling
    - Edge smoothing
    """

    def __init__(self):
        """Initialize mask refiner."""
        pass

    def refine(
        self,
        mask: np.ndarray,
        feather_radius: int = 5,
        remove_holes: bool = True,
        smooth_edges: bool = True
    ) -> np.ndarray:
        """
        Apply full refinement pipeline to mask.

        Args:
            mask: Binary mask (numpy array, 0/255 values)
            feather_radius: Radius for edge feathering in pixels
            remove_holes: Whether to fill small holes
            smooth_edges: Whether to smooth edges

        Returns:
            Refined binary mask
        """
        refined = mask.copy()

        # Remove small holes
        if remove_holes:
            refined = self.fill_holes(refined)

        # Smooth edges
        if smooth_edges:
            refined = self.smooth_edges(refined)

        # Apply feathering
        if feather_radius > 0:
            refined = self.feather_edges(refined, feather_radius)

        return refined

    def feather_edges(
        self,
        mask: np.ndarray,
        radius: int = 5
    ) -> np.ndarray:
        """
        Apply Gaussian blur to mask edges for soft transitions.

        Args:
            mask: Binary mask (0/255)
            radius: Feather radius in pixels

        Returns:
            Feathered mask
        """
        if radius == 0:
            return mask

        # Convert to float for processing
        mask_float = mask.astype(np.float32) / 255.0

        # Apply Gaussian blur
        sigma = radius / 3.0  # Convert radius to sigma
        feathered = gaussian_filter(mask_float, sigma=sigma)

        # Convert back to uint8
        feathered = (feathered * 255).astype(np.uint8)

        return feathered

    def smooth_edges(
        self,
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Smooth mask edges using morphological operations.

        Args:
            mask: Binary mask (0/255)
            kernel_size: Size of morphological kernel

        Returns:
            Smoothed mask
        """
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)

        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply morphological closing (dilation then erosion)
        # This removes small gaps and smooths boundaries
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Apply morphological opening (erosion then dilation)
        # This removes small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Convert back to 0/255
        smoothed = (opened * 255).astype(np.uint8)

        return smoothed

    def fill_holes(
        self,
        mask: np.ndarray,
        max_hole_size: int = 500
    ) -> np.ndarray:
        """
        Fill small holes in mask.

        Args:
            mask: Binary mask (0/255)
            max_hole_size: Maximum hole size in pixels to fill

        Returns:
            Mask with holes filled
        """
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)

        # Find contours
        contours, hierarchy = cv2.findContours(
            255 - binary_mask * 255,  # Invert for hole detection
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Fill small holes
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < max_hole_size:
                cv2.drawContours(binary_mask, [contour], -1, 1, -1)

        # Convert back to 0/255
        filled = (binary_mask * 255).astype(np.uint8)

        return filled

    def erode(
        self,
        mask: np.ndarray,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Erode mask to shrink boundaries.

        Args:
            mask: Binary mask (0/255)
            iterations: Number of erosion iterations

        Returns:
            Eroded mask
        """
        binary_mask = (mask > 127).astype(bool)
        eroded = binary_erosion(binary_mask, iterations=iterations)
        return (eroded * 255).astype(np.uint8)

    def dilate(
        self,
        mask: np.ndarray,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Dilate mask to expand boundaries.

        Args:
            mask: Binary mask (0/255)
            iterations: Number of dilation iterations

        Returns:
            Dilated mask
        """
        binary_mask = (mask > 127).astype(bool)
        dilated = binary_dilation(binary_mask, iterations=iterations)
        return (dilated * 255).astype(np.uint8)

    def adjust_boundary(
        self,
        mask: np.ndarray,
        offset: int
    ) -> np.ndarray:
        """
        Adjust mask boundary inward (negative) or outward (positive).

        Args:
            mask: Binary mask (0/255)
            offset: Pixel offset (positive = expand, negative = shrink)

        Returns:
            Adjusted mask
        """
        if offset > 0:
            return self.dilate(mask, iterations=offset)
        elif offset < 0:
            return self.erode(mask, iterations=abs(offset))
        else:
            return mask

    def get_edge_mask(
        self,
        mask: np.ndarray,
        thickness: int = 3
    ) -> np.ndarray:
        """
        Extract edge region of mask.

        Args:
            mask: Binary mask (0/255)
            thickness: Edge thickness in pixels

        Returns:
            Edge mask (only the boundary region)
        """
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)

        # Get eroded version
        eroded = self.erode(mask, iterations=thickness)
        eroded_binary = (eroded > 127).astype(np.uint8)

        # Edge is difference between original and eroded
        edge = binary_mask - eroded_binary

        # Convert to 0/255
        edge_mask = (edge * 255).astype(np.uint8)

        return edge_mask

    def create_alpha_matte(
        self,
        mask: np.ndarray,
        feather_radius: int = 10
    ) -> np.ndarray:
        """
        Create alpha matte (0-255 alpha channel) from mask.

        Args:
            mask: Binary mask (0/255)
            feather_radius: Feather radius for soft edges

        Returns:
            Alpha matte (0-255)
        """
        # Apply strong feathering for alpha matte
        alpha = self.feather_edges(mask, radius=feather_radius)

        return alpha

    def apply_to_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply mask to image with optional background.

        Args:
            image: RGB image (numpy array)
            mask: Binary mask (0/255)
            background: Optional background image or color

        Returns:
            Composited image
        """
        # Normalize mask to 0-1 range
        alpha = mask.astype(np.float32) / 255.0

        # Expand to 3 channels if needed
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=-1)

        if background is None:
            # No background, just apply mask
            result = (image * alpha).astype(np.uint8)
        else:
            # Composite with background
            if isinstance(background, (int, float, tuple)):
                # Solid color background
                bg = np.full_like(image, background, dtype=np.uint8)
            else:
                bg = background

            # Alpha blend
            result = (alpha * image + (1 - alpha) * bg).astype(np.uint8)

        return result
