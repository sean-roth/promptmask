"""
Temporal Smoothing
Reduces mask flickering between frames using temporal filtering.
"""

import logging
from typing import List
import numpy as np
from scipy import ndimage
from PIL import Image

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Applies temporal smoothing to mask sequences."""

    def __init__(self, window_size: int = 3):
        """
        Initialize temporal smoother.

        Args:
            window_size: Number of frames to consider for smoothing (should be odd)
        """
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        logger.info(f"TemporalSmoother initialized with window_size={self.window_size}")

    def smooth_masks(
        self,
        masks: List[np.ndarray],
        strength: float = 0.5,
        progress_callback=None
    ) -> List[np.ndarray]:
        """
        Apply temporal smoothing to a sequence of masks.

        Args:
            masks: List of binary mask arrays
            strength: Smoothing strength (0-1, where 0=no smoothing, 1=max smoothing)
            progress_callback: Optional callback(current, total)

        Returns:
            List of smoothed masks
        """
        if len(masks) == 0:
            return masks

        if strength == 0:
            logger.info("Smoothing strength=0, returning original masks")
            return masks

        logger.info(f"Applying temporal smoothing to {len(masks)} masks (strength={strength})")

        smoothed_masks = []
        half_window = self.window_size // 2

        for i, mask in enumerate(masks):
            # Determine the window range
            start_idx = max(0, i - half_window)
            end_idx = min(len(masks), i + half_window + 1)

            # Get neighboring masks
            window_masks = masks[start_idx:end_idx]

            # Apply temporal averaging
            smoothed = self._apply_temporal_average(window_masks, strength)

            smoothed_masks.append(smoothed)

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(masks))

        logger.info(f"Temporal smoothing complete")
        return smoothed_masks

    def _apply_temporal_average(
        self,
        masks: List[np.ndarray],
        strength: float
    ) -> np.ndarray:
        """
        Apply temporal averaging to a window of masks.

        Args:
            masks: List of masks in the temporal window
            strength: Smoothing strength (0-1)

        Returns:
            Smoothed mask
        """
        if len(masks) == 1:
            return masks[0]

        # Stack masks and compute weighted average
        mask_stack = np.stack(masks, axis=0).astype(np.float32)

        # Use gaussian weights favoring the center frame
        weights = self._get_gaussian_weights(len(masks))
        weights = weights.reshape(-1, 1, 1)

        # Weighted average
        averaged = np.sum(mask_stack * weights, axis=0)

        # Blend with original center mask
        center_idx = len(masks) // 2
        original = masks[center_idx].astype(np.float32)

        blended = (1 - strength) * original + strength * averaged

        # Threshold back to binary
        return (blended > 0.5).astype(np.uint8)

    def _get_gaussian_weights(self, size: int) -> np.ndarray:
        """
        Get gaussian weights for temporal window.

        Args:
            size: Window size

        Returns:
            Normalized weights
        """
        center = size // 2
        sigma = size / 6.0  # Standard deviation

        weights = np.array([
            np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
            for i in range(size)
        ])

        # Normalize
        weights = weights / np.sum(weights)

        return weights

    def reduce_flickering(
        self,
        masks: List[np.ndarray],
        threshold: float = 0.3,
        progress_callback=None
    ) -> List[np.ndarray]:
        """
        Detect and reduce flickering artifacts in masks.

        Flickering occurs when pixels rapidly change between frames.
        This method identifies and stabilizes such pixels.

        Args:
            masks: List of binary mask arrays
            threshold: Flicker detection threshold (0-1)
            progress_callback: Optional callback(current, total)

        Returns:
            List of masks with reduced flickering
        """
        if len(masks) < 3:
            return masks

        logger.info(f"Reducing flickering in {len(masks)} masks")

        # Compute frame-to-frame differences
        diffs = []
        for i in range(len(masks) - 1):
            diff = np.abs(masks[i+1].astype(np.float32) - masks[i].astype(np.float32))
            diffs.append(diff)

        # Identify consistently flickering pixels
        # (pixels that change frequently)
        diff_stack = np.stack(diffs, axis=0)
        flicker_score = np.mean(diff_stack, axis=0)

        # Create flicker mask
        flicker_mask = flicker_score > threshold

        # For flickering pixels, use temporal median
        stabilized_masks = []
        for i, mask in enumerate(masks):
            stabilized = mask.copy()

            # Get temporal neighborhood
            start_idx = max(0, i - 1)
            end_idx = min(len(masks), i + 2)
            neighborhood = masks[start_idx:end_idx]

            # Apply median filter to flickering regions
            if len(neighborhood) > 1:
                median_mask = np.median(np.stack(neighborhood, axis=0), axis=0)
                stabilized[flicker_mask] = median_mask[flicker_mask]

            stabilized_masks.append(stabilized.astype(np.uint8))

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(masks))

        logger.info(f"Flickering reduction complete")
        return stabilized_masks
