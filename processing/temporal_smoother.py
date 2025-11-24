"""
Temporal Mask Smoother

Reduces flicker and improves frame-to-frame consistency in video segmentation
masks using temporal filtering and interpolation.
"""

import numpy as np
from typing import List, Optional
from scipy.ndimage import median_filter, gaussian_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Applies temporal smoothing to video segmentation masks.

    Features:
    - Median filtering across time
    - Gaussian temporal smoothing
    - Edge-aware interpolation
    - Configurable window sizes
    """

    def __init__(self, window_size: int = 3):
        """
        Initialize temporal smoother.

        Args:
            window_size: Size of temporal window (must be odd, default: 3)
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")

        self.window_size = window_size

    def smooth(
        self,
        masks: List[np.ndarray],
        method: str = 'median',
        sigma: float = 1.0
    ) -> List[np.ndarray]:
        """
        Apply temporal smoothing to mask sequence.

        Args:
            masks: List of binary masks (numpy arrays)
            method: Smoothing method ('median', 'gaussian', or 'hybrid')
            sigma: Standard deviation for gaussian smoothing

        Returns:
            List of smoothed masks
        """
        if not masks:
            return masks

        if len(masks) == 1:
            logger.info("Single frame, no temporal smoothing applied")
            return masks

        logger.info(f"Applying {method} temporal smoothing to {len(masks)} frames...")

        if method == 'median':
            smoothed = self._median_smooth(masks)
        elif method == 'gaussian':
            smoothed = self._gaussian_smooth(masks, sigma)
        elif method == 'hybrid':
            smoothed = self._hybrid_smooth(masks, sigma)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        logger.info("Temporal smoothing complete")
        return smoothed

    def _median_smooth(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply median filtering across temporal dimension.

        Args:
            masks: List of binary masks

        Returns:
            List of smoothed masks
        """
        # Stack masks into 3D array (time, height, width)
        mask_stack = np.stack(masks, axis=0).astype(np.float32)

        # Normalize to 0-1 range
        mask_stack = mask_stack / 255.0

        # Apply median filter along time axis
        smoothed_stack = np.zeros_like(mask_stack)

        for i in range(len(masks)):
            # Define temporal window
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(len(masks), i + self.window_size // 2 + 1)

            # Extract window
            window = mask_stack[start_idx:end_idx]

            # Compute median
            smoothed_stack[i] = np.median(window, axis=0)

        # Convert back to 0-255 range
        smoothed_masks = [(mask * 255).astype(np.uint8) for mask in smoothed_stack]

        return smoothed_masks

    def _gaussian_smooth(
        self,
        masks: List[np.ndarray],
        sigma: float
    ) -> List[np.ndarray]:
        """
        Apply Gaussian smoothing across temporal dimension.

        Args:
            masks: List of binary masks
            sigma: Standard deviation for Gaussian kernel

        Returns:
            List of smoothed masks
        """
        # Stack masks into 3D array
        mask_stack = np.stack(masks, axis=0).astype(np.float32)

        # Normalize to 0-1 range
        mask_stack = mask_stack / 255.0

        # Apply Gaussian filter along time axis (axis 0)
        smoothed_stack = gaussian_filter(mask_stack, sigma=(sigma, 0, 0), mode='nearest')

        # Threshold back to binary
        smoothed_stack = (smoothed_stack > 0.5).astype(np.float32)

        # Convert back to 0-255 range
        smoothed_masks = [(mask * 255).astype(np.uint8) for mask in smoothed_stack]

        return smoothed_masks

    def _hybrid_smooth(
        self,
        masks: List[np.ndarray],
        sigma: float
    ) -> List[np.ndarray]:
        """
        Apply hybrid median + Gaussian smoothing.

        Args:
            masks: List of binary masks
            sigma: Standard deviation for Gaussian kernel

        Returns:
            List of smoothed masks
        """
        # First apply median filtering
        median_smoothed = self._median_smooth(masks)

        # Then apply Gaussian smoothing
        gaussian_smoothed = self._gaussian_smooth(median_smoothed, sigma)

        return gaussian_smoothed

    def interpolate_missing(
        self,
        masks: List[Optional[np.ndarray]],
        max_gap: int = 5
    ) -> List[np.ndarray]:
        """
        Interpolate missing masks in sequence.

        Args:
            masks: List of masks (None for missing frames)
            max_gap: Maximum gap size to interpolate

        Returns:
            List of masks with gaps filled
        """
        result = []
        n = len(masks)

        for i in range(n):
            if masks[i] is not None:
                result.append(masks[i])
            else:
                # Find nearest non-None masks
                prev_idx = self._find_previous_valid(masks, i)
                next_idx = self._find_next_valid(masks, i)

                if prev_idx is None and next_idx is None:
                    # No valid masks, use zeros
                    if result:
                        result.append(np.zeros_like(result[0]))
                    else:
                        raise ValueError("No valid masks to interpolate from")

                elif prev_idx is None:
                    # Use next mask
                    result.append(masks[next_idx].copy())

                elif next_idx is None:
                    # Use previous mask
                    result.append(masks[prev_idx].copy())

                else:
                    # Interpolate between prev and next
                    gap_size = next_idx - prev_idx - 1

                    if gap_size <= max_gap:
                        # Linear interpolation
                        alpha = (i - prev_idx) / (next_idx - prev_idx)
                        interpolated = (
                            (1 - alpha) * masks[prev_idx].astype(np.float32) +
                            alpha * masks[next_idx].astype(np.float32)
                        )
                        result.append(interpolated.astype(np.uint8))
                    else:
                        # Gap too large, use previous mask
                        result.append(masks[prev_idx].copy())

        return result

    def _find_previous_valid(
        self,
        masks: List[Optional[np.ndarray]],
        index: int
    ) -> Optional[int]:
        """Find index of previous valid mask."""
        for i in range(index - 1, -1, -1):
            if masks[i] is not None:
                return i
        return None

    def _find_next_valid(
        self,
        masks: List[Optional[np.ndarray]],
        index: int
    ) -> Optional[int]:
        """Find index of next valid mask."""
        for i in range(index + 1, len(masks)):
            if masks[i] is not None:
                return i
        return None

    def remove_flicker(
        self,
        masks: List[np.ndarray],
        threshold: float = 0.3
    ) -> List[np.ndarray]:
        """
        Remove flickering artifacts by stabilizing small changes.

        Args:
            masks: List of binary masks
            threshold: Change threshold below which to stabilize (0-1)

        Returns:
            List of de-flickered masks
        """
        if len(masks) <= 1:
            return masks

        result = [masks[0]]

        for i in range(1, len(masks)):
            prev_mask = result[-1].astype(np.float32) / 255.0
            curr_mask = masks[i].astype(np.float32) / 255.0

            # Calculate change ratio
            diff = np.abs(curr_mask - prev_mask)
            change_ratio = np.mean(diff)

            if change_ratio < threshold:
                # Small change, use previous mask to prevent flicker
                result.append(result[-1])
            else:
                # Significant change, keep new mask
                result.append(masks[i])

        return result
