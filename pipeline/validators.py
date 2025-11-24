"""
Input Validators

Validates video files, parameters, and inputs for the processing pipeline.
"""

from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoValidator:
    """
    Validates video files and processing parameters.
    """

    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    MAX_FILE_SIZE_MB = 500  # Maximum file size in MB

    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate that video file exists and is supported.

        Args:
            video_path: Path to video file

        Returns:
            True if valid

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or file is too large
        """
        path = Path(video_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValueError(f"Path is not a file: {video_path}")

        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {path.suffix}\n"
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Video file too large: {file_size_mb:.1f}MB "
                f"(max: {self.MAX_FILE_SIZE_MB}MB)"
            )

        logger.info(f"Video file validated: {path.name} ({file_size_mb:.1f}MB)")
        return True

    def validate_text_prompt(self, prompt: str) -> bool:
        """
        Validate text prompt for segmentation.

        Args:
            prompt: Text prompt string

        Returns:
            True if valid

        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Text prompt cannot be empty")

        if len(prompt) < 2:
            raise ValueError("Text prompt too short (minimum 2 characters)")

        if len(prompt) > 200:
            raise ValueError("Text prompt too long (maximum 200 characters)")

        return True

    def validate_confidence_threshold(self, threshold: float) -> bool:
        """
        Validate confidence threshold value.

        Args:
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            True if valid

        Raises:
            ValueError: If threshold is out of range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {threshold}"
            )

        return True

    def validate_fps(self, fps: float) -> bool:
        """
        Validate FPS value.

        Args:
            fps: Frames per second

        Returns:
            True if valid

        Raises:
            ValueError: If FPS is invalid
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        if fps > 120:
            raise ValueError(f"FPS too high (max 120), got {fps}")

        return True

    def validate_feather_radius(self, radius: int) -> bool:
        """
        Validate feather radius value.

        Args:
            radius: Feather radius in pixels

        Returns:
            True if valid

        Raises:
            ValueError: If radius is invalid
        """
        if radius < 0:
            raise ValueError(f"Feather radius must be non-negative, got {radius}")

        if radius > 50:
            raise ValueError(f"Feather radius too large (max 50), got {radius}")

        return True

    def validate_temporal_window(self, window: int) -> bool:
        """
        Validate temporal smoothing window size.

        Args:
            window: Window size (number of frames)

        Returns:
            True if valid

        Raises:
            ValueError: If window size is invalid
        """
        if window < 1:
            raise ValueError(f"Temporal window must be at least 1, got {window}")

        if window > 10:
            raise ValueError(f"Temporal window too large (max 10), got {window}")

        if window % 2 == 0:
            raise ValueError(f"Temporal window must be odd, got {window}")

        return True


class ParameterValidator:
    """
    Validates processing parameters and presets.
    """

    VALID_PRESETS = ['speaker_isolation', 'product_demo', 'custom']
    VALID_EXPORT_FORMATS = ['universal_png', 'png_sequence', 'video']

    def validate_preset(self, preset: str) -> bool:
        """
        Validate quality preset.

        Args:
            preset: Preset name

        Returns:
            True if valid

        Raises:
            ValueError: If preset is invalid
        """
        if preset not in self.VALID_PRESETS:
            raise ValueError(
                f"Invalid preset: {preset}\n"
                f"Valid presets: {', '.join(self.VALID_PRESETS)}"
            )

        return True

    def validate_export_format(self, format_name: str) -> bool:
        """
        Validate export format.

        Args:
            format_name: Export format name

        Returns:
            True if valid

        Raises:
            ValueError: If format is invalid
        """
        if format_name not in self.VALID_EXPORT_FORMATS:
            raise ValueError(
                f"Invalid export format: {format_name}\n"
                f"Valid formats: {', '.join(self.VALID_EXPORT_FORMATS)}"
            )

        return True
