"""
Video Frame Processor

Handles video file loading, frame extraction, and basic preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import logging

from .validators import VideoValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files and extracts frames for segmentation.

    Features:
    - Frame extraction at specified FPS
    - Frame validation and filtering
    - Memory-efficient processing
    - Video metadata extraction
    """

    def __init__(self, max_frames: int = 300):
        """
        Initialize video processor.

        Args:
            max_frames: Maximum number of frames to extract (default: 300)
        """
        self.max_frames = max_frames
        self.validator = VideoValidator()

    def extract_frames(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Tuple[List[Image.Image], Dict]:
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file
            target_fps: Target FPS for extraction (None = extract all frames)
            max_frames: Maximum frames to extract (overrides self.max_frames)

        Returns:
            Tuple of (frames, metadata)
            - frames: List of PIL Images
            - metadata: Dict with video info (fps, duration, dimensions, etc.)

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened or is invalid
        """
        # Validate video file
        self.validator.validate_video_file(video_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Get video metadata
            metadata = self._extract_metadata(cap)
            logger.info(f"Video metadata: {metadata}")

            # Calculate frame sampling
            original_fps = metadata['fps']
            total_frames = metadata['frame_count']

            if target_fps is None:
                target_fps = original_fps
                frame_interval = 1
            else:
                frame_interval = max(1, int(original_fps / target_fps))

            # Determine max frames
            if max_frames is None:
                max_frames = self.max_frames

            # Extract frames
            frames = []
            frame_idx = 0
            extracted_count = 0

            logger.info(f"Extracting frames (interval: {frame_interval}, max: {max_frames})...")

            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at interval
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)

                    frames.append(pil_image)
                    extracted_count += 1

                    if extracted_count % 30 == 0:
                        logger.info(f"Extracted {extracted_count} frames...")

                frame_idx += 1

            logger.info(f"Extracted {len(frames)} frames from video")

            # Update metadata with extraction info
            metadata['extracted_frames'] = len(frames)
            metadata['target_fps'] = target_fps
            metadata['frame_interval'] = frame_interval

            return frames, metadata

        finally:
            cap.release()

    def _extract_metadata(self, cap: cv2.VideoCapture) -> Dict:
        """
        Extract metadata from video capture object.

        Args:
            cap: OpenCV VideoCapture object

        Returns:
            Dict containing video metadata
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }

    def validate_frame_sequence(self, frames: List[Image.Image]) -> bool:
        """
        Validate that frame sequence is suitable for processing.

        Args:
            frames: List of PIL Images

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not frames:
            raise ValueError("Frame sequence is empty")

        # Check all frames have same dimensions
        first_size = frames[0].size
        for i, frame in enumerate(frames[1:], 1):
            if frame.size != first_size:
                raise ValueError(
                    f"Frame {i} has different dimensions: {frame.size} vs {first_size}"
                )

        logger.info(f"Validated {len(frames)} frames at {first_size}")
        return True

    def resize_frames(
        self,
        frames: List[Image.Image],
        target_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> List[Image.Image]:
        """
        Resize all frames to target size.

        Args:
            frames: List of PIL Images
            target_size: (width, height) tuple
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            List of resized PIL Images
        """
        resized_frames = []

        for frame in frames:
            if maintain_aspect:
                # Calculate new size maintaining aspect ratio
                frame.thumbnail(target_size, Image.Resampling.LANCZOS)
                resized_frames.append(frame)
            else:
                # Resize to exact dimensions
                resized = frame.resize(target_size, Image.Resampling.LANCZOS)
                resized_frames.append(resized)

        logger.info(f"Resized {len(frames)} frames to {target_size}")
        return resized_frames

    def save_frame(self, frame: Image.Image, output_path: str):
        """
        Save a single frame to disk.

        Args:
            frame: PIL Image to save
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.save(output_path, quality=95)

    def frames_to_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """
        Combine frames back into a video file.

        Args:
            frames: List of PIL Images
            output_path: Output video file path
            fps: Output video FPS
            codec: Video codec to use
        """
        if not frames:
            raise ValueError("No frames to save")

        # Get dimensions from first frame
        width, height = frames[0].size

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for frame in frames:
                # Convert PIL to numpy array
                frame_np = np.array(frame)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                out.write(frame_bgr)

            logger.info(f"Saved video to {output_path}")

        finally:
            out.release()
