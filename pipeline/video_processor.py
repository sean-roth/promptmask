"""
Video Processing Pipeline
Handles video upload, frame extraction, and coordination with SAM 3 inference.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import tempfile
import shutil
import subprocess
from PIL import Image
import numpy as np

from pipeline.validators import VideoValidator

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Manages video processing pipeline."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize video processor.

        Args:
            temp_dir: Directory for temporary files. Uses system temp if None.
        """
        self.validator = VideoValidator()
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "promptmask"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.current_video_path = None
        self.current_frames_dir = None
        self.current_metadata = None

        logger.info(f"VideoProcessor initialized with temp dir: {self.temp_dir}")

    def load_video(self, video_path: Path) -> Tuple[bool, str, Optional[Dict]]:
        """
        Load and validate a video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (success, error_message, metadata)
        """
        # Validate video
        is_valid, error_msg, metadata = self.validator.validate_file(video_path)

        if not is_valid:
            logger.error(f"Video validation failed: {error_msg}")
            return False, error_msg, None

        self.current_video_path = video_path
        self.current_metadata = metadata

        logger.info(f"Video loaded: {video_path}")
        return True, "", metadata

    def extract_frames(
        self,
        quality: str = "medium",
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Extract frames from the loaded video to a temporary directory.

        Args:
            quality: Quality setting ('fast', 'medium', 'accurate')
            progress_callback: Optional callback(current, total)

        Returns:
            Tuple of (success, error_message, frames_directory)
        """
        if self.current_video_path is None:
            return False, "No video loaded", None

        # Create temporary directory for frames
        video_id = self.current_video_path.stem
        self.current_frames_dir = self.temp_dir / f"{video_id}_frames"

        # Clean up existing frames
        if self.current_frames_dir.exists():
            shutil.rmtree(self.current_frames_dir)

        self.current_frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract frames using ffmpeg
            output_pattern = str(self.current_frames_dir / "frame_%06d.png")

            # Quality settings affect scale
            scale_filter = self._get_scale_filter(quality)

            cmd = [
                'ffmpeg',
                '-i', str(self.current_video_path),
                '-vf', scale_filter,
                '-q:v', '2',  # High quality
                output_pattern
            ]

            logger.info(f"Extracting frames with command: {' '.join(cmd)}")

            # Run ffmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Monitor progress
            # Note: ffmpeg outputs to stderr
            for line in process.stderr:
                if progress_callback and 'frame=' in line:
                    # Parse frame number from output
                    try:
                        frame_num = int(line.split('frame=')[1].split()[0])
                        total_frames = self.current_metadata['total_frames']
                        progress_callback(frame_num, total_frames)
                    except:
                        pass

            process.wait()

            if process.returncode != 0:
                error_msg = "ffmpeg extraction failed"
                logger.error(error_msg)
                return False, error_msg, None

            # Count extracted frames
            frame_files = list(self.current_frames_dir.glob("frame_*.png"))
            logger.info(f"Extracted {len(frame_files)} frames to {self.current_frames_dir}")

            return True, "", self.current_frames_dir

        except Exception as e:
            error_msg = f"Frame extraction failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None

    def _get_scale_filter(self, quality: str) -> str:
        """
        Get ffmpeg scale filter based on quality setting.

        Args:
            quality: Quality setting ('fast', 'medium', 'accurate')

        Returns:
            ffmpeg scale filter string
        """
        if quality == "fast":
            # Scale down to 720p for faster processing
            return "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease"
        elif quality == "accurate":
            # Keep original resolution
            return "scale=iw:ih"
        else:  # medium (default)
            # Scale to 1080p
            return "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease"

    def load_frames(self, frames_dir: Optional[Path] = None) -> List[Image.Image]:
        """
        Load extracted frames into memory as PIL Images.

        Args:
            frames_dir: Directory containing frames. Uses current if None.

        Returns:
            List of PIL Images
        """
        if frames_dir is None:
            frames_dir = self.current_frames_dir

        if frames_dir is None or not frames_dir.exists():
            logger.error("No frames directory available")
            return []

        # Get all frame files sorted by name
        frame_files = sorted(frames_dir.glob("frame_*.png"))

        logger.info(f"Loading {len(frame_files)} frames into memory...")

        frames = []
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file)
                frames.append(img)
            except Exception as e:
                logger.error(f"Failed to load frame {frame_file}: {e}")

        logger.info(f"Loaded {len(frames)} frames")
        return frames

    def save_masks(
        self,
        masks: List[np.ndarray],
        output_dir: Path
    ) -> bool:
        """
        Save masks as PNG files.

        Args:
            masks: List of mask arrays
            output_dir: Directory to save masks

        Returns:
            True if successful
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving {len(masks)} masks to {output_dir}")

            for i, mask in enumerate(masks):
                # Convert to 0-255 range
                mask_img = (mask * 255).astype(np.uint8)
                img = Image.fromarray(mask_img, mode='L')

                output_path = output_dir / f"mask_{i+1:06d}.png"
                img.save(output_path)

            logger.info(f"Saved {len(masks)} masks")
            return True

        except Exception as e:
            logger.error(f"Failed to save masks: {e}")
            return False

    def create_preview_video(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_path: Path,
        show_overlay: bool = True
    ) -> bool:
        """
        Create a preview video with mask overlay.

        Args:
            frames_dir: Directory with original frames
            masks_dir: Directory with mask frames
            output_path: Path for output video
            show_overlay: If True, overlay masks on frames

        Returns:
            True if successful
        """
        try:
            if show_overlay:
                # Create overlay frames
                overlay_dir = self.temp_dir / "overlay_frames"
                overlay_dir.mkdir(parents=True, exist_ok=True)

                frame_files = sorted(frames_dir.glob("frame_*.png"))
                mask_files = sorted(masks_dir.glob("mask_*.png"))

                for frame_file, mask_file in zip(frame_files, mask_files):
                    frame = Image.open(frame_file).convert('RGB')
                    mask = Image.open(mask_file).convert('L')

                    # Create green overlay
                    overlay = self._create_overlay(frame, mask)

                    output_file = overlay_dir / frame_file.name
                    overlay.save(output_file)

                input_dir = overlay_dir
            else:
                input_dir = frames_dir

            # Create video with ffmpeg
            input_pattern = str(input_dir / "frame_%06d.png")
            fps = self.current_metadata['fps']

            cmd = [
                'ffmpeg',
                '-framerate', str(fps),
                '-i', input_pattern,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-y',  # Overwrite output
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"ffmpeg preview creation failed: {result.stderr}")
                return False

            logger.info(f"Preview video created: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create preview video: {e}")
            return False

    def _create_overlay(
        self,
        frame: Image.Image,
        mask: Image.Image,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.4
    ) -> Image.Image:
        """Create an overlay of mask on frame."""
        frame_array = np.array(frame)
        mask_array = np.array(mask) / 255.0

        # Create colored mask
        colored_mask = np.zeros_like(frame_array)
        for i in range(3):
            colored_mask[:, :, i] = mask_array * color[i]

        # Blend
        mask_3d = np.stack([mask_array] * 3, axis=-1)
        blended = frame_array * (1 - mask_3d * alpha) + colored_mask * alpha
        blended = blended.astype(np.uint8)

        return Image.fromarray(blended)

    def cleanup(self):
        """Clean up temporary files."""
        if self.current_frames_dir and self.current_frames_dir.exists():
            logger.info(f"Cleaning up temporary frames: {self.current_frames_dir}")
            shutil.rmtree(self.current_frames_dir)
            self.current_frames_dir = None

        self.current_video_path = None
        self.current_metadata = None
