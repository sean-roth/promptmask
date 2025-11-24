"""
Video validation utilities.
Checks file format, size, resolution, and other constraints.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import subprocess
import json

logger = logging.getLogger(__name__)


class VideoValidator:
    """Validates video files before processing."""

    # Supported formats
    SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv']

    # Constraints
    MAX_FILE_SIZE_GB = 5
    MAX_RESOLUTION = (3840, 2160)  # 4K
    MAX_DURATION_MINUTES = 30

    def __init__(self):
        """Initialize validator."""
        logger.info("VideoValidator initialized")

    def validate_file(self, file_path: Path) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate a video file.

        Args:
            file_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message, metadata)
            metadata contains: resolution, fps, duration, codec, format
        """
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist", None

        # Check if it's a file
        if not file_path.is_file():
            return False, "Path is not a file", None

        # Check file extension
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Supported: {', '.join(self.SUPPORTED_FORMATS)}", None

        # Check file size
        file_size_gb = file_path.stat().st_size / (1024 ** 3)
        if file_size_gb > self.MAX_FILE_SIZE_GB:
            return False, f"File too large ({file_size_gb:.1f}GB). Max: {self.MAX_FILE_SIZE_GB}GB", None

        # Extract metadata using ffprobe
        try:
            metadata = self._get_video_metadata(file_path)
        except Exception as e:
            return False, f"Failed to read video metadata: {e}", None

        # Validate resolution
        width, height = metadata['resolution']
        max_w, max_h = self.MAX_RESOLUTION
        if width > max_w or height > max_h:
            return False, f"Resolution too high ({width}x{height}). Max: {max_w}x{max_h}", None

        # Validate duration
        duration_minutes = metadata['duration'] / 60
        if duration_minutes > self.MAX_DURATION_MINUTES:
            return False, f"Video too long ({duration_minutes:.1f}min). Max: {self.MAX_DURATION_MINUTES}min", None

        logger.info(f"Video validated successfully: {metadata}")
        return True, "", metadata

    def _get_video_metadata(self, file_path: Path) -> Dict:
        """
        Extract video metadata using ffprobe.

        Args:
            file_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        try:
            # Use ffprobe to get video info
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break

            if not video_stream:
                raise ValueError("No video stream found in file")

            # Extract metadata
            metadata = {
                'resolution': (int(video_stream['width']), int(video_stream['height'])),
                'fps': self._parse_fps(video_stream.get('r_frame_rate', '30/1')),
                'duration': float(data['format'].get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'format': data['format'].get('format_name', 'unknown'),
                'total_frames': int(video_stream.get('nb_frames', 0))
            }

            # Calculate total_frames if not available
            if metadata['total_frames'] == 0:
                metadata['total_frames'] = int(metadata['duration'] * metadata['fps'])

            return metadata

        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e}")
            raise ValueError("Failed to extract video metadata")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse video metadata: {e}")
            raise ValueError("Invalid video metadata")

    def _parse_fps(self, fps_string: str) -> float:
        """
        Parse FPS from ffprobe format (e.g., '30/1' or '30000/1001').

        Args:
            fps_string: FPS as string fraction

        Returns:
            FPS as float
        """
        try:
            parts = fps_string.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            return float(fps_string)
        except:
            logger.warning(f"Failed to parse FPS '{fps_string}', defaulting to 30")
            return 30.0

    def estimate_processing_time(
        self,
        metadata: Dict,
        fps_processing_speed: float = 3.5
    ) -> float:
        """
        Estimate processing time for a video.

        Args:
            metadata: Video metadata dict
            fps_processing_speed: Processing speed in frames per second

        Returns:
            Estimated time in minutes
        """
        total_frames = metadata['total_frames']
        processing_seconds = total_frames / fps_processing_speed
        return processing_seconds / 60
