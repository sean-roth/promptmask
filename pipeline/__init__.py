"""
Video Processing Pipeline Module

Handles video input, frame extraction, and validation.
"""

from .video_processor import VideoProcessor
from .validators import VideoValidator

__all__ = ['VideoProcessor', 'VideoValidator']
