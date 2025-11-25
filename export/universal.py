"""
Universal PNG Exporter

Exports segmentation masks as production-ready PNG sequences with
support for various formats and compositing options.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalExporter:
    """
    Exports masks to universal PNG format and other formats.

    Features:
    - PNG sequence export
    - Grayscale matte export (Adobe-compatible)
    - Alpha channel export (optional)
    - ZIP archive creation
    - Metadata preservation
    """

    def __init__(self):
        """Initialize universal exporter."""
        pass

    def export(
        self,
        masks: List[np.ndarray],
        output_path: str,
        format_type: str = 'universal_png',
        video_metadata: Optional[Dict] = None,
        create_archive: bool = True,
        include_alpha: bool = False  # Changed default to False for Adobe compatibility
    ) -> str:
        """
        Export masks to specified format.

        Args:
            masks: List of binary masks (numpy arrays, 0/255)
            output_path: Output directory or file path
            format_type: Export format ('universal_png', 'png_sequence', 'video')
            video_metadata: Optional video metadata dict
            create_archive: Whether to create ZIP archive
            include_alpha: If True, export RGBA with alpha. If False, export grayscale matte.

        Returns:
            Path to exported file or directory
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format_type == 'universal_png':
            return self._export_universal_png(
                masks,
                output_dir,
                video_metadata,
                create_archive,
                include_alpha
            )
        elif format_type == 'png_sequence':
            return self._export_png_sequence(masks, output_dir)
        elif format_type == 'video':
            return self._export_video(masks, output_dir, video_metadata)
        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _export_universal_png(
        self,
        masks: List[np.ndarray],
        output_dir: Path,
        metadata: Optional[Dict],
        create_archive: bool,
        include_alpha: bool
    ) -> str:
        """
        Export as universal PNG sequence.

        Args:
            masks: List of masks
            output_dir: Output directory
            metadata: Video metadata
            create_archive: Whether to create ZIP
            include_alpha: Whether to include alpha channel (False = grayscale matte)

        Returns:
            Path to output (directory or ZIP file)
        """
        logger.info(f"Exporting {len(masks)} masks as universal PNG (grayscale matte)...")

        # Create frames directory
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Export each mask
        for i, mask in enumerate(masks):
            frame_path = frames_dir / f"frame_{i:05d}.png"

            if include_alpha:
                # Export as RGBA with alpha channel
                self._save_mask_with_alpha(mask, frame_path)
            else:
                # Export as grayscale matte (Adobe-compatible)
                self._save_mask_simple(mask, frame_path)

        # Save metadata if provided
        if metadata:
            self._save_metadata(metadata, frames_dir / "metadata.json")

        logger.info(f"Exported {len(masks)} frames to {frames_dir}")

        # Create ZIP archive if requested
        if create_archive:
            zip_path = output_dir / "masks.zip"
            self._create_archive(frames_dir, zip_path)
            logger.info(f"Created archive: {zip_path}")
            return str(zip_path)
        else:
            return str(frames_dir)

    def _export_png_sequence(
        self,
        masks: List[np.ndarray],
        output_dir: Path
    ) -> str:
        """
        Export as simple PNG sequence.

        Args:
            masks: List of masks
            output_dir: Output directory

        Returns:
            Path to output directory
        """
        logger.info(f"Exporting {len(masks)} masks as PNG sequence...")

        for i, mask in enumerate(masks):
            frame_path = output_dir / f"mask_{i:05d}.png"
            self._save_mask_simple(mask, frame_path)

        logger.info(f"Exported {len(masks)} frames to {output_dir}")
        return str(output_dir)

    def _export_video(
        self,
        masks: List[np.ndarray],
        output_dir: Path,
        metadata: Optional[Dict]
    ) -> str:
        """
        Export as video file.

        Args:
            masks: List of masks
            output_dir: Output directory
            metadata: Video metadata

        Returns:
            Path to output video file
        """
        import cv2

        logger.info(f"Exporting {len(masks)} masks as video...")

        # Get FPS from metadata or use default
        fps = metadata.get('fps', 30.0) if metadata else 30.0

        # Get dimensions from first mask
        height, width = masks[0].shape[:2]

        # Create video writer
        output_path = output_dir / "masks.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)

        try:
            for mask in masks:
                writer.write(mask)

            logger.info(f"Exported video to {output_path}")
            return str(output_path)

        finally:
            writer.release()

    def _save_mask_simple(self, mask: np.ndarray, path: Path):
        """
        Save mask as simple grayscale PNG (Adobe-compatible matte).

        White (255) = subject/foreground
        Black (0) = background/transparent area

        Args:
            mask: Binary mask (0/255)
            path: Output file path
        """
        # Ensure mask is uint8 with 0/255 values
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Ensure values are 0 or 255
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        img = Image.fromarray(mask, mode='L')
        img.save(path, 'PNG', optimize=True)

    def _save_mask_with_alpha(self, mask: np.ndarray, path: Path):
        """
        Save mask as PNG with alpha channel.

        Creates white image where mask alpha determines transparency.

        Args:
            mask: Binary mask (0/255)
            path: Output file path
        """
        # Ensure mask is proper format
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
            
        height, width = mask.shape[:2]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Set RGB to white
        rgba[:, :, 0:3] = 255

        # Set alpha to mask
        rgba[:, :, 3] = mask

        img = Image.fromarray(rgba, mode='RGBA')
        img.save(path, 'PNG', optimize=True)

    def _save_metadata(self, metadata: Dict, path: Path):
        """
        Save metadata to JSON file.

        Args:
            metadata: Metadata dictionary
            path: Output file path
        """
        import json

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _create_archive(self, source_dir: Path, archive_path: Path):
        """
        Create ZIP archive from directory.

        Args:
            source_dir: Source directory to archive
            archive_path: Output ZIP file path
        """
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zipf.write(file_path, arcname)

    def export_preview(
        self,
        mask: np.ndarray,
        original_image: np.ndarray,
        output_path: str,
        overlay_alpha: float = 0.5,
        overlay_color: tuple = (255, 0, 0)
    ) -> str:
        """
        Export preview image with mask overlay.

        Args:
            mask: Binary mask (0/255)
            original_image: Original RGB image
            output_path: Output file path
            overlay_alpha: Overlay transparency (0-1)
            overlay_color: RGB color for overlay

        Returns:
            Path to exported preview
        """
        # Create colored overlay
        overlay = np.zeros_like(original_image)
        overlay[:, :] = overlay_color

        # Normalize mask to 0-1
        mask_normalized = mask.astype(np.float32) / 255.0

        # Expand to 3 channels
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Blend: original where mask is 0, overlay where mask is 1
        blended = (
            original_image * (1 - mask_3ch * overlay_alpha) +
            overlay * mask_3ch * overlay_alpha
        ).astype(np.uint8)

        # Save
        img = Image.fromarray(blended)
        img.save(output_path, 'PNG', quality=95)

        logger.info(f"Saved preview to {output_path}")
        return output_path

    def export_side_by_side(
        self,
        masks: List[np.ndarray],
        original_frames: List[np.ndarray],
        output_path: str,
        num_samples: int = 6
    ) -> str:
        """
        Export side-by-side comparison of original and masked frames.

        Args:
            masks: List of masks
            original_frames: List of original frames
            output_path: Output file path
            num_samples: Number of sample frames to include

        Returns:
            Path to exported comparison
        """
        if len(masks) != len(original_frames):
            raise ValueError("Number of masks must match number of frames")

        # Sample frames evenly
        indices = np.linspace(0, len(masks) - 1, num_samples, dtype=int)

        # Create grid
        rows = []
        for idx in indices:
            mask = masks[idx]
            frame = original_frames[idx]

            # Convert mask to RGB (white where mask is set)
            mask_rgb = np.stack([mask] * 3, axis=-1)

            # Concatenate horizontally
            row = np.hstack([frame, mask_rgb])
            rows.append(row)

        # Stack vertically
        grid = np.vstack(rows)

        # Save
        img = Image.fromarray(grid)
        img.save(output_path, 'PNG', quality=95)

        logger.info(f"Saved comparison to {output_path}")
        return output_path
