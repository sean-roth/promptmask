"""
Universal Export
Exports masks as PNG sequences compatible with all video editors.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)


class UniversalExporter:
    """Exports masks in universal PNG sequence format."""

    def __init__(self):
        """Initialize universal exporter."""
        logger.info("UniversalExporter initialized")

    def export(
        self,
        masks: List[np.ndarray],
        output_dir: Path,
        metadata: Dict,
        format_type: str = "black_white",
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Export masks as PNG sequence.

        Args:
            masks: List of binary mask arrays
            output_dir: Directory to save exported files
            metadata: Video metadata (resolution, fps, etc.)
            format_type: Output format ('black_white', 'alpha', 'inverted')
            progress_callback: Optional callback(current, total)

        Returns:
            True if successful
        """
        try:
            # Create output directory structure
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Exporting {len(masks)} masks to {output_dir}")

            # Export mask PNG sequence
            for i, mask in enumerate(masks):
                mask_img = self._convert_mask_to_format(mask, format_type)

                output_path = masks_dir / f"mask_{i+1:06d}.png"
                mask_img.save(output_path)

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(masks))

            # Save metadata
            self._save_metadata(output_dir, metadata, len(masks))

            # Create import guide
            self._create_import_guide(output_dir, metadata)

            logger.info(f"Export complete: {len(masks)} masks saved to {masks_dir}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _convert_mask_to_format(
        self,
        mask: np.ndarray,
        format_type: str
    ) -> Image.Image:
        """
        Convert mask array to PIL Image in specified format.

        Args:
            mask: Binary mask array
            format_type: Output format type

        Returns:
            PIL Image
        """
        if format_type == "black_white":
            # White = selected object, Black = background
            mask_img = (mask * 255).astype(np.uint8)
            return Image.fromarray(mask_img, mode='L')

        elif format_type == "inverted":
            # Black = selected object, White = background
            mask_img = ((1 - mask) * 255).astype(np.uint8)
            return Image.fromarray(mask_img, mode='L')

        elif format_type == "alpha":
            # RGBA with mask as alpha channel
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = 255  # White RGB
            rgba[:, :, 3] = (mask * 255).astype(np.uint8)  # Mask as alpha
            return Image.fromarray(rgba, mode='RGBA')

        else:
            logger.warning(f"Unknown format type '{format_type}', using black_white")
            mask_img = (mask * 255).astype(np.uint8)
            return Image.fromarray(mask_img, mode='L')

    def _save_metadata(
        self,
        output_dir: Path,
        metadata: Dict,
        num_masks: int
    ):
        """
        Save export metadata as JSON.

        Args:
            output_dir: Output directory
            metadata: Video metadata
            num_masks: Number of exported masks
        """
        export_metadata = {
            "promptmask_version": "0.1.0",
            "export_format": "universal_png",
            "video_metadata": {
                "resolution": metadata['resolution'],
                "fps": metadata['fps'],
                "duration": metadata['duration'],
                "total_frames": metadata['total_frames']
            },
            "export_info": {
                "num_masks": num_masks,
                "mask_format": "grayscale_png",
                "white_value": 255,
                "black_value": 0
            }
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(export_metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def _create_import_guide(
        self,
        output_dir: Path,
        metadata: Dict
    ):
        """
        Create a text guide for importing masks into video editors.

        Args:
            output_dir: Output directory
            metadata: Video metadata
        """
        guide_content = f"""# PromptMask - Import Guide (Universal Format)

## Overview

This export contains masks in universal PNG sequence format, compatible with:
- Adobe Premiere Pro
- Adobe After Effects
- DaVinci Resolve
- Final Cut Pro
- Any video editor that supports image sequences

## What's Included

- `masks/` - PNG sequence (mask_{i:06d}.png)
- `metadata.json` - Technical specifications
- `import_guide.txt` - This file

## Video Specifications

- Resolution: {metadata['resolution'][0]}x{metadata['resolution'][1]}
- Frame Rate: {metadata['fps']:.2f} fps
- Total Frames: {metadata['total_frames']}
- Duration: {metadata['duration']:.2f} seconds

## Mask Format

- **White (255)** = Selected object
- **Black (0)** = Background
- Format: 8-bit grayscale PNG
- Naming: Sequential with 6-digit padding (mask_000001.png, mask_000002.png, ...)

## How to Import

### Adobe Premiere Pro

1. **Import Mask Sequence:**
   - File → Import
   - Navigate to `masks/` folder
   - Select first file (mask_000001.png)
   - ✅ Check "Image Sequence"
   - Click "Import"

2. **Apply as Track Matte:**
   - Place your video on V1
   - Place mask sequence on V2
   - Right-click video → Nest
   - Apply Effect: "Track Matte Key"
   - Set Matte: Video 2
   - Composite Using: Matte Alpha

3. **Alternative - Use as Alpha:**
   - Import mask sequence
   - Place above video
   - Set blend mode to "Multiply" or use as luma matte

### Adobe After Effects

1. **Import Sequence:**
   - File → Import → File
   - Select mask_000001.png
   - Choose "PNG Sequence"
   - Click "Import"

2. **Use as Track Matte:**
   - Add video layer to composition
   - Add mask sequence layer above video
   - Set Track Matte: Luma Matte "Mask Sequence"
   - Hide mask layer (click eye icon)

3. **Apply Effects:**
   - Now you can apply effects only to masked area
   - Or replace background, add selective color grading, etc.

### DaVinci Resolve

1. **Import Sequence:**
   - Media Pool → Import Media
   - Select all mask PNG files (Shift+Click or Cmd+A)
   - Right-click → "Create timeline using selected clips"
   - Set to "Image sequence"

2. **Apply as Mask:**
   - Add video to timeline
   - Add mask sequence to track above
   - Use Fusion page to composite
   - Or use Color page for selective grading

### Final Cut Pro

1. **Import:**
   - File → Import → Files
   - Select mask_000001.png
   - Import as image sequence

2. **Apply:**
   - Place video on primary storyline
   - Place mask above as connected clip
   - Use "Draw Mask" effect to apply

## Troubleshooting

**Problem: Sequence imports as separate images**
- Solution: Make sure to select "Image Sequence" option during import
- Check that files are named sequentially with padding

**Problem: Mask not aligned with video**
- Solution: Ensure mask sequence has same resolution and frame rate as video
- Check that mask sequence starts at same frame as video

**Problem: Mask appears inverted**
- Solution: Some editors interpret black/white differently
- Use "Invert" effect to flip the mask

**Problem: Edge artifacts or jagged lines**
- Solution: Apply slight blur to mask in your editor
- Or re-export from PromptMask with higher edge feathering

## Tips

- **Preview First:** Import a few frames to test before full import
- **Match Timeline:** Ensure mask FPS matches your project FPS
- **Backup:** Keep original PromptMask export for re-processing if needed
- **Effects Order:** Apply mask before other effects for best results

## Need Help?

- Documentation: github.com/sean-roth/promptmask
- Report Issues: Use feedback button in PromptMask app
- Video Tutorials: [Coming soon]

---

Generated by PromptMask v0.1.0
https://github.com/sean-roth/promptmask
"""

        guide_path = output_dir / "import_guide.txt"
        with open(guide_path, 'w') as f:
            f.write(guide_content)

        logger.info(f"Import guide saved to {guide_path}")

    def create_contact_sheet(
        self,
        masks: List[np.ndarray],
        output_path: Path,
        grid_size: tuple = (4, 4),
        sample_interval: int = None
    ) -> bool:
        """
        Create a contact sheet preview of masks.

        Args:
            masks: List of mask arrays
            output_path: Path to save contact sheet
            grid_size: (rows, cols) for the grid
            sample_interval: Sample every Nth mask (auto if None)

        Returns:
            True if successful
        """
        try:
            rows, cols = grid_size
            total_slots = rows * cols

            # Determine sampling interval
            if sample_interval is None:
                sample_interval = max(1, len(masks) // total_slots)

            # Sample masks
            sampled_masks = masks[::sample_interval][:total_slots]

            if len(sampled_masks) == 0:
                logger.warning("No masks to create contact sheet")
                return False

            # Get mask dimensions
            h, w = sampled_masks[0].shape

            # Create contact sheet canvas
            canvas_h = h * rows
            canvas_w = w * cols
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

            # Fill canvas with masks
            for i, mask in enumerate(sampled_masks):
                row = i // cols
                col = i % cols

                y_start = row * h
                y_end = y_start + h
                x_start = col * w
                x_end = x_start + w

                canvas[y_start:y_end, x_start:x_end] = (mask * 255).astype(np.uint8)

            # Save contact sheet
            contact_img = Image.fromarray(canvas, mode='L')
            contact_img.save(output_path)

            logger.info(f"Contact sheet saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create contact sheet: {e}")
            return False
