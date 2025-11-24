"""
PromptMask - AI-Powered Video Masking
Main application entry point with Gradio UI.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple
import gradio as gr
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('promptmask.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import components
try:
    from sam3.model_loader import SAM3ModelLoader
    from sam3.inference import SAM3Inference
    from pipeline.video_processor import VideoProcessor
    from pipeline.validators import VideoValidator
    from processing.temporal_smoother import TemporalSmoother
    from processing.mask_refiner import MaskRefiner
    from export.universal import UniversalExporter
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    logger.error("Make sure all packages are installed: pip install -r requirements.txt")
    sys.exit(1)


class PromptMaskApp:
    """Main application class."""

    def __init__(self):
        """Initialize the application."""
        logger.info("Initializing PromptMask...")

        # Initialize components
        self.model_loader = None
        self.inference_engine = None
        self.video_processor = VideoProcessor()
        self.temporal_smoother = TemporalSmoother()
        self.mask_refiner = MaskRefiner()
        self.exporter = UniversalExporter()

        # State
        self.current_masks = None
        self.current_metadata = None
        self.current_frames_dir = None

        logger.info("PromptMask initialized successfully")

    def load_sam3_model(self) -> str:
        """
        Load SAM 3 model into memory.

        Returns:
            Status message
        """
        try:
            logger.info("Loading SAM 3 model...")
            self.model_loader = SAM3ModelLoader()

            success = self.model_loader.load_model()

            if success:
                self.inference_engine = SAM3Inference(self.model_loader)
                return f"✅ Model loaded successfully on {self.model_loader.get_device()}"
            else:
                return "❌ Failed to load model. Check logs for details."

        except Exception as e:
            error_msg = f"❌ Error loading model: {e}"
            logger.error(error_msg)
            return error_msg

    def validate_video(self, video_file) -> Tuple[str, str]:
        """
        Validate uploaded video.

        Args:
            video_file: Gradio file upload object

        Returns:
            Tuple of (status_message, metadata_display)
        """
        if video_file is None:
            return "❌ No video uploaded", ""

        try:
            video_path = Path(video_file.name)
            is_valid, error_msg, metadata = self.video_processor.load_video(video_path)

            if not is_valid:
                return f"❌ {error_msg}", ""

            # Store metadata
            self.current_metadata = metadata

            # Format metadata display
            validator = VideoValidator()
            est_time = validator.estimate_processing_time(metadata)

            metadata_str = f"""
**Video Information:**
- Resolution: {metadata['resolution'][0]}x{metadata['resolution'][1]}
- Frame Rate: {metadata['fps']:.2f} fps
- Duration: {metadata['duration']:.1f} seconds
- Total Frames: {metadata['total_frames']}
- Codec: {metadata['codec']}

**Estimated Processing Time:** ~{est_time:.0f} minutes
"""

            return "✅ Video validated successfully", metadata_str

        except Exception as e:
            error_msg = f"❌ Validation error: {e}"
            logger.error(error_msg)
            return error_msg, ""

    def process_video(
        self,
        video_file,
        text_prompt: str,
        quality: str,
        feather_radius: int,
        temporal_smoothing: bool,
        confidence_threshold: float,
        progress=gr.Progress()
    ) -> Tuple[str, Optional[str]]:
        """
        Process video with SAM 3 and generate masks.

        Args:
            video_file: Gradio file upload object
            text_prompt: Natural language prompt
            quality: Quality setting ('fast', 'medium', 'accurate')
            feather_radius: Edge feathering radius
            temporal_smoothing: Enable temporal smoothing
            confidence_threshold: Minimum confidence score
            progress: Gradio progress tracker

        Returns:
            Tuple of (status_message, preview_video_path)
        """
        try:
            # Validate inputs
            if video_file is None:
                return "❌ No video uploaded", None

            if not text_prompt or text_prompt.strip() == "":
                return "❌ Please enter a text prompt", None

            if self.inference_engine is None:
                return "❌ Model not loaded. Please wait for model to load.", None

            progress(0, desc="Extracting frames...")

            # Extract frames
            success, error_msg, frames_dir = self.video_processor.extract_frames(
                quality=quality,
                progress_callback=lambda curr, total: progress(
                    curr / total * 0.1,
                    desc=f"Extracting frames ({curr}/{total})..."
                )
            )

            if not success:
                return f"❌ Frame extraction failed: {error_msg}", None

            self.current_frames_dir = frames_dir

            progress(0.1, desc="Loading frames into memory...")

            # Load frames
            frames = self.video_processor.load_frames(frames_dir)

            if len(frames) == 0:
                return "❌ No frames extracted", None

            progress(0.15, desc=f"Processing {len(frames)} frames with SAM 3...")

            # Run inference on all frames
            masks, scores = self.inference_engine.segment_frame_sequence(
                frames=frames,
                text_prompt=text_prompt,
                confidence_threshold=confidence_threshold / 100.0,
                use_temporal_tracking=True,
                progress_callback=lambda curr, total: progress(
                    0.15 + (curr / total * 0.6),
                    desc=f"Processing frame {curr}/{total} (avg conf: {np.mean(scores[:curr]) if scores else 0:.0%})..."
                )
            )

            self.current_masks = masks

            # Apply temporal smoothing
            if temporal_smoothing:
                progress(0.75, desc="Applying temporal smoothing...")
                masks = self.temporal_smoother.smooth_masks(
                    masks,
                    strength=0.5,
                    progress_callback=lambda curr, total: progress(
                        0.75 + (curr / total * 0.05),
                        desc=f"Smoothing {curr}/{total}..."
                    )
                )

            progress(0.80, desc="Refining masks...")

            # Refine masks
            masks = self.mask_refiner.refine_masks_batch(
                masks,
                feather_radius=feather_radius,
                fill_holes=True,
                smooth_edges=True,
                progress_callback=lambda curr, total: progress(
                    0.80 + (curr / total * 0.05),
                    desc=f"Refining {curr}/{total}..."
                )
            )

            self.current_masks = masks

            progress(0.85, desc="Saving processed masks...")

            # Save masks to temp directory
            masks_dir = self.video_processor.temp_dir / "processed_masks"
            self.video_processor.save_masks(masks, masks_dir)

            progress(0.90, desc="Creating preview video...")

            # Create preview video
            preview_path = self.video_processor.temp_dir / "preview.mp4"
            preview_success = self.video_processor.create_preview_video(
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_path=preview_path,
                show_overlay=True
            )

            progress(1.0, desc="Complete!")

            avg_confidence = np.mean(scores) * 100

            if preview_success:
                status = f"""✅ Processing complete!

**Results:**
- Frames processed: {len(masks)}
- Average confidence: {avg_confidence:.1f}%
- Preview generated: {preview_path}

Ready to export!"""
                return status, str(preview_path)
            else:
                return f"✅ Processing complete (avg confidence: {avg_confidence:.1f}%). Preview generation failed.", None

        except Exception as e:
            error_msg = f"❌ Processing error: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg, None

    def export_masks(
        self,
        export_format: str,
        output_dir: str,
        progress=gr.Progress()
    ) -> str:
        """
        Export processed masks.

        Args:
            export_format: Export format ('universal', 'after_effects', 'premiere')
            output_dir: Output directory path
            progress: Gradio progress tracker

        Returns:
            Status message
        """
        try:
            if self.current_masks is None:
                return "❌ No masks to export. Process a video first."

            if not output_dir:
                output_dir = str(Path.home() / "Desktop" / "promptmask_export")

            output_path = Path(output_dir)

            progress(0, desc="Exporting masks...")

            # Export based on format
            if export_format == "Universal (PNG Sequence)":
                success = self.exporter.export(
                    masks=self.current_masks,
                    output_dir=output_path,
                    metadata=self.current_metadata,
                    format_type="black_white",
                    progress_callback=lambda curr, total: progress(
                        curr / total,
                        desc=f"Exporting {curr}/{total}..."
                    )
                )
            else:
                # TODO: Implement After Effects and Premiere exports in v0.2.0
                return f"❌ {export_format} export not yet implemented (coming in v0.2.0)"

            progress(1.0, desc="Export complete!")

            if success:
                return f"""✅ Export successful!

**Location:** {output_path}

**Contents:**
- masks/ - PNG sequence
- metadata.json - Technical specs
- import_guide.txt - Import instructions

See import_guide.txt for instructions on importing into your video editor."""
            else:
                return "❌ Export failed. Check logs for details."

        except Exception as e:
            error_msg = f"❌ Export error: {e}"
            logger.error(error_msg)
            return error_msg

    def create_ui(self) -> gr.Blocks:
        """
        Create Gradio UI.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="PromptMask - AI Video Masking", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# PromptMask")
            gr.Markdown("AI-powered video masking using natural language prompts")

            # Model loading status
            with gr.Row():
                model_status = gr.Textbox(
                    label="Model Status",
                    value="⏳ Loading SAM 3 model... (this may take a few minutes on first run)",
                    interactive=False
                )

            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Video")

                    video_input = gr.Video(
                        label="Video File",
                        sources=["upload"]
                    )

                    video_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Upload a video to begin..."
                    )

                    video_metadata = gr.Markdown("")

                    gr.Markdown("### 2. Describe What to Segment")

                    text_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="e.g., 'person speaking', 'product on desk', 'background trees'",
                        lines=2
                    )

                    # Preset buttons
                    with gr.Row():
                        preset_speaker = gr.Button("👤 Speaker Isolation", size="sm")
                        preset_product = gr.Button("📦 Product Demo", size="sm")

                    gr.Markdown("### 3. Processing Settings")

                    quality = gr.Radio(
                        label="Quality",
                        choices=["fast", "medium", "accurate"],
                        value="medium"
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        feather_radius = gr.Slider(
                            label="Edge Feathering",
                            minimum=0,
                            maximum=20,
                            value=10,
                            step=1
                        )

                        temporal_smoothing = gr.Checkbox(
                            label="Temporal Smoothing",
                            value=True
                        )

                        confidence_threshold = gr.Slider(
                            label="Confidence Threshold (%)",
                            minimum=0,
                            maximum=100,
                            value=70,
                            step=5
                        )

                    process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")

                # Center column - Preview
                with gr.Column(scale=1):
                    gr.Markdown("### Preview")

                    process_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        placeholder="Results will appear here..."
                    )

                    preview_video = gr.Video(
                        label="Preview (with mask overlay)",
                        interactive=False
                    )

                # Right column - Export
                with gr.Column(scale=1):
                    gr.Markdown("### Export")

                    export_format = gr.Radio(
                        label="Export Format",
                        choices=[
                            "Universal (PNG Sequence)",
                            "After Effects (v0.2.0)",
                            "Premiere Pro (v0.2.0)"
                        ],
                        value="Universal (PNG Sequence)"
                    )

                    output_dir = gr.Textbox(
                        label="Output Directory",
                        placeholder="Leave empty for Desktop/promptmask_export",
                        lines=1
                    )

                    export_btn = gr.Button("💾 Export Masks", variant="secondary", size="lg")

                    export_status = gr.Textbox(
                        label="Export Status",
                        interactive=False,
                        placeholder="Export results will appear here..."
                    )

            # Event handlers
            video_input.change(
                fn=self.validate_video,
                inputs=[video_input],
                outputs=[video_status, video_metadata]
            )

            preset_speaker.click(
                fn=lambda: "person speaking",
                outputs=[text_prompt]
            )

            preset_product.click(
                fn=lambda: "product on desk",
                outputs=[text_prompt]
            )

            process_btn.click(
                fn=self.process_video,
                inputs=[
                    video_input,
                    text_prompt,
                    quality,
                    feather_radius,
                    temporal_smoothing,
                    confidence_threshold
                ],
                outputs=[process_status, preview_video]
            )

            export_btn.click(
                fn=self.export_masks,
                inputs=[export_format, output_dir],
                outputs=[export_status]
            )

            # Load model on startup
            interface.load(
                fn=self.load_sam3_model,
                outputs=[model_status]
            )

            gr.Markdown("---")
            gr.Markdown("**PromptMask v0.1.0** | [GitHub](https://github.com/sean-roth/promptmask) | [Report Issue](https://github.com/sean-roth/promptmask/issues)")

        return interface


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Starting PromptMask v0.1.0")
    logger.info("=" * 60)

    # Create app
    app = PromptMaskApp()

    # Create and launch UI
    interface = app.create_ui()

    logger.info("Launching Gradio interface...")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
