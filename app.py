"""
PromptMask - Video Segmentation Application

Main Gradio application integrating SAM 3 model with full video processing pipeline.

NOTE: SAM 3 was released Nov 19, 2025 and requires:
- transformers installed from main branch: pip install git+https://github.com/huggingface/transformers.git
"""

import os
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import logging
from pathlib import Path

# Try to load .env for local development, but don't require it
# (HuggingFace Spaces injects secrets as environment variables directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sam3.model_loader import SAM3ModelLoader
from sam3.inference import SAM3Inference
from pipeline.video_processor import VideoProcessor
from pipeline.validators import VideoValidator, ParameterValidator
from processing.temporal_smoother import TemporalSmoother
from processing.mask_refiner import MaskRefiner
from export.universal import UniversalExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptMaskApp:
    """
    Main application class integrating all components.

    Combines:
    - SAM 3 model loading and inference
    - Full video processing pipeline
    - Temporal smoothing and mask refinement
    - Universal PNG export
    """

    def __init__(self):
        """Initialize application components."""
        logger.info("Initializing PromptMask application...")

        # Initialize components
        self.model_loader = SAM3ModelLoader()
        self.inference_engine = None
        self.video_processor = VideoProcessor(max_frames=300)
        self.temporal_smoother = TemporalSmoother(window_size=3)
        self.mask_refiner = MaskRefiner()
        self.exporter = UniversalExporter()

        # Validators
        self.video_validator = VideoValidator()
        self.param_validator = ParameterValidator()

        # State
        self.model_loaded = False
        self.current_frames = None
        self.current_masks = None

        logger.info("Application initialized successfully")

    def load_model(self) -> str:
        """
        Load model from HuggingFace.

        Returns:
            Status message
        """
        try:
            # Check if token exists (works for both .env and HF Spaces secrets)
            token = os.getenv('HUGGINGFACE_TOKEN')
            if not token:
                return (
                    "❌ Configuration error. Please contact support."
                )
            
            # Validate token format
            if not token.startswith('hf_'):
                return (
                    "❌ Configuration error. Please contact support."
                )
            
            logger.info("Loading model...")
            model, processor = self.model_loader.load_model()
            self.inference_engine = SAM3Inference(self.model_loader)
            self.model_loaded = True

            return f"✅ Model loaded successfully"

        except ValueError as e:
            logger.error(f"Auth error: {e}")
            return "❌ Model loading failed. Please contact support."
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return "❌ Model loading failed. Please contact support."
        except Exception as e:
            logger.error(f"Error: {e}")
            return "❌ Model loading failed. Please contact support."

    def process_video(
        self,
        video_file: str,
        text_prompt: str,
        quality_preset: str,
        confidence_threshold: float,
        feather_radius: int,
        temporal_smoothing: bool,
        export_format: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, Optional[str]]:
        """
        Process video with full pipeline.

        Args:
            video_file: Path to uploaded video
            text_prompt: Text description for segmentation
            quality_preset: Quality preset name
            confidence_threshold: Segmentation confidence threshold
            feather_radius: Edge feathering radius
            temporal_smoothing: Whether to apply temporal smoothing
            export_format: Export format type
            progress: Gradio progress tracker

        Returns:
            Tuple of (status_message, output_path, preview_path)
        """
        try:
            # Check if model is loaded
            if not self.model_loaded:
                return "❌ Please load the model first", None, None

            # Validate inputs
            progress(0, desc="Validating inputs...")
            self.video_validator.validate_video_file(video_file)
            self.video_validator.validate_text_prompt(text_prompt)
            self.video_validator.validate_confidence_threshold(confidence_threshold)

            # Extract frames
            progress(0.1, desc="Extracting frames...")
            frames, metadata = self.video_processor.extract_frames(video_file)
            self.current_frames = frames
            logger.info(f"Extracted {len(frames)} frames")

            # Segment frames
            progress(0.2, desc=f"Segmenting {len(frames)} frames...")
            masks, scores = self.inference_engine.segment_frame_sequence(
                frames,
                text_prompt,
                confidence_threshold,
                progress_callback=lambda cur, tot: progress(
                    0.2 + (cur / tot) * 0.5,
                    desc=f"Segmenting frame {cur}/{tot}"
                )
            )

            # Apply temporal smoothing if enabled
            if temporal_smoothing:
                progress(0.7, desc="Applying temporal smoothing...")
                masks = self.temporal_smoother.smooth(masks, method='median')

            # Refine masks
            progress(0.8, desc="Refining masks...")
            refined_masks = [
                self.mask_refiner.refine(
                    mask,
                    feather_radius=feather_radius,
                    remove_holes=True,
                    smooth_edges=True
                )
                for mask in masks
            ]
            self.current_masks = refined_masks

            # Export
            progress(0.9, desc="Exporting results...")
            output_dir = Path("output") / Path(video_file).stem
            output_path = self.exporter.export(
                refined_masks,
                str(output_dir),
                format_type=export_format,
                video_metadata=metadata,
                create_archive=True
            )

            # Create preview
            progress(0.95, desc="Creating preview...")
            preview_path = self._create_preview(frames, refined_masks, output_dir)

            # Calculate average confidence
            avg_confidence = np.mean(scores) if scores else 0.0

            progress(1.0, desc="Done!")
            status_msg = (
                f"✅ Processing complete!\n"
                f"Frames processed: {len(frames)}\n"
                f"Average confidence: {avg_confidence:.2%}\n"
                f"Output: {output_path}"
            )

            return status_msg, output_path, preview_path

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return f"❌ Error: {str(e)}", None, None

    def _create_preview(
        self,
        frames: List[Image.Image],
        masks: List[np.ndarray],
        output_dir: Path
    ) -> Optional[str]:
        """
        Create preview image showing results.

        Args:
            frames: List of original frames
            masks: List of mask arrays
            output_dir: Output directory

        Returns:
            Path to preview image or None
        """
        try:
            # Convert PIL images to numpy arrays
            frame_arrays = [np.array(frame) for frame in frames]

            # Create side-by-side preview
            preview_path = output_dir / "preview.png"
            self.exporter.export_side_by_side(
                masks,
                frame_arrays,
                str(preview_path),
                num_samples=6
            )

            return str(preview_path)

        except Exception as e:
            logger.error(f"Failed to create preview: {e}")
            return None

    def apply_preset(self, preset_name: str) -> Tuple[float, int, bool]:
        """
        Apply quality preset settings.

        Args:
            preset_name: Name of preset

        Returns:
            Tuple of (confidence_threshold, feather_radius, temporal_smoothing)
        """
        presets = {
            'speaker_isolation': (0.6, 8, True),
            'product_demo': (0.5, 5, True),
            'custom': (0.5, 5, False)
        }

        return presets.get(preset_name, presets['custom'])


def create_ui(app: PromptMaskApp) -> gr.Blocks:
    """
    Create Gradio UI.

    Args:
        app: PromptMaskApp instance

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="VFX Buddy - PromptMask") as interface:
        gr.Markdown("""
        # 🎬 VFX Buddy - PromptMask

        Upload a video and describe what you want to segment using natural language.
        The app will process each frame and export production-ready masks.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Model Loading
                gr.Markdown("### 1️⃣ Load Model")
                load_btn = gr.Button("🚀 Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded - Click 'Load Model' to start",
                    interactive=False,
                    lines=2
                )

                # Video Input
                gr.Markdown("### 2️⃣ Upload Video")
                video_input = gr.Video(
                    label="Video File",
                    sources=["upload"]
                )

                # Text Prompt
                gr.Markdown("### 3️⃣ Describe What to Segment")
                text_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="e.g., 'person' or 'product on table'",
                    lines=2
                )

                # Quality Presets
                gr.Markdown("### 4️⃣ Select Preset")
                with gr.Row():
                    speaker_btn = gr.Button("👤 Speaker Isolation")
                    product_btn = gr.Button("📦 Product Demo")
                    custom_btn = gr.Button("⚙️ Custom")

                # Advanced Options
                with gr.Accordion("Advanced Options", open=False):
                    confidence_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.50,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum confidence for segmentation (0.5 recommended)"
                    )

                    feather_radius = gr.Slider(
                        minimum=0,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Feather Radius",
                        info="Edge softness in pixels"
                    )

                    temporal_smoothing = gr.Checkbox(
                        value=True,
                        label="Temporal Smoothing",
                        info="Reduce flicker between frames"
                    )

                    export_format = gr.Radio(
                        choices=['universal_png', 'png_sequence', 'video'],
                        value='universal_png',
                        label="Export Format",
                        info="Output format type"
                    )

                # Process Button
                process_btn = gr.Button("🎬 Process Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                gr.Markdown("### 📤 Results")
                status_output = gr.Textbox(
                    label="Status",
                    lines=6,
                    interactive=False
                )

                output_file = gr.File(
                    label="Download Output",
                    interactive=False
                )

                preview_image = gr.Image(
                    label="Preview (Sample Frames)",
                    type="filepath"
                )

        # Event Handlers
        load_btn.click(
            fn=app.load_model,
            outputs=model_status
        )

        # Preset buttons
        speaker_btn.click(
            fn=lambda: app.apply_preset('speaker_isolation'),
            outputs=[confidence_threshold, feather_radius, temporal_smoothing]
        )

        product_btn.click(
            fn=lambda: app.apply_preset('product_demo'),
            outputs=[confidence_threshold, feather_radius, temporal_smoothing]
        )

        custom_btn.click(
            fn=lambda: app.apply_preset('custom'),
            outputs=[confidence_threshold, feather_radius, temporal_smoothing]
        )

        # Process button
        process_btn.click(
            fn=app.process_video,
            inputs=[
                video_input,
                text_prompt,
                gr.State("custom"),  # preset name
                confidence_threshold,
                feather_radius,
                temporal_smoothing,
                export_format
            ],
            outputs=[status_output, output_file, preview_image]
        )

        # Examples
        gr.Markdown("""
        ### 💡 Example Prompts
        - "person" - Segment people in the video
        - "product on table" - Segment a product demo
        - "hand holding object" - Segment hands with objects
        - "car" or "yellow school bus" - Descriptive phrases work great!
        """)

        gr.Markdown("""
        ### 📋 Notes
        - First run will take a moment to initialize (cached after that)
        - Supports MP4, AVI, MOV, MKV, WebM formats
        - Maximum 300 frames per video (~10 seconds at 30fps)
        """)

    return interface


def main():
    """Main application entry point."""
    # Create application
    app = PromptMaskApp()

    # Create UI
    interface = create_ui(app)

    # Launch without auth - access controlled via HuggingFace Space visibility
    # Set Space to private, make public only during demos
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
